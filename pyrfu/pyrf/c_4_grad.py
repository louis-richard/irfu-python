#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""c_4_grad.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from .resample import resample
from .c_4_k import c_4_k
from .normalize import normalize
from .avg_4sc import avg_4sc
from .dot import dot
from .cross import cross


def c_4_grad(r_list, b_list, method="grad"):
    """Calculate gradient of physical field using 4 spacecraft technique
    in [2]_ [3]_.

    Parameters
    ----------
    r_list : list of xarray.DataArray
        Time series of the positions of the spacecraft

    b_list : list of xarray.DataArray
        Time series of the magnetic field at the corresponding positions

    method : str
        Method flag :
            * "grad" : compute gradient (default)
            * "div" : compute divergence
            * "curl" : compute curl
            * "bdivb" : compute b.div(b)
            * "curv" : compute curvature

    Returns
    -------
    out : xarray.DataArray
        Time series of the derivative of the input field corresponding to
        the method

    See also
    --------
    pyrfu.pyrf.c_4_k : Calculates reciprocal vectors in barycentric
                        coordinates.

    References
    ----------
    .. [2]	Dunlop, M. W., A. Balogh, K.-H. Glassmeier, and P. Robert (2002a),
            Four-point Cluster application of	magnetic field analysis
            tools: The Curl- ometer, J. Geophys. Res., 107(A11), 1384,
            doi : https://doi.org/10.1029/2001JA005088.

    .. [3]	Robert, P., et al. (1998), Accuracy of current determination, in
            Analysis Methods for Multi-Spacecraft Data, edited by G.
            Paschmann and P. W. Daly, pp. 395–418, Int. Space Sci. Inst.,
            Bern. doi : http://www.issibern.ch/forads/sr-001-16.pdf

    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Load magnetic field and spacecraft position

    >>> b_mms = [get_data("B_gse_fgm_srvy_l2", tint, i) for i in range(1, 5)]
    >>> r_mms = [get_data("R_gse", tint, i) for i in range(1, 5)]
    >>> gradb = pyrf.c_4_grad(r_mms, b_mms, "grad")

    """

    # Resample with respect to 1st spacecraft
    r_list = [resample(r, b_list[0]) for r in r_list]
    b_list = [resample(b, b_list[0]) for b in b_list]

    # Compute reciprocal vectors in barycentric coordinates (see c_4_k)
    k_list = c_4_k(r_list)

    # Magnetic field at the center of mass of the tetrahedron
    b_avg = avg_4sc(b_list)

    b_dict = {"1": b_list[0], "2": b_list[1], "3": b_list[2], "4": b_list[3]}
    k_dict = {"1": k_list[0], "2": k_list[1], "3": k_list[2], "4": k_list[3]}

    mms_list = b_dict.keys()

    # Gradient of scalar/vector
    if len(b_dict["1"].shape) == 1:
        grad_b = np.zeros((len(b_dict["1"]), 3))

        for mms_id in mms_list:
            grad_b += k_dict[mms_id].data \
                      * np.tile(b_dict[mms_id].data, (3, 1)).T

    else:
        grad_b = np.zeros((len(b_dict["1"]), 3, 3))

        for i in range(3):
            for j in range(3):
                for mms_id in mms_list:
                    grad_b[:, j, i] += k_dict[mms_id][:, i].data \
                                       * b_dict[mms_id][:, j].data

    # Gradient
    if method.lower() == "grad":
        out_data = grad_b

    # Divergence
    elif method.lower() == "div":
        out_data = np.zeros(len(b_dict["1"]))

        for mms_id in mms_list:
            out_data += dot(k_dict[mms_id], b_dict[mms_id]).data

    # Curl
    elif method.lower() == "curl":
        out_data = np.zeros((len(b_dict["1"]), 3))

        for mms_id in mms_list:
            out_data += cross(k_dict[mms_id], b_dict[mms_id]).data

    # B.div(B)
    elif method.lower() == "bdivb":
        out_data = np.zeros(b_avg.shape)

        for i in range(3):
            out_data[:, i] = np.sum(b_avg.data * grad_b[:, i, :], axis=1)

    # Curvature
    elif method.lower() == "curv":
        b_hat_list = [normalize(b) for b in b_list]

        out_data = c_4_grad(r_list, b_hat_list, method="bdivb").data

    else:
        raise ValueError("Invalid method")

    if len(out_data.shape) == 1:
        out = xr.DataArray(out_data, coords=[b_dict["1"].time], dims=["time"])

    elif len(out_data.shape) == 2:
        out = xr.DataArray(out_data,
                           coords=[b_dict["1"].time, ["x", "y", "z"]],
                           dims=["time", "comp"])

    elif len(out_data.shape) == 3:
        out = xr.DataArray(out_data,
                           coords=[b_dict["1"].time, ["x", "y", "z"],
                                   ["x", "y", "z"]],
                           dims=["time", "vcomp", "hcomp"])

    else:
        raise TypeError("Invalid type")

    return out
