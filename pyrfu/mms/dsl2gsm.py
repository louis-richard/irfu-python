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

"""dsl2gse.py
©author: Louis Richard
"""

import warnings
import numpy as np
import xarray as xr

from astropy.time import Time

from ..pyrf import (cotrans, resample,
                    sph2cart, ts_vec_xyz)


def dsl2gsm(inp, defatt, direction=1):
    """Transform time series from DSL to GSE.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series to convert.

    defatt : xarray.Dataset or ndarray or list
        Spacecraft attitude.

    direction : int
        Direction of tranformation. +1 DSL -> GSE, -1 GSE -> DSL.
        Default is 1.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input field in the new coordinates systems.


    Examples
    --------
    >>> from pyrfu.mms import get_data, load_ancillary, dsl2gse

    Define time interval

    >>> tint = ["2015-05-09T14:00:000", "2015-05-09T17:59:590"]

    Load magentic field in spacecraft coordinates

    >>> b_xyz = get_data("b_dmpa_fgm_brst_l2", tint, 1)

    Load spacecraft attitude

    >>> defatt = load_ancillary("defatt", tint, 1)

    Transform magnetic field to GSE

    >>> b_gse = dsl2gse(b_xyz, defatt)

    """

    if direction not in [-1, 1]:
        warnings.warn("using GSE->DSL", UserWarning)

    if isinstance(defatt, xr.Dataset):
        x, y, z = sph2cart(np.deg2rad(defatt.z_ra.data),
                           np.deg2rad(defatt.z_dec), 1)
        sax_gei = np.transpose(np.vstack([defatt.time.data.view("i8") * 1e-9,
                                          x, y, z]))
        sax_gse = cotrans(sax_gei, "gei>gsm")
        sax_gse = ts_vec_xyz(Time(sax_gse[:, 0], format="unix").datetime64,
                             sax_gse[:, 1:])

        spin_ax_gse = resample(sax_gse, inp)
        spin_axis = spin_ax_gse.data

    elif isinstance(defatt, (np.ndarray, list)) and len(defatt) == 3:
        spin_axis = defatt

    else:
        raise ValueError("unrecognized DEFATT/SAX input")

    r_x, r_y, r_z = [spin_axis[:, i] for i in range(3)]

    a = 1. / np.sqrt(r_y ** 2 + r_z ** 2)
    transf_mat = np.zeros((len(a), 3, 3))
    transf_mat[:, 0, :] = np.transpose(np.stack([a * (r_y ** 2 + r_z ** 2),
                                                 -a * r_x * r_y,
                                                 -a * r_x * r_z]))

    transf_mat[:, 1, :] = np.transpose(np.stack([0. * a, a * r_z, -a * r_y]))

    transf_mat[:, 2, :] = np.transpose(np.stack([r_x, r_y, r_z]))

    if direction == 1:
        transf_mat = np.transpose(transf_mat, [0, 2, 1])

    out_data = np.einsum('kji,ki->kj', transf_mat, inp.data)

    out = inp.copy()
    out.data = out_data
    out.attrs["COORDINATE_SYSTEM"] = "GSM"

    return out
