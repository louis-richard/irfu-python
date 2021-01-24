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

import numpy as np

from . import resample, ts_scalar, ts_vec_xyz


def edb(e, b0, angle_lim=20, flag_method="E.B=0"):
    """Compute Ez under assumption :math:`\\mathbf{E}.\\mathbf{B}=0` or
    :math:`\\mathbf{E}.\\mathbf{B} \\approx 0`

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field.

    b0 : xarray.DataArray
        Time series of the background magnetic field.

    angle_lim : float
        B angle with respect to the spin plane should be less than angle_lim degrees otherwise Ez
        is set to 0.

    flag_method : str
        Assumption on the direction of the measured electric field :
            "e.b=0" :  :math:`\\mathbf{E}.\\mathbf{B}=0`.
            "e_par" :  :math:`\\mathbf{E}` field along the B projection is coming from parallel
            electric field.
            "e_perp+nan" : to fill.


    Returns
    -------
    ed : xarray.DataArray
        Time series of the electric field output.

    d : xarray.DataArray
        Time series of the B elevation angle above spin plane.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = 1

    Load magnetic field, electric field and spacecraft position

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute Ez

    >>> e_z, alpha = pyrf.edb(e_xyz, b_xyz)

    """

    default_value = 0
    if flag_method.lower() == "e_perp+nan":
        default_value = np.nan

        flag_method = "e.b=0"

    if len(b0) != len(e):
        b0 = resample(b0, e)

    bd = b0.data
    ed = e.data
    ed[:, -1] *= default_value

    if flag_method.lower() == "e.b=0":
        # Calculate using assumption E.B=0
        d = np.arctan2(bd[:, 2], np.sqrt(bd[:, 0] ** 2 + bd[:, 1] ** 2)) * 180 / np.pi

        ind = np.abs(d) > angle_lim

        if True in ind:
            ed[ind, 2] = -(ed[ind, 0] * bd[ind, 0] + ed[ind, 1] * bd[ind, 1]) / bd[ind, 2]

    elif flag_method.lower() == "e_par":
        # Calculate using assumption that E field along the B projection is coming from parallel
        # electric field
        d = np.arctan2(bd[:, 2], np.sqrt(bd[:, 0] ** 2 + bd[:, 1] ** 2)) * 180 / np.pi

        ind = np.abs(d) < angle_lim

        if True in ind:
            ed[ind, 2] = (ed[ind, 0] * bd[ind, 0] + ed[ind, 1] * bd[ind, 1])
            ed[ind, 2] = ed[ind, 2] * bd[ind, 2] / (bd[ind, 0] ** 2 + bd[ind, 1] ** 2)

    else:
        raise ValueError("Invalid flag")

    ed = ts_vec_xyz(e.time.data, ed, {"UNITS": e.attrs["UNITS"]})
    d = ts_scalar(e.time.data, d, {"UNITS": "degrees"})

    return ed, d
