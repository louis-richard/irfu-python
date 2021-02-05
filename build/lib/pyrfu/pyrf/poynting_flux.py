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

"""poynting_flux.py
@author: Louis Richard
"""

import numpy as np

from astropy.time import Time

from .calc_fs import calc_fs
from .cross import cross
from .dot import dot
from .normalize import normalize
from .resample import resample
from .time_clip import time_clip


def poynting_flux(e_xyz, b_xyz, b_hat):
    """
    Estimates Poynting flux at electric field sampling as

    .. math::

        \\mathbf{S} = \\frac{\\mathbf{E}\\times\\mathbf{B}}{\\mu_0}

    if `b0` is given project the Poynting flux along `b0`


    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.

    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    b_hat : xarray.DataArray, optional
        Time series of the direction to project the Pointing flux.

    Returns
    -------
    s : xarray.DataArray
        Time series of the Pointing flux.

    s_z : xarray.DataArray
        Time series of the projection of the Pointing flux
        (only if b0).

    int_s : xarray.DataArray
        Time series of the time integral of the Pointing flux
        (if b0 integral along b0).

    """

    # check which Poynting flux to calculate
    flag_s_z, flag_int_s_z, flag_int_s = [False, False, False]

    if b_hat is None:
        flag_int_s = True
    else:
        flag_s_z, flag_int_s_z = [True, True]

    # interval where both E & B exist
    tint = [Time(max([min(e_xyz.time.data), min(b_xyz.time.data)]),
                 format="datetime64").iso,
            Time(min([max(e_xyz.time.data), max(b_xyz.time.data)]),
                 format="datetime64").iso]

    e_xyz, b_xyz = [time_clip(e_xyz, tint), time_clip(b_xyz, tint)]

    if len(e_xyz) < len(b_xyz):
        e_xyz = resample(e_xyz, b_xyz)
        f_spl = calc_fs(b_xyz)
    elif len(e_xyz) > len(b_xyz):
        b_xyz = resample(b_xyz, e_xyz)
        f_spl = calc_fs(e_xyz)
    else:
        f_spl = calc_fs(b_xyz)

    # Calculate Poynting flux
    s_xyz = cross(e_xyz, b_xyz) / (4 * np.pi / 1e7) * 1e-9

    if flag_s_z:
        b_m = resample(b_hat, e_xyz)
        s_z = dot(normalize(b_m), s_xyz)
    else:
        s_z = None

    # time integral of Poynting flux along ambient magnetic field
    res = None

    if flag_int_s_z:
        s_z[np.isnan(s_z.data)] = 0  # set to zero points where Sz=NaN

        int_s_z = np.cumsum(s_z) / f_spl

        res = (s_xyz, s_z, int_s_z)

    if flag_int_s:  # time integral of all Poynting flux components
        # set to zero points where Sz=NaN
        s_xyz[np.isnan(s_xyz[:, 2].data)] = 0

        int_s = np.cumsum(s_xyz) / f_spl

        res = (s_xyz, int_s)

    return res
