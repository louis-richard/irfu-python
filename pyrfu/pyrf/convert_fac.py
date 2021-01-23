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
import xarray as xr

from . import resample, ts_vec_xyz, calc_fs


def convert_fac(inp, b_bgd, r=None):
    """
    Transforms to a field-aligned coordinate (FAC) system defined as :
        * R_parallel_z aligned with the background magnetic field
        * R_perp_y defined by R_parallel cross the position vector of the spacecraft (nominally
            eastward at the equator)
        * R_perp_x defined by R_perp_y cross R_par

    If inp is one vector along r direction, out is inp[perp, para] projection.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field.

    b_bgd : xarray.DataArray
        Time series of the background magnetic field.

    r : xarray.DataArray or ndarray or list
        Position vector of spacecraft.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input field in field aligned coordinates system.

    Notes
    -----
    All input parameters must be in the same coordinate system.

    Examples
    --------
    >>> import numpy
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field (FGM) and electric field (EDP)

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Convert to field aligned coordinates

    >>> e_xyzfac = pyrf.convert_fac(e_xyz, b_xyz, numpy.array([1, 0, 0]))

    """

    if r is None:
        r = np.array([1, 0, 0])

    if len(inp) != len(b_bgd):
        options = dict(fs=calc_fs(inp))
        b_bgd = resample(b_bgd, inp, **options)

    t, inp_data = [inp.time.data, inp.data]

    # Normalize background magnetic field
    bn = b_bgd / np.linalg.norm(b_bgd, axis=1, keepdims=True)

    if isinstance(r, (list, np.ndarray)) and len(r) == 3:
        r = np.tile(r, (len(b_bgd), 1))
    elif isinstance(r, xr.DataArray):
        r = resample(r, b_bgd)
    else:
        raise TypeError("Invalid type of spacecraft position")

    # Parallel
    r_par = bn

    # Perpendicular
    r_perp_y = np.cross(r_par, r, axis=1)
    r_perp_y /= np.linalg.norm(r_perp_y, axis=1, keepdims=True)
    r_perp_x = np.cross(r_perp_y, b_bgd, axis=1)
    r_perp_x /= np.linalg.norm(r_perp_x, axis=1, keepdims=True)

    n_data, n_dim = inp_data.shape

    if n_dim == 3:
        out_data = np.zeros(inp.shape)

        out_data[:, 0] = np.sum(r_perp_x * inp_data, axis=1)
        out_data[:, 1] = np.sum(r_perp_y * inp_data, axis=1)
        out_data[:, 2] = np.sum(r_par * inp_data, axis=1)

        # xarray
        out = xr.DataArray(out_data, coords=[t, inp.comp], dims=["time", "comp"])

    elif n_dim == 1:
        out_data = np.zeros([3, n_data])

        out_data[:, 0] = inp[:, 0] * (
                    r_perp_x[:, 0] * r[:, 0] + r_perp_x[:, 1] * r[:, 1] + r_perp_x[:, 2] * r[:, 2])
        out_data[:, 1] = inp[:, 0] * (
                    r_par[:, 0] * r[:, 0] + r_par[:, 1] * r[:, 1] + r_par[:, 2] * r[:, 2])

        out = ts_vec_xyz(t, out_data, attrs=inp.attrs)
    else:
        raise TypeError("Invalid dimension of inp")

    return out
