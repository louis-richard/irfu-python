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

"""eb_nrf.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from .resample import resample
from .dot import dot
from .normalize import normalize
from .cross import cross


def eb_nrf(e_xyz, b_xyz, v_xyz, flag=0):
    """Find E and B in MP system given B and MP normal vector.

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.

    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    v_xyz : xarray.DataArray
        Normal vector.

    flag : int or ndarray
        to fill.

    Returns
    -------
    out : xarray.DataArray
        to fill.

    """

    assert isinstance(flag, (int, np.ndarray)), "Invalid flag type"

    if isinstance(flag, int):
        flag_cases = ["a", "b"]
        flag_case = flag_cases[flag]
        l_direction = None

    else:
        assert np.size(flag) == 3
        l_direction = flag
        flag_case = "c"

    if flag_case == "a":
        b_data = resample(b_xyz, e_xyz).data

        n_l = b_data / np.linalg.norm(b_data, axis=0)[:, None]
        n_n = np.cross(np.cross(b_data, v_xyz), b_data)
        n_n = n_n / np.linalg.norm(n_n)[:, None]
        n_m = np.cross(n_n, n_l)  # in (vn x b) direction

    elif flag_case == "b":
        n_n = v_xyz / np.linalg.norm(v_xyz)
        n_m = normalize(cross(n_n, b_xyz.mean(dim="time")))
        n_l = cross(n_m, n_n)

    else:
        n_n = normalize(v_xyz)
        n_m = normalize(np.cross(n_n, l_direction))
        n_l = cross(n_m, n_n)

    # estimate e in new coordinates
    e_lmn = np.hstack([dot(e_xyz, vec) for vec in [n_l, n_m, n_n]])

    out = xr.DataArray(e_xyz.time.data, e_lmn, e_xyz.attrs)

    return out
