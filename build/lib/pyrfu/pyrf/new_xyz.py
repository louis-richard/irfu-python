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

"""new_xyz.py
@author: Louis Richard
"""

import xarray as xr
import numpy as np


def new_xyz(inp, trans_mat):
    """Transform the input field to the new frame.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field in the original coordinate system.

    trans_mat : ndarray
        Transformation matrix.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input in the new frame.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute MVA frame

    >>> b_lmn, l, mva = pyrf.mva(b_xyz)

    Move electric field to the MVA frame

    >>> e_lmn = pyrf.new_xyz(e_xyz, mva)

    """

    if inp.data.ndim == 3:
        out_data = np.matmul(np.matmul(trans_mat.T, inp.data), trans_mat)
    else:
        out_data = (trans_mat.T @ inp.data.T).T

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims,
                       attrs=inp.attrs)

    return out
