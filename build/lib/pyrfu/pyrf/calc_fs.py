#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_fs.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def calc_fs(inp=None):
    """Computes the sampling frequency of the input time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : float
        Sampling frequency in Hz.

    """

    if inp is None:
        raise ValueError("calc_dt requires at least one argument")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    out = 1 / (np.median(np.diff(inp.time.data)).astype(float) * 1e-9)

    return out
