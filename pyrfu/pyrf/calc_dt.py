#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_dt.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def calc_dt(inp=None):
    """Computes time step of the input time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : float
        Time step in seconds.

    """

    if inp is None:
        raise ValueError("calc_dt requires at least one argument")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    out = np.median(np.diff(inp.time.data)).astype(float) * 1e-9

    return out
