#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t_eval.py

@author : Louis RICHARD
"""

import bisect
import numpy as np
import xarray as xr


def t_eval(inp=None, t=None):
    """Evaluates the input time series at the target time.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series if the input to evaluate.

    t : numpy.ndarray
        Times at which the input will be evaluated.

    Returns
    -------
    out : DataArray
        Time series of the input at times t.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)
    assert t is not None and isinstance(t, np.ndarray)

    idx = np.zeros(len(t))

    for i, time in enumerate(t):
        idx[i] = bisect.bisect_left(inp.time.data, time)

    idx = idx.astype(int)

    if inp.ndim == 2:
        out = xr.DataArray(inp.data[idx, :], coords=[t, inp.comp], dims=["time", "comp"])
    else:
        out = xr.DataArray(inp.data[idx], coords=[t], dims=["time"])

    return out
