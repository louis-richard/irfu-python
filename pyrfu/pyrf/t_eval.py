#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def t_eval(inp, times):
    r"""Evaluates the input time series at the target time.

    There is no interpolation: each target time takes the first sample of the
    input at or after it (the first sample for the times before the input).
    Use :func:`pyrfu.pyrf.resample` to interpolate.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series if the input to evaluate.
    times : ndarray
        Times at which the input will be evaluated.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input at times t.

    Raises
    ------
    IndexError
        If a target time is after the last sample of the input.

    """

    idx = np.zeros(len(times))

    for i, time in enumerate(times):
        idx[i] = bisect.bisect_left(inp.time.data, time)

    idx = idx.astype(np.int64)

    if inp.ndim == 2:
        out = xr.DataArray(
            inp.data[idx, :],
            coords=[times, inp.comp],
            dims=["time", "comp"],
        )
    else:
        out = xr.DataArray(inp.data[idx], coords=[times], dims=["time"])

    return out
