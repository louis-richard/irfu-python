#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
import numpy as np
from xarray import DataArray, Dataset

# Local imports
from pyrfu._core.time import _ns


def calc_dt(inp: Dataset | DataArray) -> float:
    r"""Compute time step of the input time series.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the input variable.

    Returns
    -------
    float
        Time step in seconds, the median of the time steps; NaN for fewer
        than 2 samples (undefined).

    Raises
    ------
    TypeError
        If `inp` is not a DataArray or a Dataset, or if its times are not
        datetime64.

    """
    # Check input type
    if not isinstance(inp, (Dataset, DataArray)):
        raise TypeError("Input must be a time series")

    # Times as datetime64 (other numbers were read as nanoseconds and
    # truncated: a time step of 0, inf Hz, for times in seconds)
    if not np.issubdtype(inp.time.dtype, np.datetime64):
        raise TypeError(f"the times must be datetime64, not {inp.time.dtype}")

    # One sample: no time step (and no warning from the median of nothing)
    if len(inp.time) < 2:
        return np.nan

    # Median time step, from the exact steps in ns
    dt_s = float(np.median(np.diff(_ns(inp.time.data)))) / 1e9

    return dt_s
