#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset


def calc_dt(inp: Union[Dataset, DataArray]) -> float:
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

    """
    # Check input type
    if not isinstance(inp, (Dataset, DataArray)):
        raise TypeError("Input must be a time series")

    # One sample: no time step (and no warning from the median of nothing)
    if len(inp.time) < 2:
        return np.nan

    # Convert time to nanoseconds
    time_datetime64 = inp.time.data.astype("datetime64[ns]")
    time_ns = time_datetime64.astype(np.int64)

    # Calculate the sampling frequency
    dt_s = 1e-9 * float(np.median(np.diff(time_ns)))

    return dt_s
