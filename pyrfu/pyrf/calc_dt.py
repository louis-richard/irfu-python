#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from ._time import _ns


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

    # Median time step, from the exact steps in ns
    dt_s = float(np.median(np.diff(_ns(inp.time.data)))) / 1e9

    return dt_s
