#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def calc_dt(inp: Union[Dataset, DataArray]) -> float:
    r"""Compute time step of the input time series.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the input variable.

    Returns
    -------
    float
        Time step in seconds.

    """
    # Check input type
    if not isinstance(inp, (Dataset, DataArray)):
        raise TypeError("Input must be a time series")

    # Convert time to nanoseconds
    time_datetime64 = inp.time.data.astype(np.datetime64)
    time_ns = time_datetime64.astype(np.int64)

    # Calculate the sampling frequency
    dt_s = 1e-9 * np.median(np.diff(time_ns))

    return dt_s
