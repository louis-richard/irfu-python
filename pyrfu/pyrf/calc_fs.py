#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def calc_fs(inp: Union[Dataset, DataArray]) -> float:
    r"""Computes the sampling frequency of the input time series.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the input variable.

    Returns
    -------
    f_samp : float
        Sampling frequency in Hz, the inverse of the median time step; NaN
        for fewer than 2 samples (undefined, e.g., a series averaged over an
        interval, which resample, avg_4sc, convert_fac and rotate_tensor
        accept).

    """
    # Check input type
    if not isinstance(inp, (Dataset, DataArray)):
        raise TypeError("Input must be a time series")

    # One sample: no time step (and no warning from the median of nothing)
    if len(inp.time) < 2:
        return np.nan

    # Time steps in seconds, whatever the unit of the time coordinate
    time = inp.time.data.astype(np.datetime64)
    d_t = np.diff(time) / np.timedelta64(1, "s")

    # Calculate the sampling frequency
    f_samp = 1 / np.median(d_t)

    return f_samp
