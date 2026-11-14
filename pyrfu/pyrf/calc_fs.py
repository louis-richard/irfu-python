#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
import numpy as np
from xarray import DataArray, Dataset

# Local imports
from .calc_dt import calc_dt


def calc_fs(inp: Dataset | DataArray) -> float:
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

    # The inverse of the median time step (NaN for one sample, inf for a zero
    # step)
    return 1 / np.float64(calc_dt(inp))
