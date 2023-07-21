#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def calc_fs(inp):
    r"""Computes the sampling frequency of the input time series.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Time series of the input variable.

    Returns
    -------
    out : float
        Sampling frequency in Hz.

    """

    message = "Input must be a time series"
    assert isinstance(inp, (xr.Dataset, xr.DataArray)), message

    out = 1 / (np.median(np.diff(inp.time.data)).astype(np.float64) * 1e-9)

    return out
