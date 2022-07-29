#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def calc_fs(inp):
    r"""Computes the sampling frequency of the input time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : float
        Sampling frequency in Hz.

    """

    out = 1 / (np.median(np.diff(inp.time.data)).astype(float) * 1e-9)

    return out
