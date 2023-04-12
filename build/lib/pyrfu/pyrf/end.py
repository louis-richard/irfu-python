#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def end(inp):
    """Gives the last time of the time series in unix format.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : float
        Value of the last time in unix format.

    """

    out = inp.time.data[-1].astype(np.int64) / 1e9

    return out
