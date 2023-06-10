#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def start(inp):
    r"""Gives the first time of the time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series.

    Returns
    -------
    out : float
        Value of the first time in the desired format.

    """

    out = inp.time.data[0].astype(np.int64) * 1e-9

    return out
