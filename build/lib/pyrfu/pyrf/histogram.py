#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def histogram(inp, bins: int = 100, normed: bool = True):
    r"""Computes 1D histogram of the inp with bins bins

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input scalar variable.

    bins : int, Optional
        Number of bins. Default is 100.

    normed : bool, Optional
        Normalize the PDF. Default is True.

    Returns
    -------
    out : xarray.DataArray
        1D distribution of the input time series.

    """

    hist, bins = np.histogram(inp.data, bins=bins, normed=normed)
    bin_center = (bins[1:] + bins[:-1]) * 0.5

    out = xr.DataArray(hist, coords=[bin_center], dims=["bins"])

    return out
