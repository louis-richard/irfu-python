#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
histogram.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def histogram(inp=None, bins=100, normed=True):
    """
    Computes 1D histogram of the inp with bins bins

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input scalar variable.

    bins : int
        Number of bins.

    normed : bool
        Normalize the PDF.

    Returns
    -------
    out : xarray.DataArray
        1D distribution of the input time series.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)

    hist, bins = np.histogram(inp.data, bins=bins, normed=normed)
    bin_center = (bins[1:] + bins[:-1]) * 0.5

    out = xr.DataArray(hist, coords=[bin_center], dims=["bins"])

    return out
