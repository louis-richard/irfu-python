#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""histogram.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def histogram(inp, bins=100, normed=True):
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

    hist, bins = np.histogram(inp.data, bins=bins, normed=normed)
    bin_center = (bins[1:] + bins[:-1]) * 0.5

    out = xr.DataArray(hist, coords=[bin_center], dims=["bins"])

    return out
