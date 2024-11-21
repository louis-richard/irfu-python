#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import scipy
import xarray as xr

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def ts_convolve(inp, kernel, mode: str = "nearest"):

    r"""
    Compute the convolution of a time series of N-dimensional data with a N-dimensional
    kernel.
    ***Right now has only been tested for calculating moving averages of 1D time series
    ***

    The convolution is done using scipy.ndimage.convolve, with mode = "nearest"
    (read documentation for scipy.ndimage.convolve for more information).

    Parameters:
    ----------
    inp : xarray.DataArray
        The time series to be convolved with the kernel.

    kernel: nd.array
        The kernel to apply to inp for the convolution.

    Returns:
    -------
    out : xarray.DataArray
        An array containing the convolution of inp with kernel. Mode "valid" is applied
        from numpy.convolve, which does not affect the edges of the time series.
    """

    convolution = scipy.ndimage.convolve(input=inp.data, weights=kernel, mode=mode)

    out = xr.DataArray(convolution, coords=inp.coords, dims=inp.dims, attrs=inp.attrs)

    return out
