#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
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

    Right now, this has only been tested for calculating moving averages of 1D time
    series.

    The convolution is done using scipy.ndimage.convolve (read documentation for
    scipy.ndimage.convolve for more information).

    Parameters
    ----------
    inp : xarray.DataArray
        The time series to be convolved with the kernel (scalar, or vector convolved
        component by component). Integer data are converted to float64.
    kernel : numpy.ndarray
        The kernel to apply to inp for the convolution.
    mode : str, Optional
        How the edges are extended (see scipy.ndimage.convolve). Default is
        "nearest".

    Returns
    -------
    out : xarray.DataArray
        An array containing the convolution of inp with kernel, with the shape,
        coordinates and attributes of inp.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If inp has no "time" dimension or more than 2 dimensions.

    """
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Invalid input type. Input must be xarray.DataArray")

    if "time" not in inp.dims:
        message = "Invalid input dimensions. Input must have a 'time' dimension"
        raise ValueError(message)

    if not 0 < inp.data.ndim <= 2:
        raise ValueError("Invalid dimensions.")

    # scipy returns the type of the input: an average of integers is truncated
    data = inp.data
    if not np.issubdtype(data.dtype, np.inexact):
        data = data.astype(np.float64)

    if data.ndim == 2:
        convolution = np.stack(
            [
                scipy.ndimage.convolve(input=column, weights=kernel, mode=mode)
                for column in data.T
            ],
            axis=1,
        )
    else:
        convolution = scipy.ndimage.convolve(input=data, weights=kernel, mode=mode)

    out = inp.copy(data=convolution)

    return out
