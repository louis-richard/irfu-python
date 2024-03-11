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


def ts_spectr(time, ener, data, comp_name: str = "energy", attrs: dict = None):
    r"""Create a time series containing a spectrum

    Parameters
    ----------
    time : numpy.ndarray
        Array of times.
    ener : numpy.ndarray
        Y value of the spectrum (energies, frequencies, etc.)
    data : numpy.ndarray
        Data of the spectrum.
    attrs : dict, Optional
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        Time series of a spectrum

    """

    # Check input type
    assert isinstance(time, np.ndarray), "time must be a numpy.ndarray"
    assert isinstance(ener, np.ndarray), "time must be a numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be a numpy.ndarray"

    # Check input shape must be (n, m, )
    assert data.ndim == 2, "Input must be a spectrum"
    assert len(time) == data.shape[0], "len(time) and data.shape[0] must be equal"
    assert len(ener) == data.shape[1], "len(ener) and data.shape[1] must be equal"

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    out = xr.DataArray(data, coords=[time, ener], dims=["time", comp_name], attrs=attrs)
    out.attrs["TENSOR_ORDER"] = 0

    return out
