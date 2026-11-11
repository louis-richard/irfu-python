#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import xarray as xr

# Local imports
from .datetime642unix import datetime642unix

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
    inp : xarray.DataArray or xarray.Dataset
        Time series, or a Dataset with a time coordinate (e.g. a particle
        distribution).

    Returns
    -------
    out : float
        First time, in seconds since 1970-01-01 (unix time).

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray or a xarray.Dataset.
    ValueError
        If the first dimension of the xarray.DataArray inp is not time.

    """

    # Check input type
    if not isinstance(inp, (xr.DataArray, xr.Dataset)):
        raise TypeError("inp must be a xarray.DataArray or xarray.Dataset")

    # Make sure this is a time series
    if isinstance(inp, xr.DataArray) and list(inp.dims)[0] != "time":
        raise ValueError("inp must be a time series")

    # As end and datetime642unix (ns * 1e-9 differed from ns / 1e9 in the last
    # digit for about 40 % of the times)
    out = datetime642unix(inp.time.data[:1])[0]

    return out
