#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import xarray as xr

# Local imports
from .datetime642unix import datetime642unix


def end(inp):
    """Gives the last time of the time series in unix format.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Time series of the input variable.

    Returns
    -------
    out : float
        Value of the last time in unix format.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray or a xarray.Dataset.

    """

    if not isinstance(inp, (xr.DataArray, xr.Dataset)):
        raise TypeError("inp must be a xarray.DataArray or xarray.Dataset")

    # As start and datetime642unix
    out = datetime642unix(inp.time.data[-1:])[0]

    return out
