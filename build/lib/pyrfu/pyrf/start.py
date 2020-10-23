#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
start.py

@author : Louis RICHARD
"""

import xarray as xr
from astropy.time import Time


def start(inp=None):
    """Gives the first time of the time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series.

    Returns
    -------
    out : float or str
        Value of the first time in the desired format.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)

    out = Time(inp.time.data[0], format="datetime64").unix
    return out
