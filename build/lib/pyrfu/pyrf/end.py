#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
end.py

@author : Louis RICHARD
"""

import xarray as xr
from astropy.time import Time


def end(inp=None):
    """Gives the last time of the time series in unix format.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : float or str
        Value of the last time in the desired format.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)

    out = Time(inp.time.data[-1], format="datetime64").unix

    return out
