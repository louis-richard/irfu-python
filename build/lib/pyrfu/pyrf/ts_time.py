#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts_time.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from astropy.time import Time


def ts_time(t=None, fmt="unix"):
    """Creates time line in DataArray.

    Parameters
    ----------
    t : numpy.ndarray
        Input time line.

    fmt : str
        Format of the input time line.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time line.

    """

    assert t is not None and isinstance(t, np.ndarray)

    t = Time(t, format=fmt).datetime64

    out = xr.DataArray(t, coords=[t], dims=["time"])

    return out
