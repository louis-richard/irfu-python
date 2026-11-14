#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .unix2datetime64 import unix2datetime64


def ts_time(time, attrs: dict | None = None):
    r"""Creates time line in DataArray.

    Parameters
    ----------
    time : ndarray
        Input time line: Unix times in seconds (any float type, rounded to
        the nanosecond), or numpy.datetime64 in any unit.
    attrs : dict, Optional
        Attributes of the output.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time line, in datetime64[ns].

    Raises
    ------
    TypeError
        If time is not a numpy.ndarray of floats or numpy.datetime64.

    """

    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if np.issubdtype(time.dtype, np.floating):
        # Rounded to the ns (6.5e-5 * 1e9 = 64999.99999999999 was truncated)
        time = unix2datetime64(time.astype(np.float64))
    elif np.issubdtype(time.dtype, np.datetime64):
        time = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be in unix (float) or numpy.datetime64")

    out = xr.DataArray(time, coords=[time], dims=["time"], attrs=attrs)

    return out
