#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _time_interval(tint):
    if isinstance(tint, xr.DataArray):
        t_start, t_stop = tint.time.data[[0, -1]]
    elif isinstance(tint, (np.ndarray, list)):
        if isinstance(tint[0], np.datetime64):
            t_start, t_stop = tint
        elif isinstance(tint[0], str):
            t_start, t_stop = iso86012datetime64(np.array(tint))
        else:
            raise TypeError("Values must be in datetime64, or str!!")
    else:
        raise TypeError("tint must be a DataArray or array_like!!")

    return t_start, t_stop


def time_clip(inp, tint):
    r"""Time clip the input (if time interval is TSeries clip between start
    and stop).

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Time series of the quantity to clip.
    tint : xarray.DataArray or ndarray or list
        Time interval can be a time series, a array of datetime64 or a list.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the time clipped input, with the times in
        [t_start, t_stop] (inclusive). All the coordinates are kept, and the
        time dependent ones are clipped. For a Dataset, the array_like
        attributes with a first dimension the length of time are clipped too.

    """
    t_start, t_stop = _time_interval(tint)

    idx_min = np.searchsorted(inp.time.data, t_start, side="left")
    idx_max = np.searchsorted(inp.time.data, t_stop, side="right")
    out = inp.isel(time=slice(idx_min, idx_max))

    if isinstance(inp, xr.Dataset):
        # If array_like attributes have one dimension equal to time length
        # assume time dependent. One option would be move the time dependent
        # array_like attributes to time series to zVaraibles to avoid confusion
        out_attrs = {}

        for k in sorted(inp.attrs):
            attr = inp.attrs[k]

            if (
                isinstance(attr, np.ndarray)
                and attr.ndim > 0
                and attr.shape[0] == len(inp.time.data)
            ):
                out_attrs[k] = attr[idx_min:idx_max, ...]
            else:
                out_attrs[k] = attr

        # Dataset.isel shares the attributes dictionary with inp
        out.attrs = out_attrs

    return out
