#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import datetime

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.14"
__status__ = "Prototype"


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
    out : xarray.DataArray
        Time series of the time clipped input.

    """

    if isinstance(inp, xr.Dataset):
        coords_data = [inp[k] for k in filter(lambda x: x != "time", inp.dims)]
        coords_data = [time_clip(inp.time, tint), *coords_data]
        out_dict = {dim: coords_data[i] for i, dim in enumerate(inp.coords)}

        for k in inp:
            if "time" in list(inp[k].coords):
                data = time_clip(inp[k], tint)
                out_dict[k] = (list(inp[k].dims), data.data)
            else:
                out_dict[k] = (list(inp[k].dims), inp[k].data)

        out = xr.Dataset(out_dict)
        out.attrs = inp.attrs
        return out

    if isinstance(tint, xr.DataArray):
        t_start, t_stop = tint.time.data[[0, -1]]

    elif isinstance(tint, np.ndarray):
        if isinstance(tint[0], datetime.datetime) \
                and isinstance(tint[-1], datetime.datetime):
            t_start, t_stop = [tint.time[0], tint.time[-1]]

        else:
            raise TypeError('Values must be in Datetime64')

    elif isinstance(tint, list):
        t_start, t_stop = iso86012datetime64(np.array(tint))

    else:
        raise TypeError("invalid tint")

    idx_min = bisect.bisect_left(inp.time.data, np.datetime64(t_start))
    idx_max = bisect.bisect_right(inp.time.data, np.datetime64(t_stop))

    coord = [inp.time.data[idx_min:idx_max]]

    if len(inp.coords) > 1:
        for k in inp.dims[1:]:
            coord.append(inp.coords[k].data)

    out = xr.DataArray(inp.data[idx_min:idx_max, ...], coords=coord,
                       dims=inp.dims, attrs=inp.attrs)

    out.time.attrs = inp.time.attrs

    return out
