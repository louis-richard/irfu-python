#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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
                out_dict[k] = time_clip(inp[k], tint)
            else:
                out_dict[k] = inp[k]

        # Find array_like attributes
        arr_attrs = filter(
            lambda x: isinstance(inp.attrs[x], np.ndarray),
            inp.attrs,
        )
        arr_attrs = list(arr_attrs)

        # Initialize attributes dictionary with non array_like attributes
        gen_attrs = filter(lambda x: x not in arr_attrs, inp.attrs)
        out_attrs = {k: inp.attrs[k] for k in list(gen_attrs)}

        for a in arr_attrs:
            attr = inp.attrs[a]

            # If array_like attributes have one dimension equal to time
            # length assume time dependent. One option would be move the time
            # dependent array_like attributes to time series to zVaraibles to
            # avoid confusion
            if attr.shape[0] == len(inp.time.data):
                coords = [np.arange(attr.shape[i + 1]) for i in range(attr.ndim - 1)]
                dims = [f"idx{i:d}" for i in range(attr.ndim - 1)]
                attr_ts = xr.DataArray(
                    attr,
                    coords=[inp.time.data, *coords],
                    dims=["time", *dims],
                )
                out_attrs[a] = time_clip(attr_ts, tint).data
            else:
                out_attrs[a] = attr

        out_attrs = {k: out_attrs[k] for k in sorted(out_attrs)}

        out = xr.Dataset(out_dict, attrs=out_attrs)

        return out

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

    idx_min = bisect.bisect_left(inp.time.data, t_start)
    idx_max = bisect.bisect_right(inp.time.data, t_stop)

    coords = [inp.time.data[idx_min:idx_max]]
    coords_attrs = [inp.time.attrs]

    if len(inp.coords) > 1:
        for k in inp.dims[1:]:
            coords.append(inp.coords[k].data)
            coords_attrs.append(inp.coords[k].attrs)

    out = xr.DataArray(
        inp.data[idx_min:idx_max, ...],
        coords=coords,
        dims=inp.dims,
        attrs=inp.attrs,
    )

    for i, k in enumerate(inp.dims):
        out[k].attrs = coords_attrs[i]

    return out
