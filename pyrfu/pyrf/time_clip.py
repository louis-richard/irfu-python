#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu._core.time_attrs import _is_time_attr

from .iso86012datetime64 import iso86012datetime64


def _time_interval(tint):
    if isinstance(tint, xr.DataArray):
        t_start, t_stop = tint.time.data[[0, -1]]
    elif isinstance(tint, (np.ndarray, list, tuple)):
        # A tuple, and datetime.datetime values, raised TypeError
        if isinstance(tint[0], (np.datetime64, datetime.datetime)):
            t_start, t_stop = [np.datetime64(t_, "ns") for t_ in tint]
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
    tint : xarray.DataArray or ndarray or list or tuple
        Time interval: a time series (its first and last times), or the start
        and stop times as datetime64, datetime.datetime or ISO 8601 strings.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the time clipped input, with the times in
        [t_start, t_stop] (inclusive). All the coordinates are kept, and the
        time dependent ones are clipped. For a Dataset, the time dependent
        attributes are clipped too: those of the particle distributions (step
        table, energy, time and azimuthal widths, FEEPS coverage), not their
        energy tables and polar, pitch-angle or gyrophase widths, and the
        other array_like attributes with a first dimension the length of
        time.

    Notes
    -----
    To select times of a distribution, use time_clip rather than
    ``Dataset.isel`` or ``Dataset.sel``, which keep the per-time attributes
    at full length (the energy widths of `pyrfu.mms.vdf_d3v` then do not
    match the times).

    """
    t_start, t_stop = _time_interval(tint)

    idx_min = np.searchsorted(inp.time.data, t_start, side="left")
    idx_max = np.searchsorted(inp.time.data, t_stop, side="right")
    out = inp.isel(time=slice(idx_min, idx_max))

    if isinstance(inp, xr.Dataset):
        # The time dependent attributes, see _is_time_attr
        out_attrs = {}

        for k in sorted(inp.attrs):
            attr = inp.attrs[k]

            if _is_time_attr(k, attr, len(inp.time.data)):
                out_attrs[k] = attr[idx_min:idx_max, ...]
            else:
                out_attrs[k] = attr

        # Dataset.isel shares the attributes dictionary with inp
        out.attrs = out_attrs

    return out
