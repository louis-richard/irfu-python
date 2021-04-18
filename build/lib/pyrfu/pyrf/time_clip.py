#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""time_clip.py
@author: Louis Richard
"""

import bisect
import datetime
import numpy as np
import xarray as xr

from dateutil import parser
from astropy.time import Time


def time_clip(inp, tint):
    """Time clip the input (if time interval is TSeries clip between start
    and stop).

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the quantity to clip.

    tint : xarray.DataArray or ndarray or list
        Time interval can be a time series, a array of datetime64 or a list.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time clipped input.

    """

    if isinstance(tint, xr.DataArray):
        t_start, t_stop = [tint.time.data[0], tint.time.data[-1]]

    elif isinstance(tint, np.ndarray):
        if isinstance(tint[0], datetime.datetime) \
                and isinstance(tint[-1], datetime.datetime):
            t_start, t_stop = [tint.time[0], tint.time[-1]]

        else:
            raise TypeError('Values must be in Datetime64')

    elif isinstance(tint, list):
        t_start, t_stop = [parser.parse(tint[0]), parser.parse(tint[-1])]

    else:
        raise TypeError("invalid tint")

    idx_min = bisect.bisect_left(inp.time.data,
                                 Time(t_start, format="datetime").datetime64)
    idx_max = bisect.bisect_right(inp.time.data,
                                  Time(t_stop, format="datetime").datetime64)

    coord = [inp.time.data[idx_min:idx_max]]

    if len(inp.coords) > 1:
        for k in inp.dims[1:]:
            coord.append(inp.coords[k].data)

    out = xr.DataArray(inp.data[idx_min:idx_max, ...], coords=coord,
                       dims=inp.dims, attrs=inp.attrs)

    out.time.attrs = inp.time.attrs

    return out
