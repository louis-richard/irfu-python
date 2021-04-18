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

"""ts_scalar.py
@author: Louis Richard
"""

import xarray as xr


def ts_scalar(time, data, attrs=None):
    """Create a time series containing a 0th order tensor

    Parameters
    ----------
    time : ndarray
        Array of times.

    data : ndarray
        Data corresponding to the time list.

    attrs : dict
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        0th order tensor time series.

    """

    assert data.ndim == 1, "Input must be a scalar"
    assert len(time) == len(data), "Time and data must have the same length"

    if attrs is None:
        attrs = {}

    out = xr.DataArray(data, coords=[time[:]], dims="time", attrs=attrs)
    out.attrs["TENSOR_ORDER"] = 0

    return out
