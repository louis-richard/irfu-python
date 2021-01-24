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

import bisect
import numpy as np
import xarray as xr


def t_eval(inp, t):
    """Evaluates the input time series at the target time.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series if the input to evaluate.

    t : ndarray
        Times at which the input will be evaluated.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input at times t.

    """

    idx = np.zeros(len(t))

    for i, time in enumerate(t):
        idx[i] = bisect.bisect_left(inp.time.data, time)

    idx = idx.astype(int)

    if inp.ndim == 2:
        out = xr.DataArray(inp.data[idx, :], coords=[t, inp.comp], dims=["time", "comp"])
    else:
        out = xr.DataArray(inp.data[idx], coords=[t], dims=["time"])

    return out
