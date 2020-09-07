#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remove_repeated_points.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from . import ts_vec_xyz, ts_scalar


def remove_repeated_points(inp=None):
    """
    Remove repeated elements in DataArray or structure data. Important when using DEFATT products.
    Must have a time variable.

    Parameters :
        - inp : DataArray/dict
            Time series of the input variable

    Return :
        - out: DataArray/dict
            Time series of the cleaned input variable

    """
    threshold = 100  # Points separated in time by less than 100ns are treated as repeats

    if isinstance(inp, xr.DataArray):
        diffs = np.diff(inp.time.data.view("i8") * 1e-9)

        norepeat = np.ones(len(inp))
        norepeat[diffs < threshold] = 0

        newtstime = inp.time.data[norepeat == 1]
        newinp = inp.data[norepeat == 1, :]

        if newinp.ndim == 1:
            newdata = ts_scalar(newtstime, newinp)
        elif newinp.ndim == 2:
            newdata = ts_vec_xyz(newtstime, newinp)
        elif newinp.ndim == 3:
            newdata = ts_vec_xyz(newtstime, newinp)
        else:
            raise TypeError("Invalid data dimension")

    elif isinstance(inp, dict) and ("time" in inp):
        if inp["time"].dtype == "<M8[ns]":
            diffs = np.diff(inp["time"].view("i8") * 1e-9)
        else:
            diffs = np.diff(inp["time"])

        norepeat = np.ones(len(inp["time"]))
        norepeat[diffs < threshold] = 0

        varnames = inp.keys()

        for varname in varnames:
            inp[varname] = inp[varname][norepeat == 1, :]

        newdata = inp
    else:
        newdata = inp  # no change to input if it's not a TSERIES or structure

    return newdata
