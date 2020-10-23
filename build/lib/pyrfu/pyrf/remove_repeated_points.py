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
    """Remove repeated elements in DataArray or structure data. Important when using defatt
    products. Must have a time variable.

    Parameters
    ----------
    inp : xarray.DataArray or dict
        Time series of the input variable.

    Returns
    -------
    out: xarray.DataArray or dict
        Time series of the cleaned input variable.

    """

    assert inp is not None and isinstance(inp, (xr.DataArray, dict))

    threshold = 100  # Points separated in time by less than 100ns are treated as repeats

    if isinstance(inp, xr.DataArray):
        diffs = np.diff(inp.time.data.view("i8") * 1e-9)

        no_repeat = np.ones(len(inp))
        no_repeat[diffs < threshold] = 0

        new_time, new_inp = [inp.time.data[no_repeat == 1], inp.data[no_repeat == 1, :]]

        if new_inp.ndim == 1:
            new_data = ts_scalar(new_time, new_inp)
        elif new_inp.ndim == 2:
            new_data = ts_vec_xyz(new_time, new_inp)
        elif new_inp.ndim == 3:
            new_data = ts_vec_xyz(new_time, new_inp)
        else:
            raise TypeError("Invalid data dimension")

    elif isinstance(inp, dict) and ("time" in inp):
        if inp["time"].dtype == "<M8[ns]":
            diffs = np.diff(inp["time"].view("i8") * 1e-9)
        else:
            diffs = np.diff(inp["time"])

        no_repeat = np.ones(len(inp["time"]))
        no_repeat[diffs < threshold] = 0

        var_names = inp.keys()

        for var_name in var_names:
            inp[var_name] = inp[var_name][no_repeat == 1, :]

        new_data = inp
    else:
        new_data = inp  # no change to input if it's not a DataArray or structure

    return new_data
