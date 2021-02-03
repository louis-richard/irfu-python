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

import numpy as np
import xarray as xr

from .ts_vec_xyz import ts_vec_xyz
from .ts_scalar import ts_scalar


def remove_repeated_points(inp):
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
