#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .ts_vec_xyz import ts_vec_xyz
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def remove_repeated_points(inp):
    r"""Remove repeated elements in DataArray or structure data. Important
    when using defatt products. Must have a time variable.

    Parameters
    ----------
    inp : xarray.DataArray or dict
        Time series of the input variable.

    Returns
    -------
    out: xarray.DataArray or dict
        Time series of the cleaned input variable.

    """

    # Points separated in time by less than 100ns are treated as repeats
    threshold = 100

    if isinstance(inp, xr.DataArray):
        diffs = np.diff(inp.time.data.astype(int) * 1e-9)

        no_repeat = np.ones(len(inp))
        no_repeat[diffs < threshold] = 0

        new_time = inp.time.data[no_repeat == 1]
        new_inp = inp.data[no_repeat == 1, :]

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
        # no change to input if it's not a DataArray or structure
        new_data = inp

    return new_data
