#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _no_repeat(time):
    # Points separated in time by less than 100 ns from the next one are treated
    # as repeats, and the later one is kept (as mms_removerepeatpnts.m)
    time = np.asarray(time)

    if time.size == 0:
        return np.zeros(0, dtype=bool)

    if np.issubdtype(time.dtype, np.datetime64):
        diffs = np.diff(time.astype("datetime64[ns]"))
    else:
        diffs = np.diff(time.astype(np.int64)).astype("timedelta64[ns]")

    return np.r_[diffs >= np.timedelta64(100, "ns"), True]


def remove_repeated_points(inp):
    r"""Remove repeated elements in DataArray, Dataset or structure data.
    Important when using defatt products. Must have a time variable.

    Points separated in time by less than 100 ns are treated as repeats, and
    the later one is kept, as in mms_removerepeatpnts.m.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset or dict
        Time series of the input variable, e.g., defatt from
        :func:`pyrfu.mms.load_ancillary`. A dict must have a "time" key
        (datetime64, or int64 in ns), and its values are indexed along their
        first axis.

    Returns
    -------
    out: xarray.DataArray or xarray.Dataset or dict
        Time series of the cleaned input variable. Other inputs are returned
        unchanged.

    """

    if isinstance(inp, (xr.DataArray, xr.Dataset)):
        new_data = inp.isel(time=_no_repeat(inp.time.data))

    elif isinstance(inp, dict) and ("time" in inp):
        no_repeat = _no_repeat(inp["time"])
        new_data = {k: np.asarray(v)[no_repeat, ...] for k, v in inp.items()}

    else:
        # no change to input if it's not a DataArray or structure
        new_data = inp

    return new_data
