#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from .calc_fs import calc_fs
from .resample import resample

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _valid_count(inp):
    r"""1 where `inp` (array_like or DataArray) is not NaN, 0 where it is, as a
    numpy.ndarray of int (xr.where returned a numpy.ndarray for an array,
    whose ``.data`` taken by feeps_avg_4sc was a memoryview)."""
    return (~np.isnan(np.asarray(inp))).astype(np.int64)


def nanavg_4sc(b_list: Sequence[DataArray]) -> DataArray:
    r"""Average data from 4 spacecrafts while ignoring NaN values.
    Computes the input quantity at the center of mass of the MMS
    tetrahedron. When averaging, NaN values are ignored by counting the number of
    non-NaN values at each time step.

    Parameters
    ----------
    b_list : Sequence of DataArray or Dataset
        List of the time series of the quantity for each spacecraft.

    Returns
    -------
    b_avg : DataArray or Dataset
        Time series of the input quantity a the enter of mass of the
        MMS tetrahedron.

    Raises
    ------
    TypeError
        If b_list is not a list of DataArray or Dataset

    """
    # Check input type
    if not isinstance(b_list, list):
        raise TypeError("b_list must be a list")

    b_list_r = []

    for b in b_list:
        if isinstance(b, (xr.DataArray, xr.Dataset)):
            b_list_r.append(resample(b, b_list[0], f_s=calc_fs(b_list[0])))
        else:
            raise TypeError("elements of b_list must be DataArray or Dataset")

    # Values to average: the data of a DataArray, or the "data" variable of a
    # Dataset (e.g. a skymap, whose other variables are taken from the first)
    values = [
        np.asarray(b.data if isinstance(b, xr.DataArray) else b["data"].data)
        for b in b_list_r
    ]

    b_avg_data = np.zeros(values[0].shape)
    b_nan_denom = np.zeros(values[0].shape)

    for value in values:
        # Count the valid samples before replacing NaNs by zeros in the sum
        b_nan_denom += _valid_count(value)
        b_avg_data += np.nan_to_num(value, nan=0.0)

    # Copy so that the attributes of the caller's first time series are unchanged
    attrs = dict(b_list[0].attrs)

    for key in ["probe", "mms", "MMS", "mmsId"]:
        if key in attrs:
            attrs[key] = "4sc_avg"

    # NaN where no spacecraft has data
    with np.errstate(invalid="ignore", divide="ignore"):
        b_avg_data = b_avg_data / b_nan_denom

    if isinstance(b_list_r[0], xr.DataArray):
        b_avg = xr.DataArray(
            b_avg_data,
            coords=b_list_r[0].coords,
            dims=b_list_r[0].dims,
            attrs=attrs,
        )
    else:
        b_avg = b_list_r[0].assign(data=b_list_r[0]["data"].copy(data=b_avg_data))
        b_avg.attrs = attrs

    return b_avg
