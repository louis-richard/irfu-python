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


def _nan_count(inp):

    inp_to_counts = inp.where(np.isnan(inp) == True, other=1)
    inp_to_counts = inp_to_counts.where(np.isnan(inp) == False, other=0)

    return inp_to_counts


def nanavg_4sc(b_list: Sequence[DataArray]) -> DataArray:

    if not isinstance(b_list, list):
        raise TypeError("b_list must be a list")

    b_list_r = []

    for b in b_list:
        if isinstance(b, DataArray):
            b_list_r.append(resample(b, b_list[0], f_s=calc_fs(b_list[0])))
        else:
            raise TypeError("elements of b_list must be DataArray or Dataset")

    b_list_r = [b.where(np.isnan(b) == False, other=0) for b in b_list_r]

    b_avg_data = np.zeros(b_list_r[0].shape)
    b_nan_denom = np.zeros(b_list_r[0].shape)

    for b in b_list_r:

        b_avg_data += b.data
        b_nan_denom += _nan_count(b).data

    b_avg = xr.DataArray(
        b_avg_data / b_nan_denom,
        coords=b_list_r[0].coords,
        dims=b_list_r[0].dims,
        attrs=b_list_r[0].attrs,
    )

    return b_avg
