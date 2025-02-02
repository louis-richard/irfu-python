#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.calc_fs import calc_fs
from pyrfu.pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def avg_4sc(b_list: Sequence[DataArray]) -> DataArray:
    r"""Computes the input quantity at the center of mass of the MMS
    tetrahedron.

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

    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu.pyrf import avg_4sc

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> b_mms = [get_data("B_gse_fgm_srvy_l2", tint, i) for i in range(1, 5)]
    >>> b_xyz = avg_4sc(b_mms)

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

    b_avg_data = np.zeros(b_list_r[0].data.shape)

    # for b in b_list_r:
    #     b_avg_data += b.data

    # Average the resamples time series
    if isinstance(b_list_r[0], xr.DataArray):

        for b in b_list_r:
            b_avg_data += b.data

        b_avg = xr.DataArray(
            b_avg_data / len(b_list_r),
            coords=b_list_r[0].coords,
            dims=b_list_r[0].dims,
            attrs=b_list_r[0].attrs,
        )
    else:

        data_vars_names = list(b_list_r[0].data_vars)
        data_vars_coords = [
            list(list(b_list_r[0].data_vars[data_vars_name].coords))
            for data_vars_name in data_vars_names
        ]

        for b in b_list_r:
            b_avg_data += b.data.data

        data_vars_dict = {
            data_vars_name: b_list_r[0].data_vars[data_vars_name].data
            for data_vars_name in data_vars_names
            if data_vars_name != "data"
        }
        data_vars_dict["data"] = b_avg_data / len(b_list_r)
        # return data_vars_dict
        b_avg = xr.Dataset(
            data_vars={
                data_vars_name: (data_vars_coord, data_vars_dict[data_vars_name])
                for data_vars_name, data_vars_coord in zip(
                    data_vars_names, data_vars_coords
                )
            },
            coords=b_list_r[0].coords,
            attrs=b_list_r[0].attrs,
        )
    # return data_vars_dict
    return b_avg
