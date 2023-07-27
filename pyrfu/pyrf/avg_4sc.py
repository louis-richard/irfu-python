#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .calc_fs import calc_fs
from .resample import resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def avg_4sc(b_list):
    r"""Computes the input quantity at the center of mass of the MMS
    tetrahedron.

    Parameters
    ----------
    b_list : list of xarray.DataArray
        List of the time series of the quantity for each spacecraft.

    Returns
    -------
    b_avg : xarray.DataArray
        Time series of the input quantity a the enter of mass of the
        MMS tetrahedron.

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
    assert isinstance(b_list, list), "b_list must be a list"

    b_list_r = []

    for b in b_list:
        if isinstance(b, xr.DataArray):
            b_list_r.append(resample(b, b_list[0], f_s=calc_fs(b_list[0])))
        else:
            raise TypeError("elements of b_list must be xarray.DataArray")

    # Average the resamples time series
    b_avg = sum(b_list) / len(b_list)

    return b_avg
