#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
avg_4sc.py
@author : Louis RICHARD
"""

import xarray as xr

from .resample import resample


def avg_4sc(b_list=None):
    """Computes the input quantity at the center of mass of the MMS tetrahedron.

    Parameters
    ----------
    b_list : list of xarray.DataArray
        List of the time series of the quantity for each spacecraft.

    Returns
    -------
    b_avg : xarray.DataArray
        Time series of the input quantity a the enter of mass of the MMS tetrahedron.

    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu.pyrf import avg_4sc
    >>> # Time interval
    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    >>> # Spacecraft indices
    >>> b_mms = [mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id) for mms_id in range(1, 5)]
    >>> b_xyz = pyrf.avg_4sc(b_mms)

    """

    if b_list is None:
        raise ValueError("avg_4sc requires 1 argument")

    if not isinstance(b_list, list):
        raise TypeError("B must be a list of the 4 spacecraft data")

    for i, b in enumerate(b_list):
        if not isinstance(b, xr.DataArray):
            raise TypeError("B[{:d}] must be a DataArray".format(i))

    b_list = [resample(b, b_list[0]) for b in b_list]

    b_avg = sum(b_list)/len(b_list)

    return b_avg
