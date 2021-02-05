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

"""avg_4sc.py
@author: Louis Richard
"""

from .resample import resample


def avg_4sc(b_list):
    """Computes the input quantity at the center of mass of the MMS
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

    b_list = [resample(b, b_list[0]) for b in b_list]

    b_avg = sum(b_list) / len(b_list)

    return b_avg
