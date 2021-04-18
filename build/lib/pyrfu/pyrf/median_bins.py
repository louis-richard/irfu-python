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

"""median_bins.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def median_bins(inp0, inp1, bins=10):
    """Computes median of values of y corresponding to bins of x

    Parameters
    ----------
    inp0 : xarray.DataArray
        Time series of the quantity of bins.

    inp1 : xarray.DataArray
        Time series of the quantity to the median.

    bins : int
        Number of bins.

    Returns
    -------
    out : xarray.Dataset
        Dataset with :
            * bins : xarray.DataArray
                bin values of the x variable.

            * data : xarray.DataArray
                Median values of y corresponding to each bin of x.

            * sigma : xarray.DataArray
                Standard deviation.

    Examples
    --------
    >>> import numpy
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_list = numpy.arange(1,5)

    Load magnetic field and electric field

    >>> r_mms, b_mms = [[] * 4 for _ in range(2)]
    >>> for mms_id in range(1, 5):
    >>> 	r_mms.append(mms.get_data("R_gse", tint, mms_id))
    >>> 	b_mms.append(mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id))
    >>>

    Compute current density, etc

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Compute magnitude of B and J

    >>> b_mag = pyrf.norm(b_xyz)
    >>> j_mag = pyrf.norm(j_xyz)

    Median value of |J| for 10 bins of |B|

    >>> med_b_j = pyrf.mean_bins(b_mag, j_mag)

    """

    x_sort = np.sort(inp0.data)
    x_edge = np.linspace(x_sort[0], x_sort[-1], bins + 1)

    y_med, y_std = [np.zeros(bins), np.zeros(bins)]

    for i in range(bins):
        idx_left = inp0.data > x_edge[i]
        idx_right = inp0.data < x_edge[i + 1]

        y_bins = np.abs(inp1.data[idx_left * idx_right])

        y_med[i], y_std[i] = [np.median(y_bins), np.std(y_bins)]

    bins = x_edge[:-1] + np.median(np.diff(x_edge)) / 2

    out_dict = {"data": (["bins"], y_med), "sigma": (["bins"], y_std),
                "bins": bins}

    out = xr.Dataset(out_dict)

    return bins, y_med, out
