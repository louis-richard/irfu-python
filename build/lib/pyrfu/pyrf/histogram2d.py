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

"""histogram2d.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from .resample import resample


def histogram2d(inp1, inp2, bins=100):
    """Computes 2d histogram of inp2 vs inp1 with nbins number of bins.

    Parameters
    ----------
    inp1 : xarray.DataArray
        Time series of the x values.

    inp2 : xarray.DataArray
        Time series of the y values.

    bins : int
        Number of bins.

    Returns
    -------
    out : xarray.DataArray
        2D map of the density of inp2 vs inp1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = np.arange(1, 5)

    Load magnetic field and electric field

    >>> r_mms = [mms.get_data("r_gse", tint, i) for i in mms_id]
    >>> b_mms = [mms.get_data("b_gse_fgm_srvy_l2", tint, i) for i in mms_id]

    Compute current density, etc

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Compute magnitude of B and J

    >>> b_mag = pyrf.norm(b_xyz)
    >>> j_mag = pyrf.norm(j_xyz)

    Histogram of |J| vs |B|

    >>> h2d_b_j = pyrf.histogram2d(b_mag, j_mag)

    """

    # resample inp2 with respect to inp1
    if len(inp2) != len(inp1):
        inp2 = resample(inp2, inp1)

    h2d, x_edges, y_edges = np.histogram2d(inp1.data, inp2.data, bins=bins)

    x_bins = x_edges[:-1] + np.median(np.diff(x_edges)) / 2
    y_bins = y_edges[:-1] + np.median(np.diff(y_edges)) / 2

    out = xr.DataArray(h2d, coords=[x_bins, y_bins], dims=["x_bins", "y_bins"])

    return out
