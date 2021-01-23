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

import numpy as np
import xarray as xr

from scipy.stats import kurtosis


def increments(x, scale=10):
    """Returns the increments of a time series.

    .. math:: y = |x_i - x_{i+s}|

    where :math:`s` is the scale.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the increments.

    Returns
    -------
    kurt : ndarray
        kurtosis of the increments, one per product, using the Fisher's
        definition (0 value for a normal distribution).

    result : xarray.DataArray
        An xarray containing the time series increments, one per
        product in the original time series.

    """

    data = x.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    f = np.abs((data[scale:, :] - data[:-scale, :]))

    result = np.array(f)

    t, c = x.dims

    time = x.coords[t].data
    cols = x.coords[c].data

    result = xr.DataArray(result, coords=[time[0:len(f)], cols], dims=[t, c], attrs=x.attrs)

    kurt = kurtosis(result, axis=0, fisher=False)

    return kurt, result
