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


def pvi(x, scale=10):
    """Returns the PVI of a time series.

    .. math::

        y = \\frac{|x_i - x_{i+s}|^2}{<|x_i - x_{i+s}|^2>}

    where :math:`s` is the scale.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the PVI.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the pvi of the original time series.

    """

    data = x.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    f = np.abs((data[scale:, :] - data[:-scale, :]))
    f2 = np.sum(f ** 2, axis=1)
    sigma = np.mean(f2)
    result = np.array(f2 / sigma)

    t = x.dims[0]
    time = x.coords[t].data

    result = xr.DataArray(result, coords=[time[0:len(f)]], dims=[t], attrs=x.attrs)

    result.attrs["units"] = "dimensionless"

    return result
