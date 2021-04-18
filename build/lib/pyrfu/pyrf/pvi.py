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

"""pvi.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def pvi(inp, scale=10):
    """Returns the PVI of a time series.

    .. math::

        y = \\frac{|x_i - x_{i+s}|^2}{<|x_i - x_{i+s}|^2>}

    where :math:`s` is the scale.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the PVI.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the pvi of the original time series.

    """

    if len(inp.data.shape) == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    delta_inp = np.abs((data[scale:, :] - data[:-scale, :]))
    delta_inp2 = np.sum(delta_inp ** 2, axis=1)
    sigma = np.mean(delta_inp2)
    result = np.array(delta_inp2 / sigma)

    time = inp.coords[inp.dims[0]].data

    result = xr.DataArray(result, coords=[time[0:len(delta_inp)]],
                          dims=[inp.dims[0]], attrs=inp.attrs)

    result.attrs["units"] = "dimensionless"

    return result
