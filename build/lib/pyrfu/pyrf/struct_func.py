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

"""struct_func.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def struct_func(inp, scales, order):
    """Returns the structure function of a time series

    .. math::

       y= \\frac{1}{N-s}\\sum_{i=1}^{N-s}(x_i - x_{i+s})^o

    where :math:`s` is the scale, and :math:`o` is the order.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.

    scales : list or ndarray
        A list or an array containing the scales to calculate.

    order : int
        Order of the exponential of the structure function.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the structure functions, one per product in
        the original time series. The index coordinate contains the scale
        value, and the attribute 'order' keeps a record on the order used
        for its calculation.

    """

    if scales is None:
        scales = [1]

    data = inp.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    result = []
    for scale in scales:
        result.append(
            np.mean(np.abs((data[scale:, :] - data[:-scale, :]) ** order),
                    axis=0))

    result = np.array(result)

    cols = inp.coords[inp.dims[1]].data

    result = xr.DataArray(result, coords=[scales, cols],
                          dims=["scale", inp.dims[1]], attrs=inp.attrs)

    result.attrs['order'] = order

    return result
