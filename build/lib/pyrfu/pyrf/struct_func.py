#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
struct_func.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def struct_func(x=None, scale=None, order=2):
    """Returns the structure function of a time series

    .. math::

       y= \\frac{1}{N-s}\\sum_{i=1}^{N-s}(x_i - x_{i+s})^o

    where :math:`s` is the scale, and :math:`o` is the order.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series.

    scale : list or numpy.ndarray
        A list or an array containing the scales to calculate.

    order : int
        Order of the exponential of the structure function.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the structure functions, one per product in the original time
        series. The index coordinate contains the scale value, and the attribute 'order' keeps a
        record on the order used for its calculation.

    """

    assert x is not None and isinstance(x, xr.DataArray)

    if scale is None:
        scale = [1]

    data = x.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    result = []
    for s in scale:
        result.append(np.mean(np.abs((data[s:, :] - data[:-s, :]) ** order), axis=0))

    result = np.array(result)

    c = x.dims[1]
    cols = x.coords[c].data

    result = xr.DataArray(result, coords=[scale, cols], dims=["scale", c], attrs=x.attrs)

    result.attrs['order'] = order

    return result
