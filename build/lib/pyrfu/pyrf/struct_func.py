#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.14"
__status__ = "Prototype"


def struct_func(inp, scales, order):
    r"""Returns the structure function of a time series

    .. math::

       y= \frac{1}{N-s}\sum_{i=1}^{N-s}(x_i - x_{i+s})^o

    where :math:`s` is the scale, and :math:`o` is the order.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scales : array_like
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

    result = []
    for scale in scales:
        result.append(
            np.mean(np.abs(data[scale:, :] - data[:-scale, :]) ** order,
                    axis=0))

    result = np.squeeze(result)
    result = xr.DataArray(result, coords=[scales],
                          dims=["scale"], attrs=inp.attrs)

    result.attrs['order'] = order

    return result
