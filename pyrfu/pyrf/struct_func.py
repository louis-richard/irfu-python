#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def struct_func(inp, scales, order):
    r"""Returns the structure function of a time series

    .. math::

       y= \frac{1}{N-s}\sum_{i=1}^{N-s}\left | x_i - x_{i+s} \right |^o

    where :math:`s` is the scale, and :math:`o` is the order, for each
    component, NaN increments ignored.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scales : int or array_like
        Scale(s) to calculate, in samples, between 1 and len(inp) - 1. None
        gives 1 to len(inp) // 2 - 1.
    order : int
        Order of the exponential of the structure function.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the structure functions (scales, components of
        inp). The index coordinate contains the scale value, and the
        attribute 'order' keeps a record on the order used for its
        calculation.

    Raises
    ------
    ValueError
        If the scales are not integers between 1 and len(inp) - 1 (or none).

    """

    if scales is None:
        scales = np.arange(1, len(inp) // 2)

    scales = np.atleast_1d(scales)

    if (
        scales.size == 0
        or not np.issubdtype(scales.dtype, np.integer)
        or np.any(scales < 1)
        or np.any(scales >= len(inp))
    ):
        raise ValueError(
            f"scales must be integers between 1 and len(inp) - 1 = {len(inp) - 1}"
        )

    if inp.ndim == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    result = []
    for scale in scales:
        increment = np.abs(data[scale:, ...] - data[:-scale, ...])
        result.append(np.nanmean(increment**order, axis=0))

    _, *comp = [inp.coords[dim].data for dim in inp.dims]

    # One row per scale, even for a single scale (it was squeezed out)
    result = xr.DataArray(
        np.stack(result).reshape(len(scales), *inp.shape[1:]),
        coords=[scales, *comp],
        dims=["scales", *inp.dims[1:]],
        attrs=inp.attrs,
    )

    result.attrs["order"] = order

    return result
