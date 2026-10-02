#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy.stats import kurtosis

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def increments(inp, scale: int = 10):
    r"""Returns the increments of a time series.

    .. math:: y_i = x_{i+s} - x_i

    where :math:`s` is the scale. The increment is given at the time of
    :math:`x_i`.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scale : int, Optional
        Scale at which to compute the increments, in number of samples.
        Default is 10.

    Returns
    -------
    kurt : ndarray
        Kurtosis (flatness) of the increments, one per product, using
        Pearson's definition :math:`<y^4> / <y^2>^2` (3 for a normal
        distribution), ignoring NaNs.
    result : xarray.DataArray
        An xarray containing the time series increments, one per
        product in the original time series.

    Raises
    ------
    ValueError
        If scale is not a positive integer.

    """

    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"
    assert inp.ndim < 4, "inp must ber a scalar, vector or tensor"

    if not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("scale must be a positive integer")

    if inp.ndim == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    # Compute the increments
    delta_inp = data[scale:, ...] - data[:-scale, ...]

    # Compute kurtosis of the increments
    kurt = kurtosis(delta_inp, axis=0, fisher=False, nan_policy="omit")

    times, *comp = [inp.coords[dim].data for dim in inp.dims]

    if inp.ndim == 1:
        delta_inp = delta_inp[:, 0]

    result = xr.DataArray(
        delta_inp,
        coords=[times[0 : len(delta_inp)], *comp],
        dims=inp.dims,
        attrs=inp.attrs,
    )

    return kurt, result
