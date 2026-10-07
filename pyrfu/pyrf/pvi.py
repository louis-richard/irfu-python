#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def pvi(inp, scale: int = 10):
    r"""Returns the Partial Variance of Increments (PVI) of a time series.

    .. math::

        PVI_i = \frac{|x_{i+s} - x_i|}{\sqrt{<|x_{i+s} - x_i|^2>}}

    where :math:`s` is the scale and the average :math:`<.>` is taken over
    the whole interval, ignoring NaNs [1]_. The increment
    :math:`x_{i+s} - x_i` is given at the time of :math:`x_i`.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scale : int, Optional
        Scale at which to compute the PVI, in number of samples. Default
        is 10.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the pvi of the original time series.

    Raises
    ------
    ValueError
        If scale is not a positive integer.

    References
    ----------
    .. [1]  Greco, A., P. Chuychai, W. H. Matthaeus, S. Servidio, and P.
            Dmitruk (2008), Intermittent MHD structures and classical
            discontinuities, Geophys. Res. Lett., 35, L19111,
            doi: https://doi.org/10.1029/2008GL035454.

    """

    if not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("scale must be a positive integer")

    if len(inp.data.shape) == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    delta_inp = np.abs((data[scale:, :] - data[:-scale, :]))
    delta_inp2 = np.sum(delta_inp**2, axis=1)
    sigma = np.nanmean(delta_inp2)
    result = np.sqrt(delta_inp2 / sigma)

    time = inp.coords[inp.dims[0]].data

    attrs = {k: v for k, v in inp.attrs.items() if k != "UNITS"}
    result = xr.DataArray(
        result,
        coords=[time[0 : len(delta_inp)]],
        dims=[inp.dims[0]],
        attrs=attrs,
    )

    result.attrs["units"] = "dimensionless"
    result.attrs["TENSOR_ORDER"] = 0

    return result
