#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _finite_range(x):
    r"""Minimum and maximum of x, which must have two different values."""
    if x.size == 0 or np.min(x) == np.max(x):
        raise ValueError("the input must have at least two different finite values")

    return np.min(x), np.max(x)


def _bin_indices(x, x_min, x_max, n_bins):
    r"""Bin of each sample of x in n_bins equal bins of [x_min, x_max] (the last
    one closed), computed as numpy.histogram."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    indices = ((x - x_min) / (x_max - x_min) * n_bins).astype(np.intp)
    indices[indices == n_bins] -= 1
    indices[x < edges[indices]] -= 1
    indices[(x >= edges[indices + 1]) & (indices != n_bins - 1)] += 1

    return indices


def optimize_nbins_1d(x, n_min: int = 1, n_max: int = 100):
    r"""Estimates the number of bins for 1d histogram that minimizes the
    risk function in [1]_ , obtained by direct decomposition of the MISE
    following the method described in [2]_ .

    Parameters
    ----------
    x : xarray.DataArray
        Input time series
    n_min : int, Optional
        Minimum number of bins. Default is 1.
    n_max : int, Optional
        Maximum number of bins (included). Default is 100.

    Returns
    -------
    opt_n_x : int
        Number of bins that minimizes the cost function.

    Raises
    ------
    ValueError
        If x has fewer than two different finite values.

    Notes
    -----
    The non-finite samples are left out (they made the range NaN). The
    histograms are those of numpy.histogram, counted with numpy.bincount.

    References
    ----------
    .. [1]  Rudemo, M. (1982) Empirical Choice of Histograms and Kernel Density
            Estimators. Scandinavian Journal of Statistics, 9, 65-78.

    .. [2]  Shimazaki H. and Shinomoto S., A method for selecting the bin size
            of a time histogram Neural Computation (2007) Vol. 19(6), 1503-1527
    """

    # Finite samples (NaN made the range NaN)
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x_min, x_max = _finite_range(x)

    # #of Bins, n_max included (it was left out)
    ns_x = np.arange(n_min, n_max + 1)

    # Bin size vector
    ds_x = (x_max - x_min) / ns_x

    cs_x = np.zeros(ds_x.shape)
    # Computation of the cost function to x and y
    for i, n_x in enumerate(ns_x):
        # The mean and the variance are simply computed from the
        # event counts in all the bins of the 1-dimensional histogram.
        k_i = np.bincount(_bin_indices(x, x_min, x_max, n_x), minlength=n_x)
        k_ = np.mean(k_i)  # Mean of event count
        v_ = np.var(k_i)  # Variance of event count
        # The cost Function
        cs_x[i] = (2 * k_ - v_) / ds_x[i] ** 2

    # Optimal Bin Size Selection
    # combination of i and j that produces the minimum cost function
    idx_min = np.argmin(cs_x)  # get the index of the min Cxy

    opt_n_x = int(ns_x[idx_min])

    return opt_n_x
