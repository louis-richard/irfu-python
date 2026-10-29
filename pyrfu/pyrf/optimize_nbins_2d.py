#!/usr/bin/env python

# 3rd party imports
import numpy as np

# Local imports
from .optimize_nbins_1d import _bin_indices, _finite_range

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def optimize_nbins_2d(x, y, n_min: list = None, n_max: list = None):
    r"""Estimates the number of bins for 2d histogram that minimizes
    the risk function in [1]_ , obtained by direct decomposition of the
    MISE following the method described in [2]_ .

    Parameters
    ----------
    x : xarray.DataArray
        Input time series of the first variable.
    y : xarray.DataArray
        Input time series of the second variable.
    n_min : array_like, Optional
        Minimum number of bins for each time series. Default is [1, 1]
    n_max : array_like, Optional
        Maximum number of bins for each time series (included). Default is
        [100, 100]

    Returns
    -------
    opt_n_x : int
        Number of bins of the first variable that minimizes the cost function.
    opt_n_y : int
        Number of bins of the second variable that minimizes the cost function.

    Raises
    ------
    ValueError
        If x and y have different sizes, or fewer than two different finite
        values.

    Notes
    -----
    The samples where x or y is not finite are left out (they made the ranges
    NaN). The histograms are those of numpy.histogram2d, counted with
    numpy.bincount (it took about 30 s for 100 x 100 numbers of bins and 1e5
    samples).

    References
    ----------
    .. [1]  Rudemo, M. (1982) Empirical Choice of Histograms and Kernel
            Density Estimators. Scandinavian Journal of Statistics, 9, 65-78.

    .. [2]  Shimazaki H. and Shinomoto S., A method for selecting the bin
            size of a time histogram Neural Computation (2007) Vol. 19(6),
            1503-1527

    """

    n_min = n_min or [1, 1]
    n_max = n_max or [100, 100]

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of samples")

    # Samples where both are finite (NaN made the ranges NaN)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    x_min, x_max = _finite_range(x)
    y_min, y_max = _finite_range(y)

    # #of Bins, n_max included (it was left out)
    n_x, n_y = [np.arange(min_, max_ + 1) for min_, max_ in zip(n_min, n_max)]

    d_x = (x_max - x_min) / n_x  # Bin size vector
    d_y = (y_max - y_min) / n_y  # Bin size vector

    # Bins of y for each number of bins, computed once
    i_y = [
        _bin_indices(y, y_min, y_max, n_bins).astype(np.min_scalar_type(n_bins))
        for n_bins in n_y
    ]

    # Computation of the cost function to x and y
    c_xy = np.zeros((len(n_x), len(n_y)))

    for i, n_bins_x in enumerate(n_x):
        i_x = _bin_indices(x, x_min, x_max, n_bins_x)

        for j, n_bins_y in enumerate(n_y):
            # The mean and the variance are simply computed from the event
            # counts in all the bins of the 2-dimensional histogram.
            k_i = np.bincount(i_x * n_bins_y + i_y[j], minlength=n_bins_x * n_bins_y)
            # The cost Function
            c_xy[i, j] = (2 * np.mean(k_i) - np.var(k_i)) / (d_x[i] * d_y[j]) ** 2

    # Optimal Bin Size Selection
    # get the index in x and y that produces the minimum cost function
    idx_min = np.unravel_index(np.argmin(c_xy), c_xy.shape)

    return int(n_x[idx_min[0]]), int(n_y[idx_min[1]])
