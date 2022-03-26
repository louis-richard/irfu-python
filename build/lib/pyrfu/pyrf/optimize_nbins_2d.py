#!/usr/bin/env python

# Built-in imports
import itertools

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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
        Maximum number of bins for each time series. Default is [100, 100]

    Returns
    -------
    opt_n_x : int
        Number of bins of the first variable that minimizes the cost function.
    opt_n_y : int
        Number of bins of the second variable that minimizes the cost function.

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

    x_min, x_max = [np.min(x.data), np.max(x.data)]
    y_min, y_max = [np.min(y.data), np.max(y.data)]

    # #of Bins
    n_x, n_y = [np.arange(min_, max_) for min_, max_ in zip(n_min, n_max)]

    d_x = (x_max - x_min) / n_x  # Bin size vector
    d_y = (y_max - y_min) / n_y  # Bin size vector

    d_xy = [[(i, j) for j in d_y] for i in d_x]

    # matrix of bin size vector
    d_xy = np.array(d_xy, dtype=[('x', float), ('y', float)])

    # Computation of the cost function to x and y
    c_xy = np.zeros(d_xy.shape)

    for i, j in itertools.product(range(len(n_x)), range(len(n_y))):
        k_i = np.histogram2d(x, y, bins=(n_x[i], n_y[j]))
        # The mean and the variance are simply computed from the event
        # counts in all the bins of the 2-dimensional histogram.
        k_i = k_i[0]
        # The cost Function
        c_xy[i, j] = (2 * np.mean(k_i) - np.var(k_i)) / (
                    (d_xy[i, j][0] * d_xy[i, j][1]) ** 2)

    # Optimal Bin Size Selection
    # get the index in x and y that produces the minimum cost function
    n_x = n_x[np.where(c_xy == np.min(c_xy))[0][0]]
    n_y = n_y[np.where(c_xy == np.min(c_xy))[1][0]]

    return n_x, n_y
