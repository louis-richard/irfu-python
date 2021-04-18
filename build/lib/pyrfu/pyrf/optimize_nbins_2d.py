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

"""optimize_nbins_2d.py
@author: Louis Richard
"""

import numpy as np


def optimize_nbins_2d(x, y, n_min:list = None, n_max: list = None):
    r"""Estimates the number of bins for 2d histogram that minimizes
    the risk function in [1]_ , obtained by direct decomposition of the
    MISE following the method described in [2]_ .

    Parameters
    ----------
    x : xarray.DataArray
        Input time series of the first variable.

    y : xarray.DataArray
        Input time series of the second variable.

    n_min : list, optional
        Minimum number of bins for each time series. Default is [1, 1]

    n_max : list, optional
        Maximum number of bins for each time series. Default is [100, 100]

    Returns
    -------
    opt_n_x : int
        Number of bins of the first variable that minimizes the cost function.

    opt_n_y : int
        Number of bins of the second variable that minimizes the cost function.

    References
    ----------
    _[1]    Rudemo, M. (1982) Empirical Choice of Histograms and Kernel Density
            Estimators. Scandinavian Journal of Statistics, 9, 65-78.

    _[2]    Shimazaki H. and Shinomoto S., A method for selecting the bin size
            of a time histogram Neural Computation (2007) Vol. 19(6), 1503-1527

    """

    if n_max is None:
        n_max = [100, 100]
    if n_min is None:
        n_min = [1, 1]

    x_min, x_max = [np.min(x.data), np.max(x.data)]
    y_min, y_max = [np.min(y.data), np.max(y.data)]

    # #of Bins
    n_x, n_y = [np.arange(min_, max_) for min_, max_ in zip(n_min, n_max)]

    d_x = (x_max - x_min) / n_x  # Bin size vector
    d_y = (y_max - y_min) / n_y  # Bin size vector

    d_xy = [[(i, j) for j in d_y] for i in d_x]

    # matrix of bin size vector
    d_xy = np.array(d_xy, dtype=[('x', float) ,('y', float)])

    # Computation of the cost function to x and y
    c_xy = np.zeros(d_xy.shape)

    for i in range(len(n_x)):
        for j in range(len(n_y)):
            k_i = np.histogram2d(x, y, bins=(n_x[i], n_y[j]))
            # The mean and the variance are simply computed from the event
            # counts in all the bins of the 2-dimensional histogram.
            k_i = k_i[0]
            k_ = np.mean(k_i)  # Mean of event count
            v_ = np.var(k_i)   # Variance of event count
            # The cost Function
            c_xy[i ,j] = (2 * k_ - v_) / ((d_xy[i, j][0] * d_xy[i, j][1]) ** 2)

    # Optimal Bin Size Selection
    # combination of i and j that produces the minimum cost function
    idx_min = np.where(c_xy == np.min(c_xy))  # get the index of the min Cxy

    # get the index in x and y that produces the minimum cost function
    idx_nx, idx_ny = [idx_min[i][0] for i in range(2)]

    return n_x[idx_nx], n_y[idx_ny]