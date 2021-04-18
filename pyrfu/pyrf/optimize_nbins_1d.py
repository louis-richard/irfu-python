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

"""optimize_nbins_1d.py
@author: Louis Richard
"""

import numpy as np


def optimize_nbins_1d(x, n_min: int = 1, n_max: int = 100):
    r"""Estimates the number of bins for 1d histogram that minimizes
    the risk function in [1]_ , obtained by direct decomposition of the
    MISE following the method described in [2]_ .

    Parameters
    ----------
    x : xarray.DataArray
        Input time series

    n_min : int, optional
        Minimum number of bins. Default is 1.

    n_max : int, optional
        Maximum number of bins. Default is 100.

    Returns
    -------
    opt_n_x : int
        Number of bins that minimizes the cost function.

    References
    ----------
    _[1]    Rudemo, M. (1982) Empirical Choice of Histograms and Kernel Density
            Estimators. Scandinavian Journal of Statistics, 9, 65-78.

    _[2]    Shimazaki H. and Shinomoto S., A method for selecting the bin size
            of a time histogram Neural Computation (2007) Vol. 19(6), 1503-1527
    """

    x_min, x_max = [np.min(x.data), np.max(x.data)]

    # #of Bins
    n_x = np.arange(n_min, n_max)

    # Bin size vector
    d_x = (x_max - x_min) / n_x

    c_x = np.zeros(d_x.shape)
    # Computation of the cost function to x and y
    for i in range(len(n_x)):
        k_i = np.histogram(x, bins=n_x[i])
        # The mean and the variance are simply computed from the
        # event counts in all the bins of the 1-dimensional histogram.
        k_i = k_i[0]
        k_ = np.mean(k_i)  # Mean of event count
        v_ = np.var(k_i)   # Variance of event count
        # The cost Function
        c_x[i] = (2 * k_ - v_) / d_x[i] ** 2

    # Optimal Bin Size Selection
    # combination of i and j that produces the minimum cost function
    idx_min = np.argmin(c_x)  # get the index of the min Cxy

    opt_n_x = n_x[idx_min]

    return opt_n_x
