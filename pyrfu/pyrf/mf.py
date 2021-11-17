#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def mf(inp, deg):
    r"""Estimates mean field xm and wave field xw using polynomial fit of order
    `deg` for the number of columns larger than 3 assume that first column
    is time.

    Parameters
    ----------
    inp : array_like
        Input data.
    deg : int
        Degree of the fitting polynomial

    Returns
    -------
    inp_mean : numpy.ndarray
        Mean field.
    inp_wave : numpy.ndarray
        Wave field

    """
    inp_mean = inp
    inp_wave = inp

    n_rows, n_cols = inp.shape

    if n_cols >= 4:
        time = (inp[:, 0] - inp[0, 0]) / (inp[n_rows, 0] - inp[0, 0])
        data = inp[:, 1:]

    else:
        time = np.arange(len(inp))
        data = inp

    for i in range(data.shape[1]):
        res = np.polyfit(time, data[:, i], deg)
        polynomial_coeffs = res[0]
        inp_mean[:, i] = np.polyval(polynomial_coeffs, time)
        inp_wave[:, i] = data[:, i] - inp_mean[:, i]

    return inp_mean, inp_wave
