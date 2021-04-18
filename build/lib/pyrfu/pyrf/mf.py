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

"""mf.py
@author: Louis Richard
"""

import numpy as np


def mf(inp, deg):
    """Estimates mean field xm and wave field xw using polynomial fit of order
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
    inp_mean : ndarray
        Mean field.

    inp_wave : ndarray
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
