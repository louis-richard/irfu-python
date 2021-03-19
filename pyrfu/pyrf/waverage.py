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

"""waverage.py
@author: Louis Richard
"""

import numpy as np


def waverage(inp, f_sampl: float = None, n_pts: int = 7):
    """Computes weighted average.

    Parameters
    ----------
    inp : array_like
        Input data

    f_sampl : float, optional
        Sampling frequency.

    n_pts : int, optional
        Number of point to average over 5 ot 7. Default is 7

    Returns
    -------
    out : ndarray
        Weighted averaged of inp

    """

    assert n_pts in [5, 7], "n_pts  must be 5 or 7"

    if f_sampl is None:
        f_sampl = 1e9 / (inp.time.data[1] - inp.time.data[0]).view("i8")

    n_data = np.round(1e-9 * (inp.time.data[-1]
                              - inp.time.data[0]).view("i8") * f_sampl)

    inp_data = inp.data
    try:
        n_cols = inp_data.shape[1]
    except IndexError:
        inp_data = inp_data[:, None]
        n_cols = inp_data.shape[1]

    f_sampl = 1e9 * n_data / (inp.time.data[-1] - inp.time.data[0]).view("i8")
    delta_t = 1 / f_sampl

    out = np.zeros((n_data + 1, n_cols))
    out[:, 0] = np.linspace(inp_data[0, 0], inp_data[-1, 0], n_data + 1)
    indices = np.round((inp_data[:, 0] - inp_data[0, 0]) / delta_t + 1)
    out[indices, :] = inp_data[:, 1:]
    out[np.isnan(out)] = 0  # set NaNs to zeros

    for col in range(n_cols):
        if n_pts == 5:
            new_data = np.hstack([0, 0, out[: col], 0, 0])
        elif n_pts == 7:
            new_data = np.hstack([0, 0, 0, out[: col], 0, 0, 0])
        else:
            raise ValueError("n_pts must be 5 or 7")

        for j in range(n_data + 1):
            out[j, col] = _wave(new_data[j:j + n_pts - 1], n_pts)

    # Make sure we do return matrix of the same size
    out = out[indices, :]

    return out


def _wave(inp_window, n_pts):
    """computes weighted average"""

    if n_pts == 5:
        m = [.1, .25, .3, .25, .1]
    elif n_pts == 7:
        m = [.07, .15, .18, .2, .18, .15, .07]
    else:
        raise ValueError("n_pts must be 5 or 7")

    # find missing points == 0
    if np.sum(m[inp_window == 0]) == 1:
        average = 0
    else:
        average = np.sum(inp_window * m) / (1 - np.sum(m[inp_window == 0]))

    return average
