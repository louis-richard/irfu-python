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

"""find_closest.py
@author: Louis Richard
"""

import numpy as np

from scipy import interpolate


def find_closest(inp1, inp2):
    """Finds pairs that are closest to each other in two time series.

    Parameters
    ----------
    inp1 : ndarray
        Vector with time instants.

    inp2 : ndarray
        Vector with time instants.

    Returns
    -------
    t1new : ndarray
        Identified time instants that are closest each other.

    t2new : ndarray
        Identified time instants that are closest each other.

    ind1new : ndarray
        Identified time instants that are closest each other.

    ind2new : ndarray
        Identified time instants that are closest each other.

    """

    t1_orig = inp1
    t2_orig = inp2
    flag = True

    nt1, nt2 = [len(t) for t in [inp1, inp2]]

    while flag:
        flag_t1 = np.zeros(inp1.shape)
        tckt1 = interpolate.interp1d(inp1, np.arange(nt1),
                                     kind="nearest", fill_value="extrapolate")
        flag_t1[tckt1(inp2)] = 1

        flag_t2 = np.zeros(inp2.shape)
        tckt2 = interpolate.interp1d(inp2, np.arange(nt2),
                                     kind="nearest", fill_value="extrapolate")
        flag_t2[tckt2(inp1)] = 1

        ind_zeros_t1 = np.where(flag_t1 == 0)[0]
        ind_zeros_t2 = np.where(flag_t2 == 0)[0]
        if ind_zeros_t1:
            inp1 = np.delete(inp1, ind_zeros_t1)
        elif ind_zeros_t2:
            inp2 = np.delete(inp2, ind_zeros_t2)
        else:
            break

    tckt1_orig = interpolate.interp1d(t1_orig, np.arange(nt1), kind="nearest")
    tckt2_orig = interpolate.interp1d(t2_orig, np.arange(nt2), kind="nearest")

    return inp1, inp2, tckt1_orig(inp1), tckt2_orig(inp2)
