#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find_closest.py

@author : Louis RICHARD
"""

import numpy as np

from scipy import interpolate


def find_closest(t1=None, t2=None):
    """
    Finds pairs that are closest to each other in two timeseries

    Parameters :
        t1 : np.ndarray
            Vector with time instants
        t2 : np.ndarray
            Vector with time instants

    Returns :
        t1new : np.ndarray
            Identified time instants that are closest each other

        t2new : np.ndarray
            Identified time instants that are closest each other

        ind1new : np.ndarray
            Identified time instants that are closest each other

        ind2new : np.ndarray
            Identified time instants that are closest each other

    """

    t1_orig = t1
    t2_orig = t2
    flag = True

    while flag:
        flag_t1 = np.zeros(t1.shape)
        tckt1 = interpolate.interp1d(t1, np.arange(len(t1)), kind="nearest", fill_value="extrapolate")
        ind = tckt1(t2)
        flag_t1[ind] = 1

        flag_t2 = np.zeros(t2.shape)
        tckt2 = interpolate.interp1d(t2, np.arange(len(t2)), kind="nearest", fill_value="extrapolate")
        ind = tckt2(t1)
        flag_t2[ind] = 1

        ind_zeros_t1 = np.where(flag_t1 == 0)[0]
        ind_zeros_t2 = np.where(flag_t2 == 0)[0]
        if ind_zeros_t1:
            t1 = np.delete(t1, ind_zeros_t1)
        elif ind_zeros_t2:
            t2 = np.delete(t2, ind_zeros_t2)
        else:
            break

    t1new = t1
    t2new = t2

    tckt1_orig = interpolate.interp1d(t1_orig, np.arange(len(t1_orig)), kind="nearest")
    ind1new = tckt1_orig(t1new)

    tckt2_orig = interpolate.interp1d(t2_orig, np.arange(len(t2_orig)), kind="nearest")
    ind2new = tckt2_orig(t2new)

    return t1new, t2new, ind1new, ind2new
