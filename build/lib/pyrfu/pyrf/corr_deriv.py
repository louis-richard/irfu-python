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

"""corr_deriv.py
@author: Louis Richard
"""

import numpy as np

from astropy.time import Time

from .find_closest import find_closest


def corr_deriv(inp0, inp1, fla=False):
    """Correlate the derivatives of two time series

    Parameters
    ----------
    inp0 : xarray.DataArray
        Time series of the first to variable to correlate with.

    inp1 : xarray.DataArray
        Time series of the second to variable to correlate with.

    fla : bool
        Flag if False (default) returns time instants of common highest first
        and second derivatives. If True returns time instants of common
        highest first derivative and zeros crossings.

    Returns
    -------
    t1_d, t2_d : ndarray
        Time instants of common highest first derivatives.

    t1_dd, t2_dd : ndarray
        Time instants of common highest second derivatives or zero crossings.

    """

    # 1st derivative
    tx1 = Time(inp0.time.data, format="datetime64").unix
    inp0 = inp0.data
    dtx1 = tx1[:-1] + 0.5 * np.diff(tx1)
    dx1 = np.diff(inp0)

    tx2 = Time(inp1.time.data, format="datetime64").unix
    inp1 = inp1.data
    dtx2 = tx2[:-1] + 0.5 * np.diff(tx2)
    dx2 = np.diff(inp1)

    ind_zeros1 = np.where(np.sign(dx1[:-1] * dx1[1:]) < 0)[0]
    if ind_zeros1 == 0:
        ind_zeros1 = ind_zeros1[1:]

    ind_zeros2 = np.where(np.sign(dx2[:-1] * dx2[1:]) < 0)[0]
    if ind_zeros2 == 0:
        ind_zeros2 = ind_zeros2[1:]

    ind_zeros1_p = np.where(dx1[ind_zeros1 - 1] - dx1[ind_zeros1] > 0)[0]
    ind_zeros2_p = np.where(dx2[ind_zeros2 - 1] - dx2[ind_zeros2] > 0)[0]

    ind_zeros1_m = np.where(dx1[ind_zeros1 - 1] - dx1[ind_zeros1] < 0)[0]
    ind_zeros2_m = np.where(dx2[ind_zeros2 - 1] - dx2[ind_zeros2] < 0)[0]

    ind1_p = ind_zeros1[ind_zeros1_p]
    ind1_m = ind_zeros1[ind_zeros1_m]

    t_zeros1_p = dtx1[ind1_p] + (dtx1[ind1_p + 1] - dtx1[ind1_p]) / (
            1 + np.abs(dx1[ind1_p + 1]) / np.abs(dx1[ind1_p]))
    t_zeros1_m = dtx1[ind1_m] + (dtx1[ind1_m + 1] - dtx1[ind1_m]) / (
            1 + np.abs(dx1[ind1_m + 1]) / np.abs(dx1[ind1_m]))

    ind2_p = ind_zeros2[ind_zeros2_p]
    ind2_m = ind_zeros2[ind_zeros2_m]

    t_zeros2_p = dtx2[ind2_p] + (dtx2[ind2_p + 1] - dtx2[ind2_p]) / (
            1 + np.abs(dx2[ind2_p + 1]) / np.abs(dx2[ind2_p]))
    t_zeros2_m = dtx2[ind2_m] + (dtx2[ind2_m + 1] - dtx2[ind2_m]) / (
            1 + np.abs(dx2[ind2_m + 1]) / np.abs(dx2[ind2_m]))

    # Remove repeating points
    t_zeros1_p = np.delete(t_zeros1_p, np.where(np.diff(t_zeros1_p) == 0)[0])
    t_zeros2_p = np.delete(t_zeros2_p, np.where(np.diff(t_zeros2_p) == 0)[0])

    # Define identical pairs of two time axis
    t1_d_p, t2_d_p, _, _ = find_closest(t_zeros1_p, t_zeros2_p)
    t1_d_m, t2_d_m, _, _ = find_closest(t_zeros1_m, t_zeros2_m)

    t1_d = np.vstack([t1_d_p, t1_d_m])
    t1_d = t1_d[t1_d[:, 0].argsort(), 0]

    t2_d = np.vstack([t2_d_p, t2_d_m])
    t2_d = t2_d[t2_d[:, 0].argsort(), 0]

    if fla:
        # zero crossings
        ind_zeros1 = np.where(np.sign(inp0[:-1] * inp0[1:]) < 0)[0]
        ind_zeros2 = np.where(np.sign(inp1[:-1] * inp1[1:]) < 0)[0]

        ind_zeros1 = np.delete(ind_zeros1, np.where(ind_zeros1 == 1)[0])
        ind_zeros2 = np.delete(ind_zeros2, np.where(ind_zeros2 == 1)[0])

        ind_zeros1_p = np.where(inp0[ind_zeros1 - 1] - inp0[ind_zeros1] > 0)[0]
        ind_zeros2_p = np.where(inp1[ind_zeros2 - 1] - inp1[ind_zeros2] > 0)[0]

        ind_zeros1_m = np.where(inp0[ind_zeros1 - 1] - inp0[ind_zeros1] < 0)[0]
        ind_zeros2_m = np.where(inp1[ind_zeros2 - 1] - inp1[ind_zeros2] < 0)[0]

        ind1_p = ind_zeros1[ind_zeros1_p]
        ind1_m = ind_zeros1[ind_zeros1_m]

        t_zeros1_p = tx1[ind1_p] + (tx1[ind1_p + 1] - tx1[ind1_p]) / (
                1 + np.abs(inp0[ind1_p + 1]) / np.abs(inp0[ind1_p]))
        t_zeros1_m = tx1[ind1_m] + (tx1[ind1_m + 1] - tx1[ind1_m]) / (
                1 + np.abs(inp0[ind1_m + 1]) / np.abs(inp0[ind1_m]))

        ind2_p = ind_zeros2[ind_zeros2_p]
        ind2_m = ind_zeros2[ind_zeros2_m]

        t_zeros2_p = tx2[ind2_p] + (tx2[ind2_p + 1] - tx2[ind2_p]) / (
                1 + np.abs(inp1[ind2_p + 1]) / np.abs(inp1[ind2_p]))
        t_zeros2_m = tx2[ind2_m] + (tx2[ind2_m + 1] - tx2[ind2_m]) / (
                1 + np.abs(inp1[ind2_m + 1]) / np.abs(inp1[ind2_m]))

    else:
        # 2nd derivative
        dd_tx1 = dtx1[:-1] + 0.5 * np.diff(dtx1)
        ddx1 = np.diff(dx1)

        dd_tx2 = dtx2[:-1] + 0.5 * np.diff(dtx2)
        ddx2 = np.diff(dx2)

        ind_zeros1 = np.where(np.sign(ddx1[:-1] * ddx1[1:]) < 0)[0]
        ind_zeros2 = np.where(np.sign(ddx2[:-1] * ddx2[1:]) < 0)[0]

        ind_zeros1 = np.delete(ind_zeros1, np.where(ind_zeros1 == 1)[0])
        ind_zeros2 = np.delete(ind_zeros2, np.where(ind_zeros2 == 1)[0])

        ind_zeros1_p = np.where(ddx1[ind_zeros1 - 1] - ddx1[ind_zeros1] > 0)[0]
        ind_zeros2_p = np.where(ddx2[ind_zeros2 - 1] - ddx2[ind_zeros2] > 0)[0]

        ind_zeros1_m = np.where(ddx1[ind_zeros1 - 1] - ddx1[ind_zeros1] < 0)[0]
        ind_zeros2_m = np.where(ddx2[ind_zeros2 - 1] - ddx2[ind_zeros2] < 0)[0]

        ind1_p = ind_zeros1[ind_zeros1_p]
        ind1_m = ind_zeros1[ind_zeros1_m]

        t_zeros1_p = dd_tx1[ind1_p] + (dd_tx1[ind1_p + 1] - dd_tx1[ind1_p]) \
                     / (1 + np.abs(ddx1[ind1_p + 1]) / np.abs(ddx1[ind1_p]))
        t_zeros1_m = dd_tx1[ind1_m] + (dd_tx1[ind1_m + 1] - dd_tx1[ind1_m]) \
                     / (1 + np.abs(ddx1[ind1_m + 1]) / np.abs(ddx1[ind1_m]))

        ind2_p = ind_zeros2[ind_zeros2_p]
        ind2_m = ind_zeros2[ind_zeros2_m]

        t_zeros2_p = dd_tx2[ind2_p] + (dd_tx2[ind2_p + 1] - dd_tx2[ind2_p]) \
                     / (1 + np.abs(ddx2[ind2_p + 1]) / np.abs(ddx2[ind2_p]))
        t_zeros2_m = dd_tx2[ind2_m] + (dd_tx2[ind2_m + 1] - dd_tx2[ind2_m]) \
                     / (1 + np.abs(ddx2[ind2_m + 1]) / np.abs(ddx2[ind2_m]))

    # Define identical pairs of two time axis
    t1_dd_p, t2_dd_p, _, _ = find_closest(t_zeros1_p, t_zeros2_p)
    t1_dd_m, t2_dd_m, _, _ = find_closest(t_zeros1_m, t_zeros2_m)

    t1_dd = np.vstack([t1_dd_p, t1_dd_m])
    t1_dd = t1_dd[t1_dd[:, 0].argsort(), 0]

    t2_dd = np.vstack([t2_dd_p, t2_dd_m])
    t2_dd = t2_dd[t2_dd[:, 0].argsort(), 0]

    return t1_d, t2_d, t1_dd, t2_dd
