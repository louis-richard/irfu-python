#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
corr_deriv.py

@author : Louis RICHARD
"""

import numpy as np


from astropy.time import Time

from . import find_closest


def corr_deriv(x1=None, x2=None, fla=False):
    """
    Correlate the derivatives of two time series

    Parameters :
        - x1 : DataArray
            Time series of the first to variable to correlate with

        - x2 : DataArray
            Time series of the second to variable to correlate with

        - fla : bool
            Flag if False (default) returns time instants of common highest first and second derivatives.
            If True returns time instants of common highest first derivative and zeros crossings

    Return :
        t1_d, t2_d : array
            Time instants of common highest first derivatives

        t1_dd, t2_dd : array
            Time instants of common highest second derivatives or zero crossings

    """

    # 1st derivative
    tx1 = Time(x1.time.data, format="datetime64").unix
    x1 = x1.data
    dtx1 = tx1[:-1] + 0.5 * np.diff(tx1)
    dx1 = np.diff(x1)

    tx2 = Time(x2.time.data, format="datetime64").unix
    x2 = x2.data
    dtx2 = tx2[:-1] + 0.5 * np.diff(tx2)
    dx2 = np.diff(x2)

    ind_zeros1 = np.where(np.sign(dx1[:-1] * dx1[1:]) < 0)[0]
    if ind_zeros1 == 0:
        ind_zeros1 = ind_zeros1[1:]

    ind_zeros2 = np.where(np.sign(dx2[:-1] * dx2[1:]) < 0)[0]
    if ind_zeros2 == 0:
        ind_zeros2 = ind_zeros2[1:]

    ind_zeros1_plus = np.where(dx1[ind_zeros1 - 1] - dx1[ind_zeros1] > 0)[0]
    ind_zeros2_plus = np.where(dx2[ind_zeros2 - 1] - dx2[ind_zeros2] > 0)[0]

    ind_zeros1_minu = np.where(dx1[ind_zeros1 - 1] - dx1[ind_zeros1] < 0)[0]
    ind_zeros2_minu = np.where(dx2[ind_zeros2 - 1] - dx2[ind_zeros2] < 0)[0]

    ind1_plus = ind_zeros1[ind_zeros1_plus]
    ind1_minu = ind_zeros1[ind_zeros1_minu]

    t_zeros1_plus = dtx1[ind1_plus] + (dtx1[ind1_plus + 1] - dtx1[ind1_plus]) / (
                1 + np.abs(dx1[ind1_plus + 1]) / np.abs(dx1[ind1_plus]))
    t_zeros1_minu = dtx1[ind1_minu] + (dtx1[ind1_minu + 1] - dtx1[ind1_minu]) / (
                1 + np.abs(dx1[ind1_minu + 1]) / np.abs(dx1[ind1_minu]))

    ind2_plus = ind_zeros2[ind_zeros2_plus]
    ind2_minu = ind_zeros2[ind_zeros2_minu]

    t_zeros2_plus = dtx2[ind2_plus] + (dtx2[ind2_plus + 1] - dtx2[ind2_plus]) / (
                1 + np.abs(dx2[ind2_plus + 1]) / np.abs(dx2[ind2_plus]))
    t_zeros2_minu = dtx2[ind2_minu] + (dtx2[ind2_minu + 1] - dtx2[ind2_minu]) / (
                1 + np.abs(dx2[ind2_minu + 1]) / np.abs(dx2[ind2_minu]))

    # Remove repeating points
    t_zeros1_plus = np.delete(t_zeros1_plus, np.where(np.diff(t_zeros1_plus) == 0)[0])
    t_zeros2_plus = np.delete(t_zeros2_plus, np.where(np.diff(t_zeros2_plus) == 0)[0])

    # Define identical pairs of two time axis
    t1_d_plus, t2_d_plus, _, _ = find_closest(t_zeros1_plus, t_zeros2_plus)
    t1_d_minu, t2_d_minu, _, _ = find_closest(t_zeros1_minu, t_zeros2_minu)

    t1_d = np.vstack([t1_d_plus, t1_d_minu])
    t1_d = t1_d[t1_d[:, 0].argsort(), 0]

    t2_d = np.vstack([t2_d_plus, t2_d_minu])
    t2_d = t2_d[t2_d[:, 0].argsort(), 0]

    if fla:
        # zero crossings
        ind_zeros1 = np.where(np.sign(x1[:-1] * x1[1:]) < 0)[0]
        ind_zeros2 = np.where(np.sign(x2[:-1] * x2[1:]) < 0)[0]

        ind_zeros1 = np.delete(ind_zeros1, np.where(ind_zeros1 == 1)[0])
        ind_zeros2 = np.delete(ind_zeros2, np.where(ind_zeros2 == 1)[0])

        ind_zeros1_plus = np.where(x1[ind_zeros1 - 1] - x1[ind_zeros1] > 0)[0]
        ind_zeros2_plus = np.where(x2[ind_zeros2 - 1] - x2[ind_zeros2] > 0)[0]

        ind_zeros1_minu = np.where(x1[ind_zeros1 - 1] - x1[ind_zeros1] < 0)[0]
        ind_zeros2_minu = np.where(x2[ind_zeros2 - 1] - x2[ind_zeros2] < 0)[0]

        ind1_plus = ind_zeros1[ind_zeros1_plus]
        ind1_minu = ind_zeros1[ind_zeros1_minu]

        t_zeros1_plus = tx1[ind1_plus] + (tx1[ind1_plus + 1] - tx1[ind1_plus]) / (
                    1 + np.abs(x1[ind1_plus + 1]) / np.abs(x1[ind1_plus]))
        t_zeros1_minu = tx1[ind1_minu] + (tx1[ind1_minu + 1] - tx1[ind1_minu]) / (
                    1 + np.abs(x1[ind1_minu + 1]) / np.abs(x1[ind1_minu]))

        ind2_plus = ind_zeros2[ind_zeros2_plus]
        ind2_minu = ind_zeros2[ind_zeros2_minu]

        t_zeros2_plus = tx2[ind2_plus] + (tx2[ind2_plus + 1] - tx2[ind2_plus]) / (
                    1 + np.abs(x2[ind2_plus + 1]) / np.abs(x2[ind2_plus]))
        t_zeros2_minu = tx2[ind2_minu] + (tx2[ind2_minu + 1] - tx2[ind2_minu]) / (
                    1 + np.abs(x2[ind2_minu + 1]) / np.abs(x2[ind2_minu]))

    else:
        # 2nd derivative
        ddtx1 = dtx1[:-1] + 0.5 * np.diff(dtx1)
        ddx1 = np.diff(dx1)

        ddtx2 = dtx2[:-1] + 0.5 * np.diff(dtx2)
        ddx2 = np.diff(dx2)

        ind_zeros1 = np.where(np.sign(ddx1[:-1] * ddx1[1:]) < 0)[0]
        ind_zeros2 = np.where(np.sign(ddx2[:-1] * ddx2[1:]) < 0)[0]

        ind_zeros1 = np.delete(ind_zeros1, np.where(ind_zeros1 == 1)[0])
        ind_zeros2 = np.delete(ind_zeros2, np.where(ind_zeros2 == 1)[0])

        ind_zeros1_plus = np.where(ddx1[ind_zeros1 - 1] - ddx1[ind_zeros1] > 0)[0]
        ind_zeros2_plus = np.where(ddx2[ind_zeros2 - 1] - ddx2[ind_zeros2] > 0)[0]

        ind_zeros1_minu = np.where(ddx1[ind_zeros1 - 1] - ddx1[ind_zeros1] < 0)[0]
        ind_zeros2_minu = np.where(ddx2[ind_zeros2 - 1] - ddx2[ind_zeros2] < 0)[0]

        ind1_plus = ind_zeros1[ind_zeros1_plus]
        ind1_minu = ind_zeros1[ind_zeros1_minu]

        t_zeros1_plus = ddtx1[ind1_plus] + (ddtx1[ind1_plus + 1] - ddtx1[ind1_plus]) / (
                    1 + np.abs(ddx1[ind1_plus + 1]) / np.abs(ddx1[ind1_plus]))
        t_zeros1_minu = ddtx1[ind1_minu] + (ddtx1[ind1_minu + 1] - ddtx1[ind1_minu]) / (
                    1 + np.abs(ddx1[ind1_minu + 1]) / np.abs(ddx1[ind1_minu]))

        ind2_plus = ind_zeros2[ind_zeros2_plus]
        ind2_minu = ind_zeros2[ind_zeros2_minu]

        t_zeros2_plus = ddtx2[ind2_plus] + (ddtx2[ind2_plus + 1] - ddtx2[ind2_plus]) / (
                    1 + np.abs(ddx2[ind2_plus + 1]) / np.abs(ddx2[ind2_plus]))
        t_zeros2_minu = ddtx2[ind2_minu] + (ddtx2[ind2_minu + 1] - ddtx2[ind2_minu]) / (
                    1 + np.abs(ddx2[ind2_minu + 1]) / np.abs(ddx2[ind2_minu]))

    # Define identical pairs of two time axis
    [t1_dd_plus, t2_dd_plus] = find_closest(t_zeros1_plus, t_zeros2_plus)
    [t1_dd_minu, t2_dd_minu] = find_closest(t_zeros1_minu, t_zeros2_minu)

    t1_dd = np.vstack([t1_dd_plus, t1_dd_minu])
    t1_dd = t1_dd[t1_dd[:, 0].argsort(), 0]

    t2_dd = np.vstack([t2_dd_plus, t2_dd_minu])
    t2_dd = t2_dd[t2_dd[:, 0].argsort(), 0]

    return t1_d, t2_d, t1_dd, t2_dd