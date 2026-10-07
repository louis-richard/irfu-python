#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .find_closest import find_closest

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _derivative(time, inp):
    # Derivative at the middle of the time steps
    return time[:-1] + 0.5 * np.diff(time), np.diff(inp)


def _zero_crossings(time, inp):
    # Interpolated times of the downward (from positive to negative) and upward
    # zero crossings of inp, as in irf_corr_deriv.m
    ind = np.where(np.sign(inp[:-1] * inp[1:]) < 0)[0]
    ind = ind[ind > 0]

    out = []

    for ind_ in [ind[inp[ind - 1] - inp[ind] > 0], ind[inp[ind - 1] - inp[ind] < 0]]:
        frac = 1 / (1 + np.abs(inp[ind_ + 1]) / np.abs(inp[ind_]))
        out.append(np.unique(time[ind_] + (time[ind_ + 1] - time[ind_]) * frac))

    return out


def _common(zeros1, zeros2):
    # Pairs of closest crossings of the same kind, sorted in time
    t_1, t_2 = [[], []]

    for z_1, z_2 in zip(zeros1, zeros2):
        t1_, t2_, _, _ = find_closest(z_1, z_2)
        t_1.append(t1_)
        t_2.append(t2_)

    return np.sort(np.hstack(t_1)), np.sort(np.hstack(t_2))


def corr_deriv(inp0, inp1, flag: bool = False):
    r"""Correlate the derivatives of two time series

    Finds the time instants of common maxima and minima (zeros of the first
    derivative), and of common steepest gradients (zeros of the second
    derivative) or zero crossings, as irf_corr_deriv.m.

    Parameters
    ----------
    inp0 : xarray.DataArray
        Time series of the first to variable to correlate with.
    inp1 : xarray.DataArray
        Time series of the second to variable to correlate with.
    flag : bool, Optional
        Flag if False (default) returns time instants of common steepest
        gradients as t1_dd, t2_dd. If True returns time instants of common
        zeros crossings.

    Returns
    -------
    t1_d, t2_d : ndarray
        Time instants of common maxima and minima of inp0 and inp1.
    t1_dd, t2_dd : ndarray
        Time instants of common steepest gradients (or zero crossings) of
        inp0 and inp1.

    """

    # Times in seconds relative to the first time of inp0
    t_ref = inp0.time.data[0].astype("datetime64[ns]")
    tx1, tx2 = [
        (inp.time.data.astype("datetime64[ns]") - t_ref) / np.timedelta64(1, "s")
        for inp in [inp0, inp1]
    ]
    x_1, x_2 = [inp.data.astype(np.float64) for inp in [inp0, inp1]]

    # 1st derivative
    dtx1, dx1 = _derivative(tx1, x_1)
    dtx2, dx2 = _derivative(tx2, x_2)

    t1_d, t2_d = _common(_zero_crossings(dtx1, dx1), _zero_crossings(dtx2, dx2))

    if flag:
        # zero crossings
        zeros1, zeros2 = [_zero_crossings(tx1, x_1), _zero_crossings(tx2, x_2)]
    else:
        # 2nd derivative
        zeros1 = _zero_crossings(*_derivative(dtx1, dx1))
        zeros2 = _zero_crossings(*_derivative(dtx2, dx2))

    t1_dd, t2_dd = _common(zeros1, zeros2)

    t1_d, t2_d, t1_dd, t2_dd = [
        t_ref + np.round(t_ * 1e9).astype("timedelta64[ns]")
        for t_ in [t1_d, t2_d, t1_dd, t2_dd]
    ]

    return t1_d, t2_d, t1_dd, t2_dd
