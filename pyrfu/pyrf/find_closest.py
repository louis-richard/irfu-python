#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _nearest(t_ref, t):
    r"""Index of the element of the sorted t_ref nearest to each t (ties go to
    the later element, values out of range to the nearest end)."""

    if len(t_ref) == 1:
        return np.zeros(len(t), dtype=int)

    idx = np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1)
    use_left = (t - t_ref[idx - 1]) < (t_ref[idx] - t)

    return np.where(use_left, idx - 1, idx)


def find_closest(inp1, inp2):
    r"""Finds pairs that are closest to each other in two time series.

    Time instants of inp1 that are not the nearest neighbour of any time
    instant of inp2 are removed, then those of inp2 that are not the nearest
    neighbour of any time instant of inp1, until all the remaining time
    instants pair up, as in irf_find_closest.m.

    Parameters
    ----------
    inp1 : ndarray
        Sorted vector with time instants (float or datetime64).
    inp2 : ndarray
        Sorted vector with time instants (float or datetime64).

    Returns
    -------
    t1new : ndarray
        Time instants of inp1 that are closest to those of inp2, with
        t1new[i] paired with t2new[i].
    t2new : ndarray
        Time instants of inp2 that are closest to those of inp1.
    ind1new : ndarray
        Indices of t1new in inp1.
    ind2new : ndarray
        Indices of t2new in inp2.

    """

    t_1, t_2 = [np.asarray(inp) for inp in [inp1, inp2]]
    ind_1, ind_2 = [np.arange(len(t)) for t in [t_1, t_2]]

    if t_1.size == 0 or t_2.size == 0:
        return t_1[:0], t_2[:0], ind_1[:0], ind_2[:0]

    while True:
        keep_1 = np.zeros(len(t_1), dtype=bool)
        keep_1[_nearest(t_1, t_2)] = True

        keep_2 = np.zeros(len(t_2), dtype=bool)
        keep_2[_nearest(t_2, t_1)] = True

        if not keep_1.all():
            t_1, ind_1 = t_1[keep_1], ind_1[keep_1]
        elif not keep_2.all():
            t_2, ind_2 = t_2[keep_2], ind_2[keep_2]
        else:
            break

    return t_1, t_2, ind_1, ind_2
