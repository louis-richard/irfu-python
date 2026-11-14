#!/usr/bin/env python

# Built-in imports
from collections.abc import Sequence

# 3rd party imports
import numpy as np
from xarray import DataArray

# Local imports
from pyrfu._core.units import _attrs, _ratio, _units
from pyrfu.pyrf.cross import cross
from pyrfu.pyrf.dot import dot
from pyrfu.pyrf.resample import resample


def c_4_k(r_list: Sequence[DataArray]) -> Sequence[DataArray]:
    r"""Calculates reciprocal vectors in barycentric coordinates.

    Parameters
    ----------
    r_list : list of DataArray
        Position of the spacecrafts, resampled to the times of the first if
        their times differ.

    Returns
    -------
    k_list : list of DataArray
        Reciprocal vectors in barycentric coordinates.

    Raises
    ------
    ValueError
        If r_list doesn't have 4 elements.

    Notes
    -----
    The units of reciprocal vectors are the same as [1/r].

    """

    if len(r_list) != 4:
        raise ValueError("r_list must contain exactly 4 elements.")

    # The differences of positions on other time lines were on their common
    # times only (none for shifted times: IndexError)
    r_list = [
        (
            r
            if np.array_equal(r.time.data, r_list[0].time.data)
            else resample(r, r_list[0])
        )
        for r in r_list
    ]

    k_list = []

    for i in range(4):
        # The other three spacecraft, in the cyclic order i - 1, i - 2, i - 3
        alpha, beta, gamma = (i - 1) % 4, (i - 2) % 4, (i - 3) % 4

        dr_jk_x_dr_jm = cross(
            r_list[beta] - r_list[alpha],
            r_list[gamma] - r_list[alpha],
        )

        k_i = dr_jk_x_dr_jm / dot(dr_jk_x_dr_jm, r_list[i] - r_list[alpha])
        k_i.attrs = _attrs(k_i.attrs, _ratio("", _units(r_list[0])))
        k_list.append(k_i)

    return k_list
