#!/usr/bin/env python

# Built-in imports
from typing import Sequence

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf._units import _attrs, _ratio, _units
from pyrfu.pyrf.cross import cross
from pyrfu.pyrf.dot import dot
from pyrfu.pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


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

    mms_list = np.arange(4)

    k_list = [None] * 4

    mms_list_r0 = np.roll(mms_list, 0)
    mms_list_r1 = np.roll(mms_list, 1)
    mms_list_r2 = np.roll(mms_list, 2)
    mms_list_r3 = np.roll(mms_list, 3)

    for i, alpha, beta, gamma in zip(
        mms_list_r0,
        mms_list_r1,
        mms_list_r2,
        mms_list_r3,
    ):
        dr_jk_x_dr_jm = cross(
            r_list[beta] - r_list[alpha],
            r_list[gamma] - r_list[alpha],
        )

        dr12 = r_list[i] - r_list[alpha]

        k_list[i] = dr_jk_x_dr_jm / dot(dr_jk_x_dr_jm, dr12)
        k_list[i].attrs = _attrs(k_list[i].attrs, _ratio("", _units(r_list[0])))

    return k_list
