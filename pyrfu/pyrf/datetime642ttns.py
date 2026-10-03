#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
import pycdfpp
from numpy.typing import NDArray

# Local imports
from .ttns2datetime64 import TT2000_FILL, ttns2datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Earliest time that both CDF_TT2000 (from 1707-09-22, after the fill and pad
# values) and datetime64[ns] (from 1677-09-21) can represent
_DATETIME64_TT2000_MIN = ttns2datetime64(TT2000_FILL + 2)[0]

# Units that can overflow when converted to datetime64[ns]
_COARSE_UNITS = ["generic", "Y", "M", "W", "D", "h", "m", "s", "ms", "us"]


def datetime642ttns(
    time: Union[np.datetime64, NDArray[np.datetime64]],
) -> NDArray[np.int64]:
    r"""Converts datetime64 to epoch_tt2000 (nanoseconds since J2000).

    Leap seconds are taken into account. NaT maps to the CDF_TT2000 fill value.

    Parameters
    ----------
    time : numpy.datetime64 or numpy.ndarray
        Times in datetime64 format, in any unit, between
        1707-09-22T12:12:10.961224194 and 2262-04-11T23:47:16.854775807.

    Returns
    -------
    time_ttns : numpy.ndarray
        Times in epoch_tt2000 format (nanoseconds since J2000).

    Raises
    ------
    TypeError
        If time is not a numpy.datetime64 or a numpy.ndarray of datetime64.
    ValueError
        If a time is outside the range that both datetime64[ns] and
        CDF_TT2000 can represent.

    """

    if not isinstance(time, (np.datetime64, np.ndarray)) or not np.issubdtype(
        np.asarray(time).dtype, np.datetime64
    ):
        raise TypeError("time must be numpy.datetime64 or numpy.ndarray")

    time = np.atleast_1d(time)
    is_nat = np.isnat(time)

    # pycdfpp only converts datetime64 in ns units. Times outside the ns range
    # wrap around silently, and do not convert back to their own unit.
    time_datetime64 = time.astype("datetime64[ns]")

    is_out = time_datetime64 < _DATETIME64_TT2000_MIN

    if np.datetime_data(time.dtype)[0] in _COARSE_UNITS:
        is_out |= time_datetime64.astype(time.dtype) != time

    is_out &= ~is_nat

    if np.any(is_out):
        raise ValueError(
            "time must be between 1707-09-22T12:12:10.961224194 and "
            "2262-04-11T23:47:16.854775807",
        )

    time_datetime64[is_nat] = np.datetime64("2000-01-01", "ns")

    time_ttns = pycdfpp.to_tt2000(time_datetime64).view(np.int64).copy()
    time_ttns[is_nat] = TT2000_FILL

    return time_ttns
