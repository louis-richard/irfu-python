#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from pyrfu._core.time import _datetime64
from pyrfu._core.typing import TimeArray


def unix2datetime64(time: list[float] | NDArray[Any]) -> TimeArray:
    r"""Converts unix time to datetime64 in ns units.

    Parameters
    ----------
    time : numpy.ndarray
        Time in unix format.

    Returns
    -------
    time_datetime64 : numpy.ndarray
        Time in datetime64 format.

    Raises
    ------
    TypeError
        If time is not a list or numpy.ndarray.

    See Also
    --------
    pyrfu.pyrf.datetime642unix

    """
    # Check input type
    if isinstance(time, (list, np.ndarray)):
        time_array = np.array(time)
    else:
        raise TypeError("time must be list or numpy.ndarray")

    # To the nearest ns (rounded rather than truncated, e.g., 6.5e-5 * 1e9 =
    # 64999.99999999999), NaT for NaN
    return _datetime64(time_array)
