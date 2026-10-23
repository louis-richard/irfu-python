#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def datetime642unix(
    time: Union[list[np.datetime64], NDArray[np.datetime64]],
) -> NDArray[np.float64]:
    r"""Converts datetime64 in ns units to unix time.

    Parameters
    ----------
    time : list or ndarray
        Time in datetime64 format (may be empty).

    Returns
    -------
    time_unix : ndarray
        Time in unix format.

    Raises
    ------
    TypeError
        If time is not a list or an array of datetime64.

    See Also
    --------
    pyrfu.pyrf.unix2datetime64

    """

    if not isinstance(time, (list, np.ndarray)):
        raise TypeError("time must be list or numpy.ndarray of datetime64")

    # Check the type of the elements from the dtype, not the first element
    # (an empty input raised IndexError): an empty list has no type, and an
    # object array may hold datetime64
    time = np.asarray(time)

    if time.size == 0 or (
        time.dtype.kind == "O" and all(isinstance(t, np.datetime64) for t in time.flat)
    ):
        time = time.astype("datetime64[ns]")

    if time.dtype.kind != "M":
        raise TypeError("time must be list or numpy.ndarray of datetime64")

    # Make sure that time is in ns format
    time_datetime64 = time.astype("datetime64[ns]")

    # Convert to unix
    time_unix = time_datetime64.astype(np.int64) / 1e9

    return time_unix
