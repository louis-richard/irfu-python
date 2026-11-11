#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from ._time import _unix


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
        Time in unix format (seconds since 1970-01-01), NaN for NaT.

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

    # Unix times, NaN for NaT (it was -9.2e9 s, the int64 of NaT)
    return _unix(time)
