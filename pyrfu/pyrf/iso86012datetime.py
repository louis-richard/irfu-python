#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from .iso86012datetime64 import iso86012datetime64


def iso86012datetime(
    time: list[str] | tuple[str, ...] | NDArray[np.str_],
) -> list[datetime.datetime]:
    r"""Convert ISO 8601 time to datetime.

    Parameters
    ----------
    time : array_like
        Time (list, tuple or numpy.ndarray), as str (with or without the UTC
        designator "Z", as `iso86012datetime64`) or numpy.datetime64.

    Returns
    -------
    time_datetime : list of datetime.datetime
        Time in datetime format, truncated to microseconds (the precision of
        datetime).

    Raises
    ------
    TypeError
        If time is not a list, tuple or numpy.ndarray.
    ValueError
        If time has NaT (not a time).

    """

    # Check input type
    if not isinstance(time, (list, tuple, np.ndarray)):
        raise TypeError("time must be list, tuple or numpy.ndarray")

    # The same parsing as iso86012datetime64
    time_datetime64 = iso86012datetime64(time)

    if np.any(np.isnat(time_datetime64)):
        raise ValueError("time has NaT, which has no datetime")

    # Convert to datetime, with microseconds precision (max for datetime)
    time_datetime = time_datetime64.astype("datetime64[us]").astype(datetime.datetime)

    out = time_datetime.tolist()

    return out
