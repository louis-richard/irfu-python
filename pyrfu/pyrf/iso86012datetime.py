#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
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


def iso86012datetime(
    time: Union[list[str], tuple[str, ...], NDArray[np.str_]],
) -> list[datetime.datetime]:
    r"""Convert ISO 8601 time to datetime.

    Parameters
    ----------
    time : array_like
        Time (list, tuple or numpy.ndarray), as str or numpy.datetime64.

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
    # Check input type (tuples raised TypeError)
    if not isinstance(time, (list, tuple, np.ndarray)):
        raise TypeError("time must be list, tuple or numpy.ndarray")

    # Make sure that str is in ISO8601 format
    time_datetime64 = np.asarray(time).astype("datetime64[ns]")

    if np.any(np.isnat(time_datetime64)):
        raise ValueError("time has NaT, which has no datetime")

    # Convert to datetime format, with microseconds precision (max precision for
    # datetime), at once (it parsed the ISO 8601 str of each time)
    time_datetime = time_datetime64.astype("datetime64[us]").astype(datetime.datetime)

    return time_datetime.tolist()
