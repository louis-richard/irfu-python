#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, Union

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def unix2datetime64(time: Union[list[float], NDArray[Any]]) -> NDArray[np.datetime64]:
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

    # Make sure that time is in ns format
    time_unix = (time_array * 1e9).astype(np.int64)

    # Convert to unix
    time_datetime64 = time_unix.astype("datetime64[ns]")

    return time_datetime64
