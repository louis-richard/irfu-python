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


def iso86012datetime64(
    time: Union[list[str], NDArray[np.str_]],
) -> NDArray[np.datetime64]:
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : array_like
        Time in ISO 8601 format

    Returns
    -------
    time_datetime64 : numpy.ndarray
        Time in datetime64 in ns units.

    Raises
    ------
    TypeError
        If time is not a list or numpy.ndarray.


    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """
    # Check input type
    if isinstance(time, list):
        time_array = np.array(time)
    elif isinstance(time, np.ndarray):
        time_array = time
    else:
        raise TypeError("time must be list or numpy.ndarray")

    time_datetime64 = time_array.astype("datetime64[ns]")

    return time_datetime64
