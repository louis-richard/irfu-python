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


def datetime642iso8601(
    time: Union[np.datetime64, NDArray[np.datetime64]],
) -> NDArray[np.str_]:
    r"""Convert datetime64 in ns units to ISO 8601 time format .

    Parameters
    ----------
    time : numpy.datetime64 or numpy.ndarray
        Time in datetime64 in ns units.

    Returns
    -------
    time_iso8601 : numpy.ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """
    if isinstance(time, np.datetime64):
        time_datetime64 = np.atleast_1d(time).astype("datetime64[ns]")
    elif isinstance(time, np.ndarray) and isinstance(time[0], np.datetime64):
        time_datetime64 = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be numpy.datetime64 or numpy.ndarray")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601
