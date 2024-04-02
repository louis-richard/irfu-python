#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def iso86012datetime(time: NDArray[np.str_]) -> list[datetime.datetime]:
    r"""Convert ISO 8601 time to datetime.

    Parameters
    ----------
    time : numpy.ndarray
        Time

    Returns
    -------
    time_datetime : list of datetime.datetime
        Time in datetime format.

    """

    # Make sure that str is in ISO8601 format
    time_datetime64 = time.astype("datetime64[ns]")
    time_iso8601 = time_datetime64.astype(str)

    # ISO 8601 format with miliseconds precision (max precision for datetime)
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    # Convert to datetime format
    time_datetime = [datetime.datetime.strptime(t_[:-3], fmt) for t_ in time_iso8601]

    return time_datetime
