#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import datetime

import numpy as np
import pandas as pd

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def datetime2iso8601(time):
    r"""Transforms datetime to TT2000 string format.

    Parameters
    ----------
    time : datetime.datetime
        Time to convert to tt2000 string.

    Returns
    -------
    tt2000 : str
        Time in TT20000 iso_8601 format.

    """

    # Check input type
    message = "time must be array_like or datetime.datetime"
    assert isinstance(time, (list, np.ndarray, datetime.datetime)), message

    if isinstance(time, (np.ndarray, list)):
        return list(map(datetime2iso8601, time))

    assert isinstance(time, datetime.datetime), "time datetime.datetime"

    time_datetime = pd.Timestamp(time)

    # Convert to string
    datetime_str = time_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

    time_iso8601 = f"{datetime_str}{time_datetime.nanosecond:03d}"

    return time_iso8601
