#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from datetime import datetime
from typing import Union

# 3rd party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Local imports
from .datetime642iso8601 import datetime642iso8601

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def datetime2iso8601(
    time: Union[list[datetime], NDArray, datetime, np.datetime64],
) -> Union[list[str], str]:
    r"""Transforms times to ISO 8601 strings with nanoseconds.

    Parameters
    ----------
    time : datetime.datetime or numpy.datetime64 or array_like
        Time(s) to convert: ``datetime.datetime`` objects, or
        ``numpy.datetime64`` values (e.g., the times of a time series).

    Returns
    -------
    time_iso8601 : str or list of str
        Time(s) as "YYYY-MM-DDThh:mm:ss.mmmuuunnn": a str for a single time, a
        list for an array_like.

    Raises
    ------
    TypeError
        If the times are neither datetime.datetime nor numpy.datetime64.

    """

    # numpy.datetime64 times (scalar or array), e.g. those of a time series
    if np.asarray(time).dtype.kind == "M":
        time_iso8601 = [str(t_) for t_ in datetime642iso8601(np.atleast_1d(time))]

        return time_iso8601[0] if np.ndim(time) == 0 else time_iso8601

    if isinstance(time, (np.ndarray, list)):
        time_iso8601 = []

        for t in time:
            time_iso8601.append(datetime2iso8601(t))

    elif isinstance(time, datetime):
        time_datetime = pd.Timestamp(time)

        # Convert to string
        datetime_str = time_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

        time_iso8601 = f"{datetime_str}{time_datetime.nanosecond:03d}"

    else:
        raise TypeError("time must be datetime.datetime or numpy.datetime64")

    return time_iso8601
