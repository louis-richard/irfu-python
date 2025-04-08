#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from datetime import datetime
from typing import Union

# 3rd party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def datetime2iso8601(
    time: Union[list[datetime], NDArray[datetime], datetime],
) -> Union[list[str], str]:
    r"""Transforms datetime to TT2000 string format.

    Parameters
    ----------
    time : datetime
        Time to convert to tt2000 string.

    Returns
    -------
    tt2000 : str
        Time in TT20000 iso_8601 format.

    """

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
        raise TypeError("time must be array_like or datetime")

    return time_iso8601
