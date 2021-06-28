#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import pandas as pd

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def datetime_to_tt2000(time):
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

    time_datetime = pd.Timestamp(time)

    # Convert to string
    datetime_str = time_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')

    tt2000 = f"{datetime_str}{time_datetime.nanosecond:03d}"

    return tt2000
