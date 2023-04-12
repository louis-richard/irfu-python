#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def iso86012datetime(time):
    r"""Converts ISO 8601 time to datetime

    Parameters
    ----------
    time : ndarray or list
        Time

    Returns
    -------
    time_datetime : list
        Time in datetime format.

    """

    # Make sure that str is in ISO8601 format
    time = np.array(time).astype("<M8[ns]").astype(str)

    # ISO 8601 format with miliseconds precision (max precision for datetime)
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    # Convert to datetime format
    time_datetime = [datetime.datetime.strptime(t_[:-3], fmt) for t_ in time]

    return time_datetime
