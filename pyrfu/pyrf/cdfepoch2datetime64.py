#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from cdflib import cdfepoch

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _compose_date(
    years,
    months,
    days,
    hours=None,
    minutes=None,
    seconds=None,
    milliseconds=None,
    microseconds=None,
    nanoseconds=None,
):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = [
        "<M8[Y]",
        "<m8[M]",
        "<m8[D]",
        "<m8[h]",
        "<m8[m]",
        "<m8[s]",
        "<m8[ms]",
        "<m8[us]",
        "<m8[ns]",
    ]
    vals = [
        years,
        months,
        days,
        hours,
        minutes,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds,
    ]

    dates_list = []
    for t, v in zip(types, vals):
        if v is not None:
            dates_list.append(np.asarray(v, dtype=t))

    dates = sum(dates_list)

    return dates


def cdfepoch2datetime64(epochs):
    r"""Converts CDF epochs to numpy.datetime64 with nanosecond precision.

    Parameters
    ----------
    epochs : float or int or array_like
        CDF epochs to convert.

    Returns
    -------
    times : numpy.ndarray
        Array of times in datetime64([ns]).

    """

    # Check input type
    times = cdfepoch.breakdown(epochs)
    times = np.transpose(np.atleast_2d(times))

    times = _compose_date(*times).astype("datetime64[ns]")

    return times
