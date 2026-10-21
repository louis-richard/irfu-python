#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from .datetime642iso8601 import datetime642iso8601

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def extend_tint(tint, ext: list = None):
    r"""Extends time interval.

    Parameters
    ----------
    tint : list of str or list of numpy.datetime64
        Reference time interval to extend.
    ext : list of float, Optional
        Seconds added to the start and to the end of the time interval
        (negative to extend the start backwards), rounded to the nanosecond.
        Default is [-60, 60].

    Returns
    -------
    tint_new : list of str
        Extended time interval, in ISO 8601 format with nanoseconds.

    Examples
    --------
    >>> from pyrfu import pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]

    Extend it by 100 s on each side

    >>> tint_long = pyrf.extend_tint(tint, [-100, 100])

    """

    # Set default extension
    if ext is None:
        ext = [-60.0, 60.0]

    # Make sure tint and ext are 2 elements array_like
    message = "must be array_like with 2 elements"
    assert isinstance(tint, (np.ndarray, list)) and len(tint) == 2, f"tint {message}"
    assert isinstance(ext, (np.ndarray, list)) and len(ext) == 2, f"ext {message}"

    # Convert extension to timedelta64[ns], rounded (0.0157 * 1e9 is
    # 15699999.999999998, which was truncated to 1 ns less)
    ext = np.round(np.array(ext, dtype=np.float64) * 1e9).astype("timedelta64[ns]")

    # Original time interval to datetime64[ns]
    if isinstance(tint[0], np.datetime64):
        tint_ori = tint
    elif isinstance(tint[0], str):
        tint_ori = iso86012datetime64(np.array(tint))
    else:
        raise TypeError("Invalid time format!! Must be datetime64 or str!!")

    # New time interval in iso 8601 format
    tint_new = [str(t_) for t_ in datetime642iso8601(tint_ori + ext)]

    return tint_new
