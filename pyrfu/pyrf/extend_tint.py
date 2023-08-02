#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from .datetime642iso8601 import datetime642iso8601

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def extend_tint(tint, ext: list = None):
    r"""Extends time interval.

    Parameters
    ----------
    tint : list of str
        Reference time interval to extend.
    ext : list of float or list of float
        Number of seconds to extend time interval
        [left extend, right extend].

    Returns
    -------
    tint_new : list of str
        Extended time interval.

    Examples
    --------
    >>> from pyrfu import pyrf

    Time interval

    >>> tints = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]

    Load spacecraft position

    >>> tints_long = pyrf.extend_tint(tint, [-100, 100])

    """

    # Set default extension
    if ext is None:
        ext = [-60.0, 60.0]

    # Make sure tint and ext are 2 elements array_like
    message = "must be array_like with 2 elements"
    assert isinstance(tint, (np.ndarray, list)) and len(tint) == 2, f"tint {message}"
    assert isinstance(ext, (np.ndarray, list)) and len(ext) == 2, f"ext {message}"

    # Convert extension to timedelta64[ns]
    ext = (np.array(ext) * 1e9).astype("timedelta64[ns]")

    # Original time interval to datetime64[ns]
    if isinstance(tint[0], np.datetime64):
        tint_ori = tint
    elif isinstance(tint[0], str):
        tint_ori = iso86012datetime64(np.array(tint))
    else:
        raise TypeError("Invalid time format!! Must be datetime64 or str!!")

    # New time interval in iso 8601 format
    tint_new = list(datetime642iso8601(tint_ori + ext))

    return tint_new
