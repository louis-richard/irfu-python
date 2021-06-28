#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .iso86012datetime64 import iso86012datetime64
from .datetime642iso8601 import datetime642iso8601

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def extend_tint(tint, ext: list = None):
    r"""Extends time interval.

    Parameters
    ----------
    tint : list of str
        Reference time interval to extend.
    ext : list of float or list of int
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

    if ext is None:
        ext = [-60, 60]

    # Convert extension to timedelta64 in s units
    ext = np.array(ext).astype("timedelta64[s]")

    # Original time interval to datetime64 format in ns units
    tint_ori = iso86012datetime64(np.array(tint))

    # New time interval in iso 8601 format
    tint_new = list(datetime642iso8601(tint_ori + ext))

    return tint_new
