#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def datetime642unix(time):
    r"""Converts datetime64 in ns units to unix time.

    Parameters
    ----------
    time : ndarray
        Time in datetime64 format.

    Returns
    -------
    time_unix : ndarray
        Time in unix format.

    See Also
    --------
    pyrfu.pyrf.unix2datetime64

    """

    # Make sure that time is in ns format
    time_ns = time.astype("<M8[ns]")

    # Convert to unix
    time_unix = time_ns.astype("int64") / 1e9

    return time_unix
