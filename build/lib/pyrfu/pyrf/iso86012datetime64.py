#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def iso86012datetime64(time):
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : ndarray
        Time in ISO 8601 format

    Returns
    -------
    time_datetime64 : ndarray
        Time in datetime64 in ns units.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    time_datetime64 = time.astype("<M8[ns]")

    return time_datetime64
