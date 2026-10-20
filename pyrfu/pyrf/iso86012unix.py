#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def iso86012unix(time):
    r"""Converts time in ISO 8601 format to Unix time.

    Parameters
    ----------
    time : str or array_like of str
        Time in ISO 8601 format, at any precision.

    Returns
    -------
    out : ndarray of float
        Unix time: seconds since 1970-01-01T00:00:00, without leap seconds,
        always 1-D. "NaT" gives NaN.

    Notes
    -----
    A float64 resolves about 0.24 microseconds for current dates. Use
    pyrfu.pyrf.iso86012datetime64 to keep nanoseconds.

    See Also
    --------
    pyrfu.pyrf.iso86012datetime64, pyrfu.pyrf.datetime642unix

    """

    assert isinstance(time, (str, list, np.ndarray)), "time must be a str or array_like"

    time = np.atleast_1d(time).astype("datetime64[ns]")

    out = time.astype(np.int64) / 1e9
    out[np.isnat(time)] = np.nan

    return out
