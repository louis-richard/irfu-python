#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .datetime642unix import datetime642unix
from .iso86012datetime64 import iso86012datetime64


def iso86012unix(time):
    r"""Converts time in ISO 8601 format to Unix time.

    Parameters
    ----------
    time : str or array_like of str
        Time in ISO 8601 format, at any precision, with or without the UTC
        designator "Z" (list, tuple or numpy.ndarray), as
        `iso86012datetime64`.

    Returns
    -------
    out : ndarray of float
        Unix time: seconds since 1970-01-01T00:00:00, without leap seconds,
        always 1-D. "NaT" gives NaN.

    Raises
    ------
    TypeError
        If time is not a str or array_like.

    Notes
    -----
    A float64 resolves about 0.24 microseconds for current dates. Use
    pyrfu.pyrf.iso86012datetime64 to keep nanoseconds.

    See Also
    --------
    pyrfu.pyrf.iso86012datetime64, pyrfu.pyrf.datetime642unix

    """

    if not isinstance(time, (str, list, tuple, np.ndarray)):
        raise TypeError("time must be a str or array_like")

    # The same parsing as iso86012datetime64 (tuples raised TypeError and "Z"
    # gave a numpy warning)
    return datetime642unix(iso86012datetime64(time))
