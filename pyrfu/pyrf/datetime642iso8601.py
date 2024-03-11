#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def datetime642iso8601(time):
    r"""Convert datetime64 in ns units to ISO 8601 time format .

    Parameters
    ----------
    time : ndarray
        Time in datetime64 in ns units.

    Returns
    -------
    time_iso8601 : ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    if isinstance(time, np.datetime64):
        time = np.array([time])
        time_datetime64 = time.astype("datetime64[ns]")
    elif isinstance(time, (list, np.ndarray)) and isinstance(time[0], np.datetime64):
        time_datetime64 = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be numpy.datetime64 or array_like")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601
