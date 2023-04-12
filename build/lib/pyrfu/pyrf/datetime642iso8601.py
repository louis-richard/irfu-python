#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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

    # Convert to required precision
    time_datetime64 = time.astype("<M8[ns]")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601
