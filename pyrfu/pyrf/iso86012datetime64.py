#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def iso86012datetime64(time: NDArray[np.str_]) -> NDArray[np.datetime64]:
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : numpy.ndarray
        Time in ISO 8601 format

    Returns
    -------
    time_datetime64 : numpy.ndarray
        Time in datetime64 in ns units.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    time_datetime64 = time.astype("datetime64[ns]")

    return time_datetime64
