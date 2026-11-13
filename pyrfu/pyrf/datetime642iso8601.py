#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
import numpy as np
from numpy.typing import NDArray


def datetime642iso8601(
    time: np.datetime64 | NDArray[np.datetime64],
) -> NDArray[np.str_]:
    r"""Convert datetime64 to ISO 8601 time format, with nanoseconds.

    Parameters
    ----------
    time : numpy.datetime64 or numpy.ndarray
        Time in datetime64, in any unit (arrays in s, min, h or D raised
        TypeError).

    Returns
    -------
    time_iso8601 : numpy.ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """
    if isinstance(time, np.datetime64):
        time_datetime64 = np.atleast_1d(time).astype("datetime64[ns]")
    elif isinstance(time, np.ndarray) and time.dtype.kind == "M":
        time_datetime64 = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be numpy.datetime64 or numpy.ndarray")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601
