#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from datetime import datetime
from typing import Any, cast

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from pyrfu._core.typing import TimeArray

from .datetime642iso8601 import datetime642iso8601


def datetime2iso8601(
    time: list[datetime] | NDArray | datetime | np.datetime64,
) -> list[str] | str:
    r"""Transforms times to ISO 8601 strings with nanoseconds.

    Parameters
    ----------
    time : datetime.datetime or numpy.datetime64 or array_like
        Time(s) to convert: ``datetime.datetime`` objects, or
        ``numpy.datetime64`` values (e.g., the times of a time series).

    Returns
    -------
    time_iso8601 : str or list of str
        Time(s) as "YYYY-MM-DDThh:mm:ss.mmmuuunnn": a str for a single time, a
        list for an array_like.

    Raises
    ------
    TypeError
        If the times are neither datetime.datetime nor numpy.datetime64.

    """

    # A list for an array_like (nested for nested arrays), a str for a time
    time_iso8601: list[Any] | str

    # numpy.datetime64 times (scalar or array), e.g. those of a time series
    if np.asarray(time).dtype.kind == "M":
        time_m = cast(np.datetime64 | TimeArray, time)
        time_iso8601 = [str(t_) for t_ in datetime642iso8601(np.atleast_1d(time_m))]

        return time_iso8601[0] if np.ndim(time_m) == 0 else time_iso8601

    if isinstance(time, (np.ndarray, list)):
        time_iso8601 = []

        for t in time:
            time_iso8601.append(datetime2iso8601(t))

    elif isinstance(time, datetime):
        # Through numpy.datetime64 (pandas was imported for the nanoseconds of
        # a datetime, always 0); the wall time, the time zone is not converted
        time_dt64 = np.datetime64(time.replace(tzinfo=None), "ns")
        time_iso8601 = str(datetime642iso8601(time_dt64)[0])

    else:
        raise TypeError("time must be datetime.datetime or numpy.datetime64")

    return time_iso8601
