#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def iso86012datetime64(
    time: Union[str, list[str], tuple[str, ...], NDArray[np.str_]],
) -> NDArray[np.datetime64]:
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : str or array_like
        Time in ISO 8601 format, at any precision; a trailing "Z" (UTC) is
        accepted.

    Returns
    -------
    time_datetime64 : numpy.ndarray
        Time in datetime64 in ns units, 1-D (also for a single str).

    Raises
    ------
    TypeError
        If time is not a str, a list, a tuple or a numpy.ndarray.


    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """
    # Check input type
    if isinstance(time, (str, list, tuple, np.ndarray)):
        time_array = np.atleast_1d(np.asarray(time))
    else:
        raise TypeError("time must be a str, a list, a tuple or a numpy.ndarray")

    # The UTC designator "Z": numpy.datetime64 has no time zones (it warns,
    # and will raise)
    if time_array.dtype.kind == "U":
        time_array = np.char.rstrip(time_array, "Z")

    time_datetime64 = time_array.astype("datetime64[ns]")

    return time_datetime64
