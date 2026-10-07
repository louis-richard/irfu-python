#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
import pycdfpp
from numpy.typing import NDArray

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# CDF_TT2000 fill and pad values, which have no datetime64 equivalent
TT2000_FILL = np.iinfo(np.int64).min
TT2000_PAD = TT2000_FILL + 1

# pycdfpp only converts arrays of its own CDF_TT2000 type
_TT2000_DTYPE = np.dtype([("nseconds", "<i8")])

# CDF_TT2000 spans 1707-09-22 to 2292-04-11 and datetime64[ns] spans
# 1677-09-21 to 2262-04-11, so only the end of CDF_TT2000 does not fit
_DATETIME64_NS_MAX = np.datetime64(np.iinfo(np.int64).max, "ns")
TT2000_MAX = int(
    pycdfpp.to_tt2000(np.array([_DATETIME64_NS_MAX])).view(np.int64)[0],
)


def ttns2datetime64(
    time: Union[int, float, list, NDArray[np.int64]],
) -> NDArray[np.datetime64]:
    r"""Convert time in epoch_tt2000 (nanoseconds since J2000) to datetime64
    in ns units.

    Leap seconds are taken into account. A time inside a leap second
    (23:59:60.x UTC), which datetime64 cannot represent, maps to 00:00:00.x of
    the next day. The CDF fill and pad values, and times after
    2262-04-11T23:47:16.854775807 (the end of the datetime64[ns] range), map
    to NaT.

    Parameters
    ----------
    time : int or float or list or numpy.ndarray
        Time in epoch_tt2000 (nanoseconds since J2000) format.

    Returns
    -------
    time_datetime64 : numpy.ndarray
        Time in datetime64 format in ns units.

    Raises
    ------
    TypeError
        If time is not a float, an int or array_like.

    """

    if not isinstance(time, (float, int, list, np.ndarray)):
        raise TypeError("time must be float, int, or array_like")

    time_tt2000 = np.ascontiguousarray(np.atleast_1d(time), dtype=np.int64)

    time_datetime64 = pycdfpp.to_datetime64(time_tt2000.view(_TT2000_DTYPE))
    time_datetime64 = time_datetime64.astype("datetime64[ns]")

    # pycdfpp wraps times beyond the datetime64[ns] range around silently
    is_nat = np.isin(time_tt2000, [TT2000_FILL, TT2000_PAD])
    is_nat |= time_tt2000 > TT2000_MAX
    time_datetime64[is_nat] = np.datetime64("NaT", "ns")

    return time_datetime64
