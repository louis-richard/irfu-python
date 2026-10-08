#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from .ttns2datetime64 import ttns2datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Seconds from 0000-01-01T00:00:00 (CDF_EPOCH and CDF_EPOCH16 origin) to the
# Unix epoch 1970-01-01T00:00:00
_EPOCH0_TO_UNIX_S = 62167219200

# datetime64[ns] spans +/- 2**63 ns around 1970 (1677-09-21 to 2262-04-11).
# The 1 ms and 1 s margins keep the integer arithmetic below from overflowing.
_EPOCH_MAX_MS = np.iinfo(np.int64).max // 10**6 - 1
_EPOCH16_MAX_S = np.iinfo(np.int64).max // 10**9 - 1


def _epoch2datetime64(epochs):
    # CDF_EPOCH: milliseconds since 0000-01-01, without leap seconds. The
    # difference with the Unix offset is exact in float64, then integer
    # arithmetic keeps the nanoseconds exact. NaN, the fill (-1e31) and pad
    # (0.0) values, and times outside the datetime64[ns] range map to NaT.
    millis = epochs - _EPOCH0_TO_UNIX_S * 1e3
    is_nat = ~(np.abs(millis) <= _EPOCH_MAX_MS)
    millis[is_nat] = 0.0

    whole = np.floor(millis)
    nanos = whole.astype(np.int64) * 10**6
    nanos += np.round((millis - whole) * 1e6).astype(np.int64)

    times = nanos.astype("datetime64[ns]")
    times[is_nat] = np.datetime64("NaT", "ns")
    return times


def _epoch162datetime64(epochs):
    # CDF_EPOCH16: seconds since 0000-01-01 (real part), without leap
    # seconds, and picoseconds within the second (imaginary part). NaN, the
    # fill (-1e31) and pad (0.0) values, and times outside the datetime64[ns]
    # range map to NaT.
    seconds = epochs.real - _EPOCH0_TO_UNIX_S
    picos = epochs.imag.copy()
    is_nat = ~(np.abs(seconds) <= _EPOCH16_MAX_S)
    is_nat |= ~((picos >= 0.0) & (picos < 1e12))
    seconds[is_nat], picos[is_nat] = 0.0, 0.0

    nanos = seconds.astype(np.int64) * 10**9
    nanos += np.floor(picos / 1e3).astype(np.int64)

    times = nanos.astype("datetime64[ns]")
    times[is_nat] = np.datetime64("NaT", "ns")
    return times


def cdfepoch2datetime64(
    epochs: Union[int, float, complex, list, NDArray],
) -> NDArray[np.datetime64]:
    r"""Converts CDF epochs to numpy.datetime64 with nanosecond precision.

    The CDF time type is inferred from the data type:

    * integers are CDF_TT2000 (nanoseconds since J2000, with leap seconds),
    * floats are CDF_EPOCH (milliseconds since 0000-01-01),
    * complex numbers are CDF_EPOCH16 (seconds since 0000-01-01 and
      picoseconds).

    Fill and pad values, NaN, and times outside the datetime64[ns] range
    (1677-09-21 to 2262-04-11) map to NaT.

    Parameters
    ----------
    epochs : int or float or complex or array_like
        CDF epochs to convert.

    Returns
    -------
    times : numpy.ndarray
        Array of times in datetime64([ns]).

    Raises
    ------
    TypeError
        If epochs are not integers, floats or complex numbers.

    See Also
    --------
    pyrfu.pyrf.ttns2datetime64

    """

    epochs = np.atleast_1d(np.asarray(epochs))

    if np.issubdtype(epochs.dtype, np.integer):
        times = ttns2datetime64(epochs)
    elif np.issubdtype(epochs.dtype, np.floating):
        times = _epoch2datetime64(epochs.astype(np.float64))
    elif np.issubdtype(epochs.dtype, np.complexfloating):
        times = _epoch162datetime64(epochs.astype(np.complex128))
    else:
        raise TypeError("epochs must be CDF_TT2000, CDF_EPOCH or CDF_EPOCH16")

    return times
