#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import signal
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def medfilt(inp: DataArray, kernel_size: Optional[int] = None) -> DataArray:
    r"""Applies a median filter over `kernel_size` points to each component
    of the time series `inp` (scipy.signal.medfilt, as MATLAB medfilt1).

    Parameters
    ----------
    inp : DataArray
        Time series of the input variable, with time as first dimension.
    kernel_size : int, Optional
        Number of points of the median filter, made odd by adding 1 if even;
        a float is rounded (e.g. a sampling frequency, for 1 s). Default is
        3.

    Returns
    -------
    DataArray
        Time series of the median filtered input, in double precision, with
        the coordinates and attributes of `inp`.

    Raises
    ------
    TypeError
        If inp is not a DataArray.

    Notes
    -----
    The edges are zero-padded, as scipy.signal.medfilt and MATLAB medfilt1:
    the first and last kernel_size // 2 points are biased towards 0. NaNs
    are not handled (remove or fill them first).

    Examples
    --------
    >>> import numpy
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_list = numpy.arange(1,5)

    Load magnetic field and electric field

    >>> r_mms, b_mms = [[] * 4 for _ in range(2)]
    >>> for mms_id in range(1, 5):
    >>> 	r_mms.append(mms.get_data("R_gse", tint, mms_id))
    >>> 	b_mms.append(mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id))
    >>>

    Compute current density, etc

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Get J sampling frequency

    >>> fs = pyrf.calc_fs(j_xyz)

    Median filter over 1s

    >>> j_xyz = pyrf.medfilt(j_xyz,fs)

    """

    # Check input
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    # Odd number of points
    kernel_size = 3 if kernel_size is None else int(np.round(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    # Each component (all the dimensions after time) in double precision
    inp_data = inp.data.astype(np.float64).reshape(len(inp), -1)
    out_data = np.empty_like(inp_data)

    for i in range(inp_data.shape[1]):
        out_data[:, i] = signal.medfilt(inp_data[:, i], kernel_size)

    out = inp.copy(data=out_data.reshape(inp.shape))

    return out
