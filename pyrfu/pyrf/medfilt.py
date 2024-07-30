#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import signal
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def medfilt(inp: DataArray, kernel_size: Optional[int] = None) -> DataArray:
    r"""Applies a median filter over npts points to inp.

    Parameters
    ----------
    inp : DataArray
        Time series of the input variable.
    kernel_size : int, Optional
        Number of points of median filter. Default is a 3-point median filter.

    Returns
    -------
    DataArray
        Time series of the median filtered input variable.

    Raises
    ------
    TypeError
        If inp is not a DataArray.
    ValueError
        If inp is not 1D, 2D or 3D.

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

    # Get number of times
    n_times: int = len(inp)

    # Check kernel size
    if kernel_size is None:
        # Set default kernel size to 3
        kernel_size = 3
    elif kernel_size % 2 == 0:
        # If kernel size is even, add 1.
        kernel_size += 1

    # Check if input is 1D, 2D or 3D
    if inp.ndim == 1:
        # Add a dimension to the input if it is 1D
        inp_data: NDArray[np.float32] = inp.data[:, np.newaxis]
    elif inp.ndim == 2:
        # Keep input as is if it is 2D
        inp_data = inp.data
    elif inp.ndim == 3:
        # Reshape input if it is 3D to 2D (n_times, 9)
        inp_data = np.reshape(inp.data, [n_times, 9])
    else:
        raise ValueError("Input must be 1D, 2D or 3D")

    # Preallocate output
    out_data: NDArray[np.float32] = np.zeros(inp_data.shape, dtype=np.float32)

    # Apply median filter
    for i in range(inp_data.shape[1]):
        out_data[:, i] = signal.medfilt(inp_data[:, i], kernel_size)

    # Reshape output if input was 3D
    if inp_data.shape[1] == 9:
        out_data = np.reshape(out_data, [n_times, 3, 3])

    # Create output DataArray
    out: DataArray = xr.DataArray(
        np.squeeze(out_data), coords=inp.coords, dims=inp.dims, attrs=inp.attrs
    )

    return out
