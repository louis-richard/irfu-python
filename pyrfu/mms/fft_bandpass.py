#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import numbers
from typing import Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.calc_fs import calc_fs
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def fft_bandpass(inp: DataArray, f_min: float, f_max: float) -> DataArray:
    r"""Perform simple bandpass using FFT - returns fields between with
    ``f_min`` < f < ``f_max``.

    Parameters
    ----------
    inp : DataArray
        Time series to be bandpass filtered.
    f_min : float
        Minimum frequency of filter in Hz, f < ``f_min`` are removed.
    f_max : float
        Maximum frequency of filter in Hz, f > ``f_max`` are removed.

    Returns
    -------
    DataArray
        Time series of the bandpass filtered data.

    Raises
    ------
    TypeError
        * If input is not a xarray.DataArray.
        * If f_min or f_max is not a real number.
    ValueError
        If f_min is larger than f_max.

    Notes
    -----
    Can be some spurious effects near boundary. Can take longer interval then
    use pyrfu.pyrf.time_clip to remove. With an odd number of samples, the
    last one is dropped (as irfu-matlab ``mms.fftbandpass``).

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load Electric Field

    >>> e_xyz = mms.get_data("e_gse_edp_brst_l2", tint, mms_id)

    Bandpass filter

    >>> e_xyz_bp = mms.fft_bandpass(e_xyz, 1e1, 1e2)

    """

    # Make sure input is a DataArray
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a xarray.DataArray")

    # Check f_min and f_max (any real number, e.g., an int)
    for name, value in [("f_min", f_min), ("f_max", f_max)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")

    f_min, f_max = float(f_min), float(f_max)

    # Check that f_min < f_max
    if f_min >= f_max:
        raise ValueError("f_min must be smaller than f_max")

    # Get time and data
    inp_time: NDArray[np.datetime64] = inp.time.data
    inp_data: NDArrayFloats = inp.data
    precision = inp_data.dtype

    # Reshape to column vector if input is a scalar
    if inp_data.ndim == 1:
        # If scalar, reshape to column vector
        inp_data = inp_data[:, np.newaxis]
    elif inp_data.ndim > 2:
        raise ValueError("Input must be a scalar or a vector")

    # Make sure number of elements is an even number, if odd remove last
    # element to make an even number
    if len(inp_time) % 2:
        inp_time = inp_time[:-1]
        inp_data = inp_data[:-1, :]

    # Set NaN values to zero so FFT works (in a new array: inp_data is a view of
    # the caller's data)
    idx_nans = np.isnan(inp_data)
    inp_data = np.where(idx_nans, 0.0, inp_data)

    # Bandpass filter field data: the frequencies of the FFT bins (as
    # irfu-matlab, -f_nyq:df:f_nyq-df once shifted; they were spaced by
    # 2 f_nyq / (N - 1), so that the positive and negative frequencies of a
    # component could fall on either side of a cutoff, which halved it)
    f_sam = calc_fs(inp)
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(inp_time), 1 / f_sam))

    # Preallocate output
    out_data_64: NDArray[np.float64] = np.empty_like(inp_data, dtype=np.float64)

    # FFT and remove frequencies
    for i in range(inp_data.shape[1]):
        inp_tmp: NDArray[np.float64] = inp_data[:, i].astype(np.float64)
        inp_fft: NDArray[np.complex128] = np.fft.fft(inp_tmp)
        inp_fft = np.fft.fftshift(inp_fft)

        inp_fft[np.abs(frequencies) < f_min] = 0.0 + 0.0j
        inp_fft[np.abs(frequencies) > f_max] = 0.0 + 0.0j

        inp_fft = np.fft.ifftshift(inp_fft)
        out_tmp = np.fft.ifft(inp_fft)

        out_data_64[:, i] = np.real(out_tmp)

    # Put back original NaNs and back to original shape and precision
    out_data_64[idx_nans] = np.nan
    out_data_64 = np.squeeze(out_data_64)
    out_data: NDArrayFloats = out_data_64.astype(precision)

    # Return data in the same format as input
    if out_data.ndim == 1:
        out: DataArray = ts_scalar(inp_time, out_data, attrs=inp.attrs)
    else:
        out = ts_vec_xyz(inp_time, out_data, attrs=inp.attrs)

    return out
