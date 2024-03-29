#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd oarty imports
import numpy as np

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def fft_bandpass(inp, f_min, f_max):
    r"""Perform simple bandpass using FFT - returns fields between with
    ``f_min`` < f < ``f_max``.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to be bandpassed filtered.
    f_min : float or int
        Minimum frequency of filter, f < ``f_min`` are removed.
    f_max : float or int
        Maximum frequency of filter, f > ``f_max`` are removed.

    Returns
    -------
    out : xarray.DataArray
        Time series of the bandpassed filtered data.

    Notes
    -----
    Can be some spurius effects near boundary. Can take longer interval then
    use tlim to remove.

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

    inp_time, inp_data = [inp.time.data, inp.data]

    # Make sure number of elements is an even number, if odd remove last
    # element to make an even
    # number
    n_els = len(inp_data)

    if n_els % 2:
        inp_time = inp_time[:-1]
        inp_data = inp_data[:-1, :]

        n_els = len(inp_data)

    try:
        num_fields = inp_data.shape[1]
    except IndexError:
        num_fields = 1
        inp_data = inp_data[:, np.newaxis]

    # Set NaN values to zero so FFT works
    idx_nans = np.isnan(inp_data)
    inp_data[idx_nans] = 0.0

    # Bandpass filter field data
    f_sam = calc_fs(inp)
    f_nyq = f_sam / 2
    frequencies = np.linspace(-f_nyq, f_nyq, n_els)

    # FFT and remove frequencies
    for i in range(num_fields):
        inp_temp = np.fft.fft(inp_data[:, i])
        inp_temp = np.fft.fftshift(inp_temp)

        inp_temp[np.abs(frequencies) < f_min] = 0
        inp_temp[np.abs(frequencies) > f_max] = 0

        inp_temp = np.fft.ifftshift(inp_temp)
        inp_data[:, i] = np.fft.ifft(inp_temp)

    # Put back original NaNs
    inp_data[idx_nans] = np.nan

    # Return data in the same format as input
    if num_fields == 1:
        out = ts_scalar(inp_time, inp_data, attrs=inp.attrs)
    elif num_fields == 3:
        out = ts_vec_xyz(inp_time, inp_data, attrs=inp.attrs)
    else:
        raise ValueError("Invalid shape")

    return out
