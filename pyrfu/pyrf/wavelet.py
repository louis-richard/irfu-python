#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os

# 3rd party imports
import numpy as np
import xarray as xr
import numba

from scipy import fft

# Local imports
from .calc_fs import calc_fs

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _scales(f_range, f_nyq, f_s, n_freqs, linear_df, delta_f):
    if linear_df:
        scale_number = np.floor(f_nyq / delta_f).astype(int)

        f_range = [delta_f, scale_number * delta_f]

        scales = f_nyq / (np.linspace(f_range[0], f_range[1], scale_number))
    else:
        scale_min = np.log10(.5 * f_s / f_range[1])
        scale_max = np.log10(.5 * f_s / f_range[0])
        scales = np.logspace(scale_min, scale_max, n_freqs)

    return scales


@numba.jit(nopython=True, fastmath=True)
def _ww(s_ww, scales_mat, sigma, frequencies_mat, f_nyq):
    w_w = s_ww * np.exp(-sigma * sigma
                        * ((scales_mat * frequencies_mat - f_nyq) ** 2) / 2)
    w_w = w_w * np.sqrt(1)
    return w_w


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _power_r(power, new_freq_mat):
    power2 = np.absolute((2 * np.pi) * np.conj(power) * power / new_freq_mat)
    return power2


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _power_c(power, new_freq_mat):
    power2 = np.sqrt(np.absolute((2 * np.pi) / new_freq_mat)) * power
    return power2


def wavelet(inp, f_range: list = None, n_freqs: int = 200,
            w_width: float = 5.36, delta_f: float = 100.,
            linear_df: bool = False, cut_edge: bool = True,
            return_power: bool = True):
    r"""Computes wavelet spectrogram based on fast FFT algorithm.

    Parameters
    ----------
    inp : xarray.DataArray
        Input quantity.
    f_range : list, Optional
        Vector [f_min f_max], calculate spectra between frequencies f_min
        and f_max.
    n_freqs : int, Optional
        Number of frequency bins. Default is 200.
    w_width : float, Optional
        Width of the Morlet wavelet. Default 5.36.
    delta_f : float, Optional
        Spacing between frequencies. Default is 100.
    linear_df : bool, Optional
        Linear spacing between frequencies of df. Default is False (use
        log spacing)
    return_power : bool, Optional
        Set to True to return the power, False for complex wavelet
        transform. Default True.
    cut_edge : bool, Optional
        Set to True to set points affected by edge effects to NaN, False to
        keep edge affect points. Default True.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Wavelet transform of the input.

    """

    # Unpack time and data
    data = inp.data

    # Compute sampling frequency
    f_s = calc_fs(inp)

    scale_min, scale_max = [0.01, 2]

    if f_range is None:
        f_range = [.5 * f_s / 10 ** scale_max, .5 * f_s / 10 ** scale_min]

    f_nyq, scale_number, sigma = [f_s / 2, n_freqs, w_width / (f_s / 2)]

    scales = _scales(f_range, f_nyq, f_s, n_freqs, linear_df, delta_f)

    # Remove the last sample if the total number of samples is odd
    if len(data) / 2 != np.floor(len(data) / 2):
        time = inp.time.data[:-1]
        data = data[:-1, ...]
    else:
        time = inp.time.data

    # Check for NaNs
    scales[np.isnan(scales)] = 0

    # Find the frequencies for an FFT of all data
    freq = f_s * .5 * np.arange(1, 1 + len(data) / 2) / (len(data) / 2)

    # The frequencies corresponding to FFT
    frequencies = np.hstack([0, freq, -np.flip(freq[:-1])])

    # Get the correct frequencies for the wavelet transform
    new_freq = f_nyq / scales

    new_freq_mat, temp_freq = np.meshgrid(new_freq, frequencies, sparse=True)

    _, frequencies_mat = np.meshgrid(scales, frequencies, sparse=True)

    if len(inp.shape) == 1:
        data = data[:, np.newaxis]  # if scalar add virtual axis
        out_dict, power2 = [None, np.zeros((len(inp.data), n_freqs))]
    elif len(inp.shape) == 2:
        out_dict, power2 = [{}, None]
    else:
        raise TypeError("Invalid shape of the inp")

    # go through all the data columns
    for i in range(data.shape[1]):
        # Make the FFT of all data
        data_col = data[:, i]

        # Wavelet transform of the data
        # Forward FFT
        s_w = fft.fft(data_col, workers=os.cpu_count())

        scales_mat, s_w_mat = np.meshgrid(scales, s_w, sparse=True)

        # Calculate the FFT of the wavelet transform
        w_w = _ww(s_w_mat, scales_mat, sigma, frequencies_mat, f_nyq)

        # Backward FFT
        power = fft.ifft(w_w, axis=0, workers=os.cpu_count())

        # Calculate the power spectrum
        if return_power:
            power2 = _power_r(power, np.tile(new_freq_mat, (len(power), 1)))
        else:
            power2 = _power_c(power, np.tile(new_freq_mat, (len(power), 1)))

        if cut_edge:
            censure = np.floor(2 * scales).astype(int)

            for j in range(scale_number):
                power2[:censure[j], j] = np.nan

                power2[len(data_col) - censure[j]:len(data_col), j] = np.nan
        if len(inp.shape) == 2:
            out_dict[inp.comp.data[i]] = (["time", "frequency"], power2)

    if len(inp.shape) == 1:
        out = xr.DataArray(power2, coords=[time, new_freq],
                           dims=["time", "frequency"])
    elif len(inp.shape) == 2:
        out = xr.Dataset(out_dict,
                         coords={"time": time, "frequency": new_freq})
    else:
        raise TypeError("Invalid shape")

    return out
