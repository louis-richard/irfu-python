#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
from typing import Dict, Optional, Union

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import fft
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from .calc_fs import calc_fs

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


@numba.jit(nopython=True, fastmath=True)  # type: ignore
def _ww(
    s_ww: NDArray[np.complex128],
    scales_mat: NDArray[np.float64],
    sigma: float,
    frequencies_mat: NDArray[np.float64],
    f_nyq: float,
) -> NDArray[np.complex128]:
    # TODO : use nested for loop and math instead of numpy and test speed!!
    w_w: NDArray[np.complex128] = s_ww * np.exp(
        -sigma * sigma * ((scales_mat * frequencies_mat - f_nyq) ** 2) / 2,
    )
    w_w = w_w * np.sqrt(1.0)
    return w_w


@numba.jit(nopython=True, parallel=True, fastmath=True)  # type: ignore
def _power_r(
    power: NDArray[np.complex128], new_freq_mat: NDArray[np.float64]
) -> NDArray[np.float64]:
    power2: NDArray[np.float64] = np.absolute(
        (2 * np.pi) * np.conj(power) * power / new_freq_mat
    )
    return power2


@numba.jit(nopython=True, parallel=True, fastmath=True)  # type: ignore
def _power_c(
    power: NDArray[np.complex128], new_freq_mat: NDArray[np.float64]
) -> NDArray[np.complex128]:
    power2: NDArray[np.complex128] = (
        np.sqrt(np.absolute((2 * np.pi) / new_freq_mat)) * power
    )
    return power2


def wavelet(
    inp: DataArray,
    f_s: Optional[float] = None,
    f: Optional[list[float]] = None,
    n_freqs: Optional[int] = None,
    linear: Optional[Union[float, bool]] = None,
    wavelet_width: Optional[float] = None,
    cut_edge: Optional[bool] = True,
    return_power: Optional[bool] = True,
) -> Union[DataArray, Dataset]:
    """Computes wavelet spectrogram based on fast FFT algorithm.
    Parameters
    ----------
    inp : DataArray
        Input quantity.
    f_s : float, Optional
        Sampling frequency of the input time series.
    f : list, Optional
        Vector [f_min f_max], calculate spectra between frequencies
        f_min and f_max.
    n_freqs : int, Optional
        Number of frequency bins.
    linear : float or bool, Optional
        Linear spacing between frequencies of df.
    wavelet_width : float, Optional
        Width of the Morlet wavelet. Default 5.36.
    cut_edge : bool, Optional
        Set to True to set points affected by edge effects to NaN,
        False to keep edge affect points. Default True
    return_power : bool, Optional
        Set to True to return the power, False for complex wavelet
        transform. Default True.

    Returns
    -------
    DataArray or Dataset
        Wavelet transform of the input.


    Raises
    ------
    TypeError
        If linear keyword argument is not bool or float.
    ValueError
        If input is not 1D or 2D.

    """

    # Check input
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    if f_s is None:
        f_s = calc_fs(inp)

    if n_freqs is None:
        n_freqs = 200

    if wavelet_width is None:
        wavelet_width = 5.36

    if linear is not None:
        if isinstance(linear, float):
            delta_f: float = linear
            linear_df: bool = True
        elif isinstance(linear, bool) and linear:
            delta_f = 100.0
            linear_df = True
            logging.warning("Unknown input for linear delta_f set to 100")
        else:
            raise TypeError("linear keyword argument must be bool or float")
    else:
        delta_f = 100.0
        linear_df = False

    # Nyquist frequency and wavelet width
    f_nyq: float = f_s / 2
    sigma: float = wavelet_width / f_nyq

    # Frequency range
    if f is None:
        f_min: float = f_nyq / 10**2
        f_max: float = f_nyq / 10**-2
    else:
        f_min, f_max = sorted(f)

    if linear_df:
        scale_number: int = int(np.floor(f_nyq / delta_f))

        # Scales range
        scale_min: float = delta_f
        scale_max: float = scale_number * delta_f
        scales: NDArray[np.float64] = f_nyq / (
            np.linspace(scale_max, scale_min, scale_number, dtype=np.float64)
        )
    else:
        scale_number = n_freqs
        scale_min = np.log10(f_nyq / f_max)
        scale_max = np.log10(f_nyq / f_min)
        scales = np.logspace(scale_min, scale_max, scale_number, dtype=np.float64)

    # Unpack time and data.
    # Remove the last sample if the total number of samples is odd.
    if len(inp.time.data) % 2:
        time: NDArray[np.datetime64] = inp.time.data[:-1]
        data: NDArray[np.float64] = inp.data[:-1, ...].astype(np.float64)
    else:
        time = inp.time.data
        data = inp.data.astype(np.float64)

    # Preallocate power2
    if return_power:
        power2: NDArray[Union[np.float64, np.complex128]] = np.zeros(
            (len(time), n_freqs), dtype=np.float64
        )
    else:
        power2 = np.zeros((len(time), n_freqs), dtype=np.complex128)

    # Check for NaNs
    scales[np.isnan(scales)] = 0.0

    # Find the frequencies for an FFT of all data
    freq: NDArray[np.float64] = (
        f_nyq * np.arange(1, 1 + len(data) / 2) / (len(data) / 2)
    )

    # The frequencies corresponding to FFT
    freqs_fft: NDArray[np.float64] = np.hstack([0, freq, -np.flip(freq[:-1])])
    _, freqs_fft_mat = np.meshgrid(scales, freqs_fft, sparse=True)

    # Get the correct frequencies for the wavelet transform
    freqs_cwt: NDArray[np.float64] = f_nyq / scales
    freqs_cwt_mat, _ = np.meshgrid(freqs_cwt, freqs_fft, sparse=True)

    if data.ndim in [1, 2]:
        out_dict: Dict[str, object] = {}
    else:
        raise ValueError("Input data must be 1D or 2D")

    # if scalar add virtual axis
    if len(inp.shape) == 1:
        data = data[:, np.newaxis]

    # go through all the data columns
    for i in range(data.shape[1]):
        # Make the FFT of all data
        data_col: NDArray[np.float64] = data[:, i]

        # Wavelet transform of the data
        # Forward FFT
        s_w: NDArray[np.complex128] = fft.fft(data_col, workers=os.cpu_count())

        scales_mat, s_w_mat = np.meshgrid(scales, s_w, sparse=True)

        # Calculate the FFT of the wavelet transform
        w_w: NDArray[np.complex128] = _ww(
            s_w_mat, scales_mat, sigma, freqs_fft_mat, f_nyq
        )

        # Backward FFT
        power: NDArray[np.complex128] = fft.ifft(w_w, axis=0, workers=os.cpu_count())

        # Calculate the power spectrum
        if return_power:
            power2 = _power_r(power, np.tile(freqs_cwt_mat, (len(power), 1)))
        else:
            power2 = _power_c(power, np.tile(freqs_cwt_mat, (len(power), 1)))

        if cut_edge:
            censure = np.floor(2 * scales).astype(int)

            for j in range(scale_number):
                power2[: censure[j], j] = np.nan

                power2[len(data_col) - censure[j] : len(data_col), j] = np.nan

        if len(inp.shape) == 2:
            # Construct xarray.DataArray here
            out_dict[str(inp.comp.data[i])] = (
                ["time", "frequency"],
                np.fliplr(power2),
            )

    if len(inp.shape) == 1:
        out: Union[DataArray, Dataset] = xr.DataArray(
            np.fliplr(power2),
            coords=[time, np.flip(freqs_cwt)],
            dims=["time", "frequency"],
        )
    else:
        out = xr.Dataset(
            out_dict,
            coords={"time": time, "frequency": np.flip(freqs_cwt)},
        )

    return out
