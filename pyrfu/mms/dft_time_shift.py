#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dft_time_shift(inp, tau):
    r"""Shifts the input signal ``inp`` by ``tau`` seconds using discrete
    fourier transform (DFT). Particularly useful when calculating the
    frequency-wavenumber spectrum of the mms' spin-plane or axial probes.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to be shifted (Note : Only tensor order 1).
    tau : float
        Applied shift in seconds.

    Returns
    -------
    out : xarray.DataArray
        Time series of the shifted input.

    See also
    --------
    pyrfu.mms.fk_power_spectrum : Calculates the frequency-wave number
    power spectrum.

    """

    time, sig = [inp.time.data, inp.data]

    # Sampling frequency
    f_sampling = calc_fs(inp)

    # Applied delay in samples.
    delay = np.floor(tau * f_sampling)

    # Forward FFT
    sig_fft = np.fft.fft(sig)
    n_p = len(sig)

    # Disregard Nyquist frequency for even-sized dft
    if not n_p % 2:
        sig_fft[int(n_p / 2 + 1)] = 0

    freq = (np.arange(n_p) + np.floor(n_p / 2)) % n_p - np.floor(n_p / 2)
    freq /= n_p

    # Backward FFT
    out_data = np.fft.ifft(sig_fft * np.exp(-2j * np.pi * delay * freq))

    out_time = time + int(tau * 1e9)

    out = ts_scalar(out_time, out_data, attrs=inp.attrs)

    return out
