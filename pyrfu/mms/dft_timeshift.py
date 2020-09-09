#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dft_timeshift.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from ..pyrf import calc_fs, ts_scalar


def dft_timeshift(inp=None, tau=None):
    """
    Shifts the input signal "sig" by "tau" seconds using discrete fourier transform (DFT).
    Particularly useful when calculating the frequency-wavenumber spectrum of the mms' spin-plane or axial probes.
    See: mms.fk_powerspectrum.

    Parameters :
        inp : DataArray
            Time series to be shifted (Note : Only tensor order 1)

        tau : float
            Applied shift in seconds

    Returns :
        out : DataArray
            Time series of the shifted input.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)
    assert tau is not None and isinstance(tau, float)

    time, sig = [inp.time.data, inp.data]

    # Sampling frequency
    fs = calc_fs(inp)

    # Applied delay in samples.
    ds = np.floor(tau * fs)

    # Forward FFT
    u = np.fft.fft(sig)
    n_p = len(sig)

    # Disregard Nyquist frequency for even-sized dft
    if not n_p % 2:
        u[int(n_p / 2 + 1)] = 0

    f = ((np.arange(n_p) + np.floor(n_p / 2)) % n_p - np.floor(n_p / 2)) / n_p

    # Backward FFT
    out_data = np.fft.ifft(u * np.exp(-2j * np.pi * ds * f))

    out_time = time + int(tau * 1e9)

    out = ts_scalar(out_time, out_data, attrs=inp.attrs)

    return out
