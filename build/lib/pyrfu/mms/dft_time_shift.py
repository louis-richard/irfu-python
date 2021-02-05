#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""dft_time_shift.py
@author: Louis Richard
"""

import numpy as np

from ..pyrf import calc_fs, ts_scalar


def dft_time_shift(inp, tau):
    """Shifts the input signal ``inp`` by ``tau`` seconds using
    discrete fourier transform (DFT). Particularly useful when
    calculating the frequency-wavenumber spectrum of the mms'
    spin-plane or axial probes.

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

    freq = ((np.arange(n_p) + np.floor(n_p / 2)) % n_p - np.floor(n_p / 2))
    freq /= n_p

    # Backward FFT
    out_data = np.fft.ifft(sig_fft * np.exp(-2j * np.pi * delay * freq))

    out_time = time + int(tau * 1e9)

    out = ts_scalar(out_time, out_data, attrs=inp.attrs)

    return out
