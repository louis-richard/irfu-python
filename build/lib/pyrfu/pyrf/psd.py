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

"""psd.py
@author: Louis Richard
"""

import warnings
import numpy as np
import xarray as xr

from scipy import signal


def psd(inp, n_fft=256, n_overlap=128, window="hamming", d_flag="constant",
        scaling="density"):
    """Estimate power spectral density using Welch's method.

    Welch's method [11]_ computes an estimate of the power spectral
    density by dividing the data into overlapping segments, computing a
    modified periodogram for each segment and averaging the
    periodograms.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of measurement values.

    n_fft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.
        Default to 256.

    n_overlap : int, optional
        Number of points to overlap between segments. Default to 128.

    window : str, optional
        Desired window to use. It is passed to `get_window` to generate
        the window values, which are DFT-even by default.
        See "get_window" or a list of windows and required parameters.
        Default Hanning

    d_flag : str, optional
        Specifies how to detrend each segment. It is passed as the
        "type" argument to the"detrend" function.
        Default to "constant".

    scaling : str, optional
        Selects between computing the power spectral density
        ('density') where `Pxx` has units of V**2/Hz and computing the
        power spectrum ("spectrum") where "Pxx" has units of V**2, if
        `x` is measured in V and "fs" is measured in Hz.
        Default to 'density'

    Returns
    -------
    out : xarray.DataArray
        Power spectral density or power spectrum of inp.

    References
    ----------
    .. [11] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time
            averaging over short, modified periodograms",
            IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.

    """

    if inp.ndim == 2 and inp.shape[-1] == 3:
        inp = np.abs(inp)

    if n_overlap is None:
        n_persegs = 256
        n_overlap = n_persegs / 2
    else:
        n_persegs = 2 * n_overlap

    if n_fft < n_persegs:
        n_fft = n_persegs
        warnings.warn("nfft < n_persegs. set to n_persegs", UserWarning)

    f_samp = 1e9 / np.median(np.diff(inp.time.data)).astype(float)

    freqs, p_xx = signal.welch(inp.data, nfft=n_fft, fs=f_samp,
                               window=window, noverlap=n_overlap,
                               detrend=d_flag, nperseg=n_persegs,
                               scaling=scaling, return_onesided=True, axis=-1)

    out = xr.DataArray(p_xx, coords=[freqs], dims=["f"])

    return out
