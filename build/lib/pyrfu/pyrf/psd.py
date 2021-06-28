#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import signal

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def psd(inp, n_fft: int = 256, n_overlap: int = 128, window: str = "hamming",
        d_flag: str = "constant", scaling: str = "density"):
    r"""Estimate power spectral density using Welch's method.

    Welch's method [11]_ computes an estimate of the power spectral
    density by dividing the data into overlapping segments, computing a
    modified periodogram for each segment and averaging the
    periodograms.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of measurement values.
    n_fft : int, Optional
        Length of the FFT used, if a zero padded FFT is desired.
        Default to 256.
    n_overlap : int, Optional
        Number of points to overlap between segments. Default to 128.
    window : str, Optional
        Desired window to use. It is passed to `get_window` to generate
        the window values, which are DFT-even by default.
        See "get_window" or a list of windows and required parameters.
        Default Hanning
    d_flag : str, Optional
        Specifies how to detrend each segment. It is passed as the
        "type" argument to the"detrend" function. Default to "constant".
    scaling : str, Optional
        Selects between computing the power spectral density
        ('density') where `Pxx` has units of V**2/Hz and computing the
        power spectrum ("spectrum") where "Pxx" has units of V**2, if
        `x` is measured in V and "fs" is measured in Hz. Default to 'density'

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
