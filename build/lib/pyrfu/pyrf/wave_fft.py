#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wave_fft.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from scipy import signal


def wave_fft(x=None, window="hamming", frame_overlap=10, frame_length=20, fs=None):
    """Short-Time Fourier Transform.

    Parameters
    ----------
    x : xarray.DataArray
        Time series of the one dimension data.

    window : str
        Window function such as rectwin, hamming (default).

    frame_overlap : float
        Length of each frame overlaps in second.

    frame_length : float
        Length of each frame in second.

    fs : float
        Sampling frequency.

    Returns
    -------
    s : numpy.ndarray
        Spectrogram of x.

    t : numpy.ndarray
        Value corresponds to the center of each frame (x-axis) in sec.

    f : numpy.ndarray
        Vector of frequencies (y-axis) in Hz.

    """

    assert x is not None and isinstance(x, xr.DataArray)

    if fs is None:
        dt = np.median(np.diff(x.time.data).astype(float)) * 1e-9
        fs = 1 / dt

    n_per_seg = np.round(frame_length * fs).astype(int)  # convert ms to points
    n_overlap = np.round(frame_overlap * fs).astype(int)  # convert ms to points

    f, t, s = signal.spectrogram(x, fs=fs, window=window, nperseg=n_per_seg, noverlap=n_overlap,
                                 mode='complex')

    return f, t, s
