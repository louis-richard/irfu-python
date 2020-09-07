#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wavefft.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from scipy import signal


def waveftt(x=None, window="hamming", frame_overlap=10, frame_length=20, fs=None):
    """
    Short-Time Fourier Transform

    Parameters :
        x : DataArray
            Time series of the one dimension data

        window : str
            Window function such as rectwin, hamming (default)

        frame_overlap : float
            Length of each frame overlaps in second

        frame_length : float
            Length of each frame in second.

        fs : float
            Sampling frequency

    Return :
        s : array
            Spectrogram of x

        t : array
            Value corresponds to the center of each frame (x-axis) in sec

        f : array
            Vector of frequencies (y-axis) in Hz

    """

    if not isinstance(x, xr.DataArray):
        raise TypeError("x must be a DataArray")

    if fs is None:
        dt = np.median(np.diff(x.time.data).astype(float)) * 1e-9
        fs = 1 / dt

    nperseg = np.round(frame_length * fs).astype(int)  # convert ms to points
    noverlap = np.round(frame_overlap * fs).astype(int)  # convert ms to points

    f, t, s = signal.spectrogram(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, mode='complex')

    return f, t, s
