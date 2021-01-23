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

import numpy as np

from scipy import signal


def wave_fft(x, window, frame_overlap=10., frame_length=20., fs=None):
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
    s : ndarray
        Spectrogram of x.

    t : ndarray
        Value corresponds to the center of each frame (x-axis) in sec.

    f : ndarray
        Vector of frequencies (y-axis) in Hz.

    """

    if fs is None:
        dt = np.median(np.diff(x.time.data).astype(float)) * 1e-9
        fs = 1 / dt

    n_per_seg = np.round(frame_length * fs).astype(int)  # convert ms to points
    n_overlap = np.round(frame_overlap * fs).astype(int)  # convert ms to points

    options = dict(fs=fs, window=window, nperseg=n_per_seg, noverlap=n_overlap, mode='complex')
    f, t, s = signal.spectrogram(x, **options)

    return f, t, s
