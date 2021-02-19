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

"""wave_fft.py
@author: Louis Richard
"""

import numpy as np

from scipy import signal


def wave_fft(inp, window, frame_overlap: float = 10.,
             frame_length: float = 20., f_sampling: float = None):
    """Short-Time Fourier Transform.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the one dimension data.

    window : str
        Window function such as rectwin, hamming (default).

    frame_overlap : float, optional
        Length of each frame overlaps in second.

    frame_length : float, optional
        Length of each frame in second.

    f_sampling : float, optional
        Sampling frequency.

    Returns
    -------
    spectrogram : ndarray
        Spectrogram of x.

    time : ndarray
        Value corresponds to the center of each frame (x-axis) in sec.

    frequencies : ndarray
        Vector of frequencies (y-axis) in Hz.

    """

    if f_sampling is None:
        delta_t = np.median(np.diff(inp.time.data).astype(float)) * 1e-9
        f_sampling = 1 / delta_t

    # convert ms to points
    n_per_seg = np.round(frame_length * f_sampling).astype(int)
    n_overlap = np.round(frame_overlap * f_sampling).astype(int)

    options = dict(fs=f_sampling, window=window, nperseg=n_per_seg,
                   noverlap=n_overlap, mode='complex')
    frequencies, time, spectrogram = signal.spectrogram(inp, **options)

    return frequencies, time, spectrogram
