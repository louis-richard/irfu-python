#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import signal

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def wave_fft(inp, window, frame_overlap: float = 10.,
             frame_length: float = 20., f_sampling: float = None):
    r"""Short-Time Fourier Transform.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the one dimension data.
    window : str
        Window function such as rectwin, hamming (default).
    frame_overlap : float, Optional
        Length of each frame overlaps in second.
    frame_length : float, Optional
        Length of each frame in second.
    f_sampling : float, Optional
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
