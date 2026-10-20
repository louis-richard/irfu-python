#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import signal

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def wave_fft(
    inp,
    window,
    frame_overlap: float = 10.0,
    frame_length: float = 20.0,
    f_sampling: float = None,
):
    r"""Short-Time Fourier Transform.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the one dimension data.
    window : str or tuple
        Window function, as for ``scipy.signal.spectrogram`` (e.g.,
        "hamming", "boxcar"). Required.
    frame_overlap : float, Optional
        Overlap of consecutive frames, in seconds. Default is 10.
    frame_length : float, Optional
        Length of each frame, in seconds. Default is 20.
    f_sampling : float, Optional
        Sampling frequency, in Hz. Default is that of inp.

    Returns
    -------
    frequencies : ndarray
        Frequencies (y-axis), in Hz.
    time : ndarray
        Centre of each frame (x-axis), in seconds from the start of inp.
    spectrogram : ndarray
        Complex short-time Fourier transform, of shape (frequencies, time).

    """

    if f_sampling is None:
        delta_t = np.median(np.diff(inp.time.data).astype(np.float64)) * 1e-9
        f_sampling = 1 / delta_t

    # convert ms to points
    n_per_seg = np.round(frame_length * f_sampling).astype(np.int64)
    n_overlap = np.round(frame_overlap * f_sampling).astype(np.int64)

    options = {
        "fs": f_sampling,
        "window": window,
        "nperseg": n_per_seg,
        "noverlap": n_overlap,
        "mode": "complex",
    }
    frequencies, time, spectrogram = signal.spectrogram(inp, **options)

    return frequencies, time, spectrogram
