#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import signal

# Local imports
from .calc_fs import calc_fs


def wave_fft(
    inp,
    window,
    frame_overlap: float = 10.0,
    frame_length: float = 20.0,
    f_sampling: float | None = None,
    scaled: bool = True,
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
    scaled : bool, Optional
        True for the convention of ``scipy.signal.spectrogram``: the transform
        scaled by sqrt(1 / (fs sum(w^2))), a periodic window when given by
        name, all the frequencies up to Nyquist and the frames timed at their
        centre. False for that of irfu-matlab ``irf_wavefft``: the raw FFT of
        the windowed frames, a symmetric window when given by name (e.g.
        ``hamming(n)``), the first round(n / 2) frequencies (without the
        Nyquist bin for an even frame length) and the frames timed
        round(n / 2) samples after their start. Default is True.

    Returns
    -------
    frequencies : ndarray
        Frequencies (y-axis), in Hz.
    time : ndarray
        Centre of each frame (x-axis), in seconds from the start of inp.
    spectrogram : ndarray
        Complex short-time Fourier transform, of shape (frequencies, time).

    Notes
    -----
    Each frame is transformed as it is, without removing its mean, as
    irfu-matlab ``irf_wavefft`` (``scipy.signal.spectrogram`` removed it by
    default, 63 % off in the lowest bins of a signal with an offset). With
    ``scaled=False``, the result is that of ``irf_wavefft`` (see `scaled`).
    In both conventions, the frame lengths are in seconds (``irf_wavefft``:
    milliseconds) and the last frame is kept when it ends on the last sample
    (``irf_wavefft`` drops it).

    """

    if f_sampling is None:
        # From the times in ns (they were taken as ns in any unit)
        f_sampling = calc_fs(inp)

    # Frame length and overlap in samples
    n_per_seg = np.round(frame_length * f_sampling).astype(np.int64)
    n_overlap = np.round(frame_overlap * f_sampling).astype(np.int64)

    if not scaled and isinstance(window, (str, tuple)):
        # irf_wavefft: the symmetric window, e.g. hamming(n)
        window = signal.get_window(window, n_per_seg, fftbins=False)

    options = {
        "fs": f_sampling,
        "window": window,
        "nperseg": n_per_seg,
        "noverlap": n_overlap,
        "mode": "complex",
        # Frames transformed as they are, as irf_wavefft (scipy removes the
        # mean of each frame by default)
        "detrend": False,
    }
    frequencies, time, spectrogram = signal.spectrogram(inp, **options)

    if not scaled:
        # irf_wavefft: the raw FFT, rows 1:round(n / 2), and the frames timed
        # round(n / 2) samples after their start
        n_rows = (n_per_seg + 1) // 2
        spectrogram = spectrogram[:n_rows] * np.sqrt(
            f_sampling * np.sum(np.square(window, dtype=np.float64))
        )
        frequencies = frequencies[:n_rows]
        time = (n_rows + np.arange(len(time)) * (n_per_seg - n_overlap)) / f_sampling

    return frequencies, time, spectrogram
