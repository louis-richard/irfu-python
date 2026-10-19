#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dft_time_shift(inp, tau, f_s: float = None):
    r"""Shifts the time series `inp` by `tau` seconds using the discrete
    Fourier transform (DFT), as irfu-matlab mms.dft_timeshift: the output at
    the times t + `tau` is the input signal interpolated at t + `tau`.
    Particularly useful to align the samples of the different probes when
    calculating frequency-wavenumber spectra.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to shift, scalar (time,) or with components (time, ...).
    tau : float
        Shift in seconds: the output times are the input times plus `tau`.
    f_s : float, Optional
        Sampling frequency (Hz). Default is the mean sampling frequency of
        `inp`, 1 / ((t_N - t_1) / (N - 1)).

    Returns
    -------
    out : xarray.DataArray
        Time series of `inp` at the times shifted by `tau`, with the
        coordinates and attributes of `inp`.

    See Also
    --------
    pyrfu.mms.fk_power_spectrum_4sc : Calculates the frequency-wave number
        power spectrum.

    Notes
    -----
    The shift is a phase ramp in the frequency domain, so it can be a
    fraction of a sample. The DFT treats the signal as periodic: near the
    ends of a signal that isn't, the shifted samples mix with the other end,
    as in irfu-matlab, so keep margins around the interval of interest. The
    Nyquist component of an even number of samples is set to zero, as
    irfu-matlab. The time tags are whole nanoseconds, so that 1 / median(dt)
    is off by up to 3e-6 for 8192 Hz data; the mean time step is exact to
    about 1 ns / (N - 1). irfu-matlab rounds the sampling frequency to an
    integer number of Hz: give `f_s` (e.g. 8192) for its results to
    round-off.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2015-10-16T13:07:02.000", "2015-10-16T13:07:03.000"]
    >>> e_dsl = mms.get_data("e_dsl_edp_brst_l2", tint, 1)

    Electric field half a sample later (8192 Hz)

    >>> e_dsl_shifted = mms.dft_time_shift(e_dsl, 0.5 / 8192)

    """
    # Shift in samples (a fraction of a sample is allowed)
    if f_s is None:
        d_t = (inp.time.data[-1] - inp.time.data[0]) / np.timedelta64(1, "s")
        f_s = (len(inp.time) - 1) / d_t

    d_s = tau * f_s

    sig_fft = np.fft.fft(inp.data, axis=0)
    n_p = sig_fft.shape[0]

    # Disregard the Nyquist frequency of an even-sized DFT
    if not n_p % 2:
        sig_fft[n_p // 2, ...] = 0

    # Signed frequencies in cycles per sample. irfu-matlab writes the phase
    # ramp exp(-2i pi dS f)', whose conjugate transpose makes it
    # exp(2i pi dS f): the output at t + tau is the input at t + tau
    freq = ((np.arange(n_p) + n_p // 2) % n_p - n_p // 2) / n_p
    phase = np.exp(2j * np.pi * d_s * freq)
    phase = phase.reshape((-1,) + (1,) * (sig_fft.ndim - 1))
    out_data = np.fft.ifft(sig_fft * phase, axis=0)

    if not np.iscomplexobj(inp.data):
        out_data = np.real(out_data)

    # Time tags shifted by tau (ns)
    out_time = inp.time.data + np.timedelta64(int(np.round(tau * 1e9)), "ns")

    out = inp.copy(data=out_data)
    out = out.assign_coords(time=("time", out_time, dict(inp.time.attrs)))

    return out
