#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional, Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.extend_tint import extend_tint
from ..pyrf.resample import resample
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.wavelet import wavelet
from .probe_fields import probe_fields

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.6"
__status__ = "Prototype"

# Separation along the boom of the two single-probe fields of a pair (m):
# half of the tip-to-tip lengths 120 m and 29.2 m (each field is between its
# probe and the centre); the two probe potentials are twice as far apart
_SEPARATIONS = {1: 60.0, 3: 60.0, 5: 14.6}

# Angle of probes 1, 2, 3 and 4 from the spin phase (rad), irfu-matlab
_PROBE_PHASES = {1: np.pi / 6, 2: 7 * np.pi / 6, 3: 2 * np.pi / 3, 4: 5 * np.pi / 3}


def _unwrap_spin_phase(z_phase, time):
    r"""Spin phase (deg) unwrapped and interpolated at the times, as
    irfu-matlab mms.fk_powerspectrum: 360 deg added after each decrease
    (between increasing times), repeated times dropped."""
    t_z, data = z_phase.time.data, z_phase.data.astype(np.float64)
    step = np.diff(t_z) > np.timedelta64(0, "ns")
    data = data + 360.0 * np.r_[0, np.cumsum(step & (np.diff(data) < 0))]
    keep = np.r_[True, step]

    return resample(ts_scalar(t_z[keep], data[keep]), ts_scalar(time, time)).data


def _probe_directions(probe, z_phase, n_t):
    r"""Directions of the two probes of the pair (unit vectors, (n_t, 3)) in
    the frame of the magnetic field (DMPA): in the spin plane from the spin
    phase (deg) for probes 1-4, along the spin axis for probes 5 and 6."""
    if probe == 5:
        return np.tile([0.0, 0.0, 1.0], (n_t, 1)), np.tile([0.0, 0.0, -1.0], (n_t, 1))

    directions = []

    for p in [probe, probe + 1]:
        phase = np.deg2rad(z_phase) + _PROBE_PHASES[p]
        directions.append(np.column_stack([np.cos(phase), np.sin(phase), 0 * phase]))

    return directions


def fk_power_spectrum(
    v6: DataArray,
    b_xyz: DataArray,
    tint: Sequence[str],
    probe: int,
    z_phase: Optional[DataArray] = None,
    cav: int = 128,
    field: bool = True,
    n_k: int = 101,
    n_f: int = 200,
    df: Optional[float] = None,
) -> DataArray:
    r"""Frequency-wave number power spectrum from the probes of one pair of
    MMS EDP (single-spacecraft interferometry), as irfu-matlab
    ``mms.fk_powerspectrum`` (Graham et al. 2016 [1]_).

    The fields at the two probes of the pair, aligned with the magnetic field
    and 60 m apart (14.6 m for probes 5 and 6), are wavelet transformed; the
    phase of their cross-spectrum, averaged over ``cav`` samples, gives the
    wave number along B at each frequency, and the power is binned in
    frequency and wave number.

    Parameters
    ----------
    v6 : xarray.DataArray
        Time series of the probe-to-spacecraft potentials of the six probes
        (burst, ``v6_edp_brst_l2``), without the timing correction (applied
        here, by a DFT shift, see ``mms.probe_fields``), covering tint and 1 s
        on each side.
    b_xyz : xarray.DataArray
        Time series of the magnetic field in DMPA (or DSL) coordinates.
    tint : list of str
        Time interval of the spectrum, when B is closely aligned with the pair
        of probes (see ``mms.probe_align_times``).
    probe : {1, 3, 5}
        Pair of probes: 1 (probes 1 and 2), 3 (3 and 4) or 5 (5 and 6).
    z_phase : xarray.DataArray, Optional
        Time series of the spin phase in degrees (``z_phase`` of
        ``mms.load_ancillary("defatt", ...)``), for probes 1 to 4.
    cav : int, Optional
        Number of samples over which the cross-spectrum is averaged to
        estimate the phase. Default is 128.
    field : bool, Optional
        Use the single-probe fields (the probe potentials against the
        spacecraft potential estimated by the other probes, 60 m or 14.6 m
        apart), as default, or the two probe potentials (False, 120 m or
        29.2 m apart, so that the wave numbers span half the range).
    n_k : int, Optional
        Number of wave numbers. Default is 101.
    n_f : int, Optional
        Number of frequencies (logarithmically spaced). Default is 200.
    df : float, Optional
        Linearly spaced frequencies with this step (Hz) instead (``n_f`` is
        not used).

    Returns
    -------
    xarray.DataArray
        Power (f, k) normalised to its maximum, NaN in the bins without
        power and at least 1e-3 elsewhere, with the frequencies f in Hz and
        the wave numbers k = 2 pi / lambda in 1/m (left edges of the bins, as
        irfu-matlab), positive along B.

    Raises
    ------
    ValueError
        If probe is not 1, 3 or 5, if z_phase is missing for probes 1 to 4,
        or if tint holds fewer than cav samples.

    See Also
    --------
    pyrfu.mms.probe_fields, pyrfu.mms.probe_align_times,
    pyrfu.mms.estimate_phase_speed, pyrfu.mms.fk_power_spectrum_4sc

    Notes
    -----
    The single-probe fields are accurate for wave vectors within about 30 deg
    of the pair of probes (10 deg for probes 5 and 6, and only in dense
    plasmas; Steinvall et al. 2022, see ``mms.probe_fields``). The wave
    number is the phase over the separation along B, 60 m times the cosine of
    the angle between B and the pair (14.6 m for probes 5 and 6), so that
    the wave numbers span +-pi / (the smallest separation in tint).

    With ``field=False``, irfu-matlab takes the second potential with a minus
    sign (adding pi to the phase) and the separation of the fields (half that
    of the potentials): its wave numbers are wrong. Here the two potentials
    are used as they are, with their separation.

    References
    ----------
    .. [1] Graham, D. B., Khotyaintsev, Yu. V., Vaivads, A., and André, M.
           (2016), Electrostatic solitary waves and electrostatic waves at
           the magnetopause, J. Geophys. Res. Space Physics, 121,
           3069-3092, doi:10.1002/2015JA021527.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    >>> tint = ["2017-05-05T20:09:50.820", "2017-05-05T20:09:50.839"]
    >>> tint_long = pyrf.extend_tint(tint, [-2, 2])
    >>> v6 = mms.get_data("v6_edp_brst_l2", tint_long, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint_long, 1)
    >>> power = mms.fk_power_spectrum(v6, b_dmpa, tint, 5)
    >>> v_ph = mms.estimate_phase_speed(power.data, power.f.data, power.k.data)

    """

    if probe not in (1, 3, 5):
        raise ValueError(f"probe must be 1, 3 or 5, not {probe!r}")

    if probe != 5 and z_phase is None:
        raise ValueError("z_phase is needed for probes 1 to 4")

    # Fields over tint and 1 s on each side (wavelet edges), the timing
    # correction applied to the whole input, as irfu-matlab
    tint_long = extend_tint(tint, [-1.0, 1.0])
    fields = time_clip(probe_fields(v6, timing="dft"), tint_long)
    time = fields.time.data

    # Single-probe fields, or the probe potentials, and their separation
    if field:
        signals, separation = fields.e_probe.data, _SEPARATIONS[probe]
    else:
        signals, separation = fields.v.data, 2 * _SEPARATIONS[probe]

    # Magnetic field and directions of the two probes at the times of the
    # fields; cosines of their angles to B
    b_data = resample(time_clip(b_xyz, tint_long), fields.v[:, 0]).data[:, :3]
    b_mag = np.linalg.norm(b_data, axis=1)
    phase = (
        None if probe == 5 else _unwrap_spin_phase(time_clip(z_phase, tint_long), time)
    )
    r_hat = _probe_directions(probe, phase, len(time))
    cos_pb = [np.sum(r * b_data, axis=1) / b_mag for r in r_hat]

    # Samples in tint (an even number, as irfu-matlab)
    t_0, t_1 = np.datetime64(tint[0], "ns"), np.datetime64(tint[1], "ns")
    idx = np.where((time >= t_0) & (time <= t_1))[0]
    idx = idx[: len(idx) - len(idx) % 2]
    n_av = len(idx) // cav

    if n_av < 1:
        raise ValueError(f"tint must hold at least cav = {cav} samples")

    # The two signals, the one on the -B side first (k > 0 along B): probe + 1
    # if the probe points along B
    e_1, e_2 = [ts_scalar(time, signals[:, p]) for p in [probe, probe - 1]]

    if np.nanmean(cos_pb[0][idx]) <= 0:
        e_1, e_2 = e_2, e_1

    # Frequencies of irf_wavelet: logarithmic from f_nyq / 100 to
    # f_nyq / 10^0.01, or linear from df to f_nyq
    options = {"return_power": False, "cut_edge": False}

    if df is None:
        f_nyq = calc_fs(e_1) / 2
        options.update({"n_freqs": n_f, "f": [f_nyq / 100, f_nyq / 10**0.01]})
    else:
        options["linear"] = df

    w_1, w_2 = [wavelet(e_i, **options) for e_i in [e_1, e_2]]
    freq = w_1.frequency.data
    w_1, w_2 = w_1.data[idx], w_2.data[idx]
    power = 0.5 * (np.abs(w_1) ** 2 + np.abs(w_2) ** 2)

    # Cross-spectrum and power over windows of cav samples
    windows = np.arange(n_av * cav).reshape(n_av, cav)
    cross = np.nanmean((w_1 * np.conj(w_2))[windows], axis=1)
    power_av = np.nansum(power[windows], axis=1)

    # Separation along B at the middle times of the windows: |cos| resampled
    # as irfu-matlab, the average over each window (the value at the middle
    # time for a single window)
    t_mid = time[idx][windows[:, cav // 2 - 1]]
    cos_mid = resample(ts_scalar(time, np.abs(cos_pb[0])), ts_scalar(t_mid, t_mid))
    r_cos = separation * cos_mid.data

    # Wave number of each window and frequency, and the power binned in wave
    # number (left edges of the bins, the last one closed on its right)
    k_val = np.arctan2(cross.imag, cross.real) / r_cos[:, None]
    k_min, k_max = -np.pi / np.min(r_cos), np.pi / np.min(r_cos)
    d_k = (k_max - k_min) / n_k
    k_vec = k_min + np.arange(n_k) * d_k
    finite = np.isfinite(k_val)
    i_k = np.floor((k_val[finite] - k_min) / d_k).astype(np.int64)
    i_f = np.broadcast_to(np.arange(len(freq)), k_val.shape)[finite]
    f_k_power = np.zeros((n_k, len(freq)))
    np.add.at(f_k_power, (np.clip(i_k, 0, n_k - 1), i_f), power_av[finite])

    # Normalised to the maximum, NaN without power, floor at 1e-3
    f_k_power[f_k_power == 0] = np.nan
    f_k_power /= np.nanmax(f_k_power)
    f_k_power[f_k_power < 1e-3] = 1e-3

    out = xr.DataArray(
        f_k_power.T,
        coords=[freq, k_vec],
        dims=["f", "k"],
        attrs={"probe": probe, "cav": cav, "field": int(field), "UNITS": ""},
    )
    out.f.attrs["UNITS"] = "Hz"
    out.k.attrs["UNITS"] = "1/m"

    return out
