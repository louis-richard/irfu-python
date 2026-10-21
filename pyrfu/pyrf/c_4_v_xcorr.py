#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from typing import Optional, Sequence

# 3rd party imports
import numpy as np
from scipy import optimize, signal
from xarray.core.dataarray import DataArray

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.4"
__status__ = "Prototype"

# Elements of the arrays evaluated at once (memory bound for long windows)
_CHUNK = 2**21


def _seconds(time, t_ref):
    r"""Times in s from t_ref, from the exact steps in ns."""
    return (np.asarray(time).astype("datetime64[ns]") - t_ref).astype(np.int64) * 1e-9


class _BandLimited:
    r"""Band-limited (DFT) interpolant of a uniformly sampled signal (D-EH1):
    the shift keeps the variance of the noise at any fraction of a sample,
    unlike linear or cubic interpolation, which bias the delays towards
    half-samples.

    The samples in [t_min, t_max] and up to `margin` more on each side are
    detrended by the line through the end samples (continuous periodic
    extension), interpolated through their DFT, and the line is added back.

    """

    def __init__(self, time, data, t_min, t_max, name, margin=64):
        if time[0] > t_min + 1e-9 or time[-1] < t_max - 1e-9:
            raise ValueError(
                f"{name} does not cover tint extended by the search window "
                "(max_lag around tau)"
            )

        i_0 = max(0, int(np.searchsorted(time, t_min - 1e-9)) - margin)
        i_1 = min(len(time), int(np.searchsorted(time, t_max + 1e-9)) + margin)
        time = time[i_0:i_1]
        data = np.asarray(data, dtype=np.float64)[i_0:i_1]

        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} has NaN in tint extended by the search window")

        steps = np.diff(time)
        gaps = np.where(steps > 1.5 * np.median(steps))[0]

        if gaps.size:
            raise ValueError(
                f"{name} has a gap after {time[gaps[0]]:.6f} s from tint[0], in "
                "tint extended by the search window"
            )

        # Uniform sampling (the time tags are rounded to the ns)
        self.m = len(time)
        self.d_t, self.t_0 = np.polyfit(np.arange(self.m), time, 1)

        self.scalar = data.ndim == 1
        data = data.reshape(self.m, -1)
        # Line through the end samples: (value at the start, rise)
        self.line = (data[0], data[-1] - data[0])
        ramp = np.linspace(0.0, 1.0, self.m)[:, None]
        self.coef = np.fft.rfft(data - self.line[0] - self.line[1] * ramp, axis=0)
        self.freq = np.fft.rfftfreq(self.m)

    def exact(self, pos):
        r"""Values at any (fractional) sample positions: the trigonometric
        polynomial of the DFT (the Nyquist term as a cosine)."""
        weights = np.full(len(self.freq), 2.0)
        weights[0] = 1.0

        if self.m % 2 == 0:
            weights[-1] = 1.0

        coef = weights[:, None] * self.coef / self.m
        n_chunk = max(1, _CHUNK // len(self.freq))
        out = np.empty((len(pos), coef.shape[1]))

        for i in range(0, len(pos), n_chunk):
            phase = np.exp(2j * np.pi * np.outer(pos[i : i + n_chunk], self.freq))
            out[i : i + n_chunk] = np.real(phase @ coef)

        return out

    def __call__(self, grid, taus):
        r"""Values at the times grid = (t_start, d_t, n_t) shifted by each
        delay in taus: (n_tau, n_t) or (n_tau, n_t, n_comp)."""
        t_start, d_t, n_t = grid
        taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        ratio = d_t / self.d_t
        pos_0 = (t_start + taus - self.t_0) / self.d_t
        k = np.arange(n_t)

        if abs(ratio - 1) * (n_t - 1) < 1e-3:
            # Same cadence: one fractional shift of the whole segment per delay
            k_0 = np.floor(pos_0).astype(int)
            frac = pos_0 - k_0
            pos = pos_0[:, None] + k[None, :]
            out = np.empty((len(taus), n_t, self.coef.shape[1]))
            n_chunk = max(1, _CHUNK // (self.m * self.coef.shape[1]))

            for i in range(0, len(taus), n_chunk):
                phase = np.exp(2j * np.pi * np.outer(frac[i : i + n_chunk], self.freq))
                shifted = np.fft.irfft(
                    self.coef[None, ...] * phase[..., None], n=self.m, axis=1
                )
                idx = k_0[i : i + n_chunk, None] + k[None, :]
                out[i : i + n_chunk] = np.take_along_axis(
                    shifted, idx[..., None], axis=1
                )
        else:
            # Other cadence: the exact interpolant at every position
            pos = pos_0[:, None] + ratio * k[None, :]
            out = self.exact(pos.ravel()).reshape(len(taus), n_t, -1)

        out += self.line[0] + self.line[1] * (pos / (self.m - 1))[..., None]

        return out[..., 0] if self.scalar else out


def _msd_ts(taus, s_sc, p_pat, grid):
    r"""Mean square deviation I (Vogt et al. 2011, eq. 45) between the signal
    shifted by each tau and the pattern, and the residuals h (eq. 48)
    (irfu-matlab msdTS). The signal is shifted by band-limited interpolation
    (MATLAB: linear S.resample(P)), and I is taken over the samples of the
    pattern for every tau."""
    n_chunk = max(1, _CHUNK // (s_sc.m * p_pat.size // grid[2]))
    i_msd = np.empty(len(taus))

    for i in range(0, len(taus), n_chunk):
        h_res = s_sc(grid, taus[i : i + n_chunk]) - p_pat[None, ...]
        i_msd[i : i + n_chunk] = np.sum(h_res.reshape(h_res.shape[0], -1) ** 2, axis=1)

    return i_msd / grid[2], h_res


def _get_tau(s_sc, p_pat, grid, start_tau, max_lag):
    r"""Time difference of the signal to the pattern: the minimum of I
    (irfu-matlab fminsearch(@(x) msdTS(x,S,P,tint), startTau)). Here, a global
    search on the sample grid within start_tau +- max_lag, then a bounded
    refinement within +-1 sample of the grid minimum: no start guess needed,
    and no wrong local minimum. Returns (tau, flag): NaN and True when the
    grid minimum is at the edge of the window."""
    t_samp = grid[1]
    k_max = int(np.floor(max_lag / t_samp + 1e-9))
    taus = start_tau + t_samp * np.arange(-k_max, k_max + 1)
    i_grid = int(np.argmin(_msd_ts(taus, s_sc, p_pat, grid)[0]))

    if i_grid in (0, len(taus) - 1):
        return np.nan, True

    res = optimize.minimize_scalar(
        lambda tau: _msd_ts(np.array([tau]), s_sc, p_pat, grid)[0][0],
        bounds=(taus[i_grid] - t_samp, taus[i_grid] + t_samp),
        method="bounded",
        options={"xatol": 1e-6 * t_samp},
    )

    return float(res.x), False


def _corr(x, y):
    r"""Correlation coefficient of two signals (0 if one is constant)."""
    x, y = x.ravel() - np.mean(x), y.ravel() - np.mean(y)
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))

    return float(np.sum(x * y) / norm) if norm > 0 else 0.0


def _lagged_mean(x, y):
    r"""< x(t) . y(t + Delta) > over the overlapping samples, for all the
    shifts Delta = k t_samp, k = -(N - 1), ..., N - 1 (dot product over the
    components; MATLAB: the mean over the non-zero values)."""
    n_pts = x.shape[0]
    x, y = x.reshape(n_pts, -1), y.reshape(n_pts, -1)
    xy_sum = sum(
        signal.correlate(y[:, i], x[:, i], mode="full") for i in range(x.shape[1])
    )

    return xy_sum / (n_pts - np.abs(np.arange(-n_pts + 1, n_pts)))


def _corr_g(p_prime, t_w, t_samp):
    r"""G(Delta), eq. (49), for all the shifts at once (irfu-matlab corrG):
    (1 - |Delta| / T_w) < P'(t) P'(t + Delta) > / < P'^2 >."""
    n_pts = p_prime.shape[0]
    delta = t_samp * np.arange(-n_pts + 1, n_pts)
    p_prime2 = np.mean(np.sum(p_prime.reshape(n_pts, -1) ** 2, axis=1))

    return (1 - np.abs(delta) / t_w) * _lagged_mean(p_prime, p_prime) / p_prime2


def _corr_h(h_a, h_b):
    r"""H_ab(Delta), eq. (50), for all the shifts at once (irfu-matlab corrH):
    < h_a(t) h_b(t + Delta) > / (sqrt(< h_a^2 >) sqrt(< h_b^2 >))."""
    n_pts = h_a.shape[0]
    rms_a, rms_b = [
        np.sqrt(np.mean(np.sum(h.reshape(n_pts, -1) ** 2, axis=1))) for h in [h_a, h_b]
    ]

    if rms_a * rms_b == 0:
        return np.zeros(2 * n_pts - 1)

    return _lagged_mean(h_a, h_b) / (rms_a * rms_b)


def _get_disc_vel(r_xyz, tau, dtau2):
    r"""Velocity with errors with the timing method (irfu-matlab get_disc_vel;
    Vogt et al. 2011, sections 3 and 4)."""
    # Position vectors with origin in the tetrahedron center
    r_star = r_xyz - np.mean(r_xyz, axis=0)

    # Position tensor Rstar (1), generalized reciprocal vectors q (3) and the
    # reciprocal tensor K
    r_star_tensor = r_star.T @ r_star
    q = np.linalg.solve(r_star_tensor, r_star.T).T
    k_mat = q.T @ q

    # Slowness vector (17), which holds for arbitrary time offsets; normal
    # vector and discontinuity speed from (9)
    m = q.T @ tau
    n = m / np.linalg.norm(m)
    v = 1 / np.linalg.norm(m)

    if dtau2 is None:
        return v, n, m, None, None, None, None

    # Scalar time error, as MATLAB: dT = trace(sqrt(dT_tensor)) / 4
    d_t = np.trace(np.sqrt(np.abs(dtau2))) / 4

    # Absolute rms error in velocity (33)
    rms_v = v**2 * d_t * np.sqrt(n @ k_mat @ n)

    # Errors in angle (32), dTh = V dT sqrt(e K e) for e perpendicular to n:
    # min and max are the eigenvalues of K projected on the plane
    # perpendicular to n, and eTh the eigenvectors (MATLAB: a scan of 10^4
    # directions from cross([0 1 0], n))
    proj = np.eye(3) - np.outer(n, n)
    eig_val, eig_vec = np.linalg.eigh(proj @ k_mat @ proj)
    in_plane = np.sort(np.argsort(np.abs(eig_vec.T @ n))[:2])
    d_th = v * d_t * np.sqrt(np.abs(eig_val[in_plane]))
    e_th = eig_vec[:, in_plane].T

    # Error in m, from (21) with dk = 0
    dm2 = q.T @ dtau2 @ q

    return v, n, m, rms_v, d_th, e_th, dm2


def _positions(r_mms, t_center):
    r"""Positions of the four spacecraft at t_center (linear interpolation;
    MATLAB: R?.resample(T))."""
    r_xyz = np.zeros((4, 3))

    for i, r_sc in enumerate(r_mms):
        time = _seconds(r_sc.time.data, t_center)

        if time[0] > 0 or time[-1] < 0:
            raise ValueError(f"r_mms[{i}] does not cover the middle of tint")

        r_xyz[i] = [np.interp(0.0, time, r_sc.data[:, k]) for k in range(3)]

    return r_xyz


def c_4_v_xcorr(
    tint: Sequence[str],
    s_mms: Sequence[DataArray],
    r_mms: Sequence[DataArray],
    max_lag: Optional[float] = None,
    tau: Optional[Sequence[float]] = None,
    pattern: Optional[DataArray] = None,
    errors: bool = True,
) -> dict:
    r"""Four-spacecraft timing by cross-correlation, with error estimates.

    The delays between the signals of the four spacecraft are found by
    minimising the mean square difference to a common pattern, and the velocity
    of the structure and its errors follow from the delays and the positions
    of the spacecraft (Vogt et al. 2011 [1]_). It combines irfu-matlab
    ``c_4_v_xcorr`` (global search of the delays) and ``irf_4_v_xcorr_errors``
    (average pattern, continuous delays and error estimates).

    Parameters
    ----------
    tint : list of str
        Time interval of the structure, as tight around it as possible (see
        Notes). The signals must extend at least ``max_lag`` beyond it on both
        sides.
    s_mms : list of xarray.DataArray
        Signals of the four spacecraft, scalar or vector (the vector
        components are fitted together), uniformly sampled without gaps around
        tint. Their cadences may differ slightly (clocks) and their sample
        times may differ.
    r_mms : list of xarray.DataArray
        Positions of the four spacecraft in km, around tint.
    max_lag : float, Optional
        Largest delay searched, in s, around ``tau``. Set it, e.g., to the
        largest separation over the slowest expected speed: in a train of
        structures, it keeps each delay on the same structure. Default is the
        largest delay the data allow.
    tau : list of float, Optional
        Centre of the search window of each delay, in s, relative to MMS1 (to
        the pattern, if given), e.g., from a first run or from picked times.
        Default is zero.
    pattern : xarray.DataArray, Optional
        Pattern to fit the four signals to, covering tint (e.g., the pattern
        of a previous call). Default is the average of the four signals shifted
        to MMS1.
    errors : bool, Optional
        Compute the errors (Vogt et al. 2011, eqs. 45-53). Default is True.

    Returns
    -------
    dict
        * tau : numpy.ndarray
            Delays of the four spacecraft relative to MMS1 (tau[0] = 0) in s,
            positive when the spacecraft crosses after MMS1 (as the crossing
            times of ``c_4_v``). NaN where the minimum is at the edge of the
            search window (see flags).
        * dtau2 : numpy.ndarray
            Covariance of the delays <dtau_a dtau_b> (4, 4), in s^2 (eq. 52;
            MATLAB dtau2).
        * v : numpy.ndarray
            Velocity of the structure (3,) in km/s, in the frame of r_mms.
        * speed : float
            Speed in km/s (MATLAB V).
        * n : numpy.ndarray
            Unit normal (direction of propagation) (3,) (MATLAB n).
        * m : numpy.ndarray
            Slowness n / speed (3,) in s/km (MATLAB m).
        * dm2 : numpy.ndarray
            Covariance of the slowness <dm dm^T> (3, 3) in s^2/km^2 (MATLAB
            dm2).
        * d_speed : float
            Root-mean-square error of the speed in km/s (MATLAB dV).
        * d_theta : numpy.ndarray
            Smallest and largest errors of the direction of propagation (2,),
            in rad (semi-axes of the error ellipse; MATLAB dTh).
        * e_theta : numpy.ndarray
            Axes of the error ellipse (2, 3), perpendicular to n (MATLAB eTh,
            as rows).
        * corr : numpy.ndarray
            Correlation coefficient of each shifted signal with the pattern
            (4,): close to 1 for a good fit (e.g., 0.98 on a synthetic hole
            with 5 % noise), with a warning below 0.7.
        * flags : numpy.ndarray
            True where the minimum is at the edge of the search window (4,):
            the delay is NaN, increase max_lag or move tau.
        * pattern : xarray.DataArray
            Pattern the signals are fitted to, on the times of MMS1 in tint.

        With ``errors=False`` or a flagged delay, dtau2, dm2, d_speed, d_theta
        and e_theta are None; with a flagged delay, v, speed, n and m are NaN.

    Raises
    ------
    TypeError
        If s_mms or r_mms are not lists of four DataArrays.
    ValueError
        If a signal does not cover tint extended by the search window, has NaN
        or a gap there, if the pattern does not cover tint, or if the
        positions do not cover the middle of tint.

    Warns
    -----
    UserWarning
        If a delay is at the edge of the search window, or if a shifted signal
        correlates poorly with the pattern (below 0.7): a structure outside
        the search window gives a flat mean square difference, whose minimum
        is anywhere inside it.

    See Also
    --------
    pyrfu.pyrf.c_4_v : Velocity from crossing times.
    pyrfu.mms.mms4_displacement : Separations along and across B.

    Notes
    -----
    Steps:

    1. Each delay is searched on the sample grid of MMS1 within ``tau`` +-
       ``max_lag``, then refined continuously within +-1 sample. The signals
       are shifted by band-limited (DFT) interpolation, which keeps the
       variance of the noise at any fraction of a sample: linear or cubic
       interpolation lowers it between samples (by 50 % and 25 % at half a
       sample) and pulls the delays towards half-samples. The mean square
       difference is taken over the same samples of tint for every delay.
    2. Without a pattern, MMS2-4 are first fitted to MMS1; the pattern is the
       average of the four signals shifted to MMS1, and the four signals are
       fitted to it.
    3. The slowness follows from the reciprocal vectors of the tetrahedron at
       the middle of tint, and the errors from Vogt et al. (2011), eqs. 45-53
       and 32-33.

    **Choice of tint.** Take tint as tight around the structure as possible.
    The pattern is built from the noisy signals, and the noise-by-noise terms
    of the mean square difference grow with the length of tint: on a
    synthetic electron hole in tint +-10-25 samples, the spread of the delays
    over noise draws is at the least-squares limit and within 1.1 of dtau2,
    and the speed spread within 1.4 of d_speed. For a step-like boundary, or
    a tint much longer than the structure, the errors are underestimated (by
    1.4-2 on a tanh boundary in +-12-33 samples, more for longer tint).
    There, the noise above the band of the structure dominates: low-pass
    filtering the four signals to that band with the same zero-phase filter
    (which leaves the delays unchanged) brings the spread close to the
    least-squares limit (on the tanh boundary in +-33 samples at 0.25 f_s:
    from 1.4-1.7 to 1.0-1.2 times the limit, and the errors within 1.5 instead
    of 2). A cutoff inside the band of the structure increases the spread
    instead.

    Differences from irfu-matlab:

    * ``c_4_v_xcorr``: the global search is restricted to +-``max_lag`` with
      the same samples for every delay (MATLAB scans every overlap, normalised
      by its length, and finds the edge for a localised structure); scalar
      signals are accepted; the "minimum doubles" error is replaced by the
      Vogt errors.
    * ``irf_4_v_xcorr_errors``: no start guess needed (MATLAB's local search
      from -10 ms can find a wrong minimum); band-limited instead of linear
      interpolation; P' by central differences; the direction errors are the
      eigenvalues of K projected on the plane perpendicular to n (exact, for
      any n; MATLAB scans 10^4 directions from ``cross([0 1 0], n)``); vector
      signals (dot products in eqs. 48-50).

    The spacecraft are assumed not to move during tint (the velocity is in the
    spacecraft frame).

    References
    ----------
    .. [1] Vogt, J., Haaland, S., and Paschmann, G. (2011), Accuracy of
           multi-point boundary crossing time analysis, Ann. Geophys., 29,
           2239-2252, doi:10.5194/angeo-29-2239-2011.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Interplanetary shock (the irf_4_v_xcorr_errors example)

    >>> tint = ["2018-01-08T06:41:10.500", "2018-01-08T06:41:11.000"]
    >>> tint_long = ["2018-01-08T06:41:10.000", "2018-01-08T06:41:11.500"]
    >>> b_mms = [mms.get_data("b_gse_fgm_brst_l2", tint_long, i) for i in range(1, 5)]
    >>> tint_r = pyrf.extend_tint(tint_long, [-60, 60])
    >>> r_mms = [mms.get_data("r_gse_mec_srvy_l2", tint_r, i) for i in range(1, 5)]
    >>> res = pyrf.c_4_v_xcorr(tint, [b[:, 2] for b in b_mms], r_mms, max_lag=0.3)
    >>> res["v"], res["d_speed"]

    """
    # %% Input
    if not isinstance(s_mms, (list, tuple)) or len(s_mms) != 4:
        raise TypeError("s_mms must be a list of the 4 signals")

    if not isinstance(r_mms, (list, tuple)) or len(r_mms) != 4:
        raise TypeError("r_mms must be a list of the 4 positions")

    if not all(isinstance(inp, DataArray) for inp in [*s_mms, *r_mms]):
        raise TypeError("s_mms and r_mms must be lists of xarray.DataArray")

    start_tau = np.zeros(4) if tau is None else np.asarray(tau, dtype=np.float64)

    if start_tau.shape != (4,):
        raise ValueError("tau must have the 4 delays")

    # %% Prepare analysis
    # Times in s from tint[0]; the pattern is on the samples of MMS1 in tint,
    # as a uniform grid (t_start, t_samp, N) (MATLAB: P = P.tlim(tint))
    t_ref, t_end = iso86012datetime64(np.asarray(tint))
    time_1 = _seconds(s_mms[0].time.data, t_ref)
    in_tint = (time_1 >= 0) & (time_1 <= _seconds(t_end, t_ref))

    if np.sum(in_tint) < 5:
        raise ValueError("tint must contain at least 5 samples of MMS1")

    # Sample time (Tsamp), from the time tags rounded to the ns
    t_samp, t_start = np.polyfit(np.arange(np.sum(in_tint)), time_1[in_tint], 1)
    grid = (t_start, t_samp, int(np.sum(in_tint)))

    # Search windows: start_tau +- max_lag, by default the largest the data
    # allow
    if max_lag is None:
        margins = []

        for s_sc, tau_a in zip(s_mms, start_tau):
            time = _seconds(s_sc.time.data, t_ref)
            margins.append(time_1[in_tint][0] + tau_a - time[0])
            margins.append(time[-1] - time_1[in_tint][-1] - tau_a)

        max_lag = min(margins) - 2 * t_samp

    if max_lag < 2 * t_samp:
        raise ValueError(
            "the signals must extend beyond tint by at least 2 samples plus tau"
        )

    # S1, ..., S4 for shifts within the search windows
    s_1, s_2, s_3, s_4 = [
        _BandLimited(
            _seconds(s_sc.time.data, t_ref),
            s_sc.data,
            t_start + tau_a - max_lag - t_samp,
            t_start + (grid[2] - 1) * t_samp + tau_a + max_lag + t_samp,
            f"s_mms[{i}]",
        )
        for i, (s_sc, tau_a) in enumerate(zip(s_mms, start_tau))
    ]

    # %% If no pattern in input -> fit to S1 and then construct average pattern
    if pattern is None:
        # Time differences with the first spacecraft as reference
        p_pat = s_1(grid, 0.0)[0]
        tau_12 = _get_tau(s_2, p_pat, grid, start_tau[1], max_lag)[0]
        tau_13 = _get_tau(s_3, p_pat, grid, start_tau[2], max_lag)[0]
        tau_14 = _get_tau(s_4, p_pat, grid, start_tau[3], max_lag)[0]

        # Average signal to use as pattern (the ones found)
        p_pat = np.mean(
            [
                s_sc(grid, tau_1a)[0]
                for s_sc, tau_1a in zip(
                    [s_1, s_2, s_3, s_4], [0.0, tau_12, tau_13, tau_14]
                )
                if np.isfinite(tau_1a)
            ],
            axis=0,
        )
    else:
        p_in = _BandLimited(
            _seconds(pattern.time.data, t_ref),
            pattern.data,
            t_start,
            t_start + (grid[2] - 1) * t_samp,
            "pattern",
        )
        p_pat = p_in(grid, 0.0)[0]

    # %% Get time differences using the pattern
    tau_1, tau_2, tau_3, tau_4 = [
        _get_tau(s_sc, p_pat, grid, tau_a, max_lag)
        for s_sc, tau_a in zip([s_1, s_2, s_3, s_4], start_tau)
    ]
    flags = np.array([tau_1[1], tau_2[1], tau_3[1], tau_4[1]])
    tau_p = np.array([tau_1[0], tau_2[0], tau_3[0], tau_4[0]])

    for i in np.where(flags)[0]:
        warnings.warn(
            f"the delay of MMS{i + 1} is at the edge of the search window: "
            "increase max_lag or move tau",
            UserWarning,
        )

    # Residuals h (48), minimum mean square deviations Imin, and correlation
    # with the pattern
    h_res, i_min = [None] * 4, np.full(4, np.nan)

    for i, (s_sc, tau_a) in enumerate(zip([s_1, s_2, s_3, s_4], tau_p)):
        if np.isfinite(tau_a):
            i_min_a, h_a = _msd_ts(np.array([tau_a]), s_sc, p_pat, grid)
            i_min[i], h_res[i] = i_min_a[0], h_a[0]

    corr = np.array(
        [_corr(h_a + p_pat, p_pat) if h_a is not None else np.nan for h_a in h_res]
    )

    for i in np.where(corr < 0.7)[0]:
        warnings.warn(
            f"MMS{i + 1} correlates poorly with the pattern ({corr[i]:.2f}): its "
            "structure may be outside the search window, or the signals differ",
            UserWarning,
        )

    # Delays relative to MMS1 (MATLAB: tau = -[tau1, ...], the sign being
    # "empirical"; here the sign of the crossing times)
    tau = tau_p - tau_p[0]

    # %% Get time difference errors
    # The goal is to calculate eq (52)
    if errors and not np.any(flags):
        # The time interval used for averaging, and the number of points
        t_w = (grid[2] - 1) * t_samp / 2
        n_pts = grid[2]

        # Derivative of P (MATLAB: forward differences) and G (49)
        p_prime = np.gradient(p_pat, t_samp, axis=0)
        g_corr = _corr_g(p_prime, t_w, t_samp)
        p_prime2 = np.mean(np.sum(p_prime.reshape(n_pts, -1) ** 2, axis=1))

        # Shifts Delta within +-T_w (MATLAB: Delta = P.time - mean(P.time))
        delta = t_samp * np.arange(-n_pts + 1, n_pts)
        in_tw = np.abs(delta) <= t_w + 1e-9 * t_samp

        # Error through eq (52), with H (50)
        dtau2 = np.zeros((4, 4))

        for i, j in np.ndindex(4, 4):
            h_ij = _corr_h(h_res[i], h_res[j])
            dtau2[i, j] = (
                np.sqrt(i_min[i] * i_min[j])
                / (n_pts * p_prime2)
                * np.sum((g_corr * h_ij)[in_tw])
            )
    else:
        dtau2 = None

    # %% Get propagation velocity from time lags
    if np.any(flags):
        v_n, n, m = np.nan, np.full(3, np.nan), np.full(3, np.nan)
        rms_v, d_th, e_th, dm2 = None, None, None, None
    else:
        # Time in center of interval
        mean_t = t_ref + (t_end - t_ref) // 2
        r_xyz = _positions(r_mms, mean_t)
        v_n, n, m, rms_v, d_th, e_th, dm2 = _get_disc_vel(r_xyz, tau, dtau2)

    # %% Set output
    out = {
        "tau": tau,
        "dtau2": dtau2,
        "v": v_n * n,
        "speed": v_n,
        "n": n,
        "m": m,
        "dm2": dm2,
        "d_speed": rms_v,
        "d_theta": d_th,
        "e_theta": e_th,
        "corr": corr,
        "flags": flags,
        "pattern": s_mms[0][in_tint].copy(data=p_pat),
    }

    return out
