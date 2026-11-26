#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from scipy import interpolate

# Local imports
from .filt import filt

# Phase bins per degree of the spin epoch (0.5 deg)
_STEPS_PER_DEG = 2

# Implementation notes (numba kernel _running_nanmedian): each series keeps
# the values of its window without NaN in a sorted buffer, the value leaving
# the window removed before the one entering is inserted (the buffer holds
# at most n_pts values), by binary search; the series are independent (one
# per phase bin and component), in parallel.


@numba.njit(cache=True, parallel=True)
def _running_nanmedian(arr, n_pts):
    r"""Moving median of each column of `arr` (n, m) over `n_pts` rows, NaN
    left out, n_pts // 2 rows before and (n_pts - 1) // 2 after, fewer at
    the ends; NaN without data."""

    n_rows, n_cols = arr.shape
    before, after = n_pts // 2, (n_pts - 1) // 2
    out = np.empty((n_rows, n_cols))

    for j in numba.prange(n_cols):
        buf = np.empty(n_pts)
        n_buf = 0

        # The rows after the first, up to its window
        for k in range(min(after, n_rows)):
            if not np.isnan(arr[k, j]):
                i = np.searchsorted(buf[:n_buf], arr[k, j])
                buf[i + 1 : n_buf + 1] = buf[i:n_buf].copy()
                buf[i] = arr[k, j]
                n_buf += 1

        for k in range(n_rows):
            # The row leaving the window, then the row entering it
            k_out, k_in = k - before - 1, k + after

            if k_out >= 0 and not np.isnan(arr[k_out, j]):
                i = np.searchsorted(buf[:n_buf], arr[k_out, j])
                buf[i : n_buf - 1] = buf[i + 1 : n_buf].copy()
                n_buf -= 1

            if k_in < n_rows and not np.isnan(arr[k_in, j]):
                i = np.searchsorted(buf[:n_buf], arr[k_in, j])
                buf[i + 1 : n_buf + 1] = buf[i:n_buf].copy()
                buf[i] = arr[k_in, j]
                n_buf += 1

            if n_buf == 0:
                out[k, j] = np.nan
            elif n_buf % 2:
                out[k, j] = buf[n_buf // 2]
            else:
                out[k, j] = 0.5 * (buf[n_buf // 2 - 1] + buf[n_buf // 2])

    return out


def _moving_nanmedian(arr, n_pts: int):
    r"""Moving median along the first axis over `n_pts` points, NaN left
    out, n_pts // 2 points before and (n_pts - 1) // 2 after, the windows
    shrinking at the ends."""

    flat = np.ascontiguousarray(arr, dtype=np.float64).reshape(len(arr), -1)

    return _running_nanmedian(flat, n_pts).reshape(arr.shape)


def spin_epoch(inp, phase, f_cut: float | None = None, n_spins: int = 31):
    r"""Spin epoch model of a signal: its part synchronous with the spin,
    the moving median over `n_spins` spins at each half degree of spin
    phase, e.g. the disturbance of the electric field by the shadow of the
    axial booms.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the signal, scalar or vector.
    phase : xarray.DataArray
        Time series of the spin phase (degrees), at its own times, sampled
        at least twice per spin (e.g. ``z_phase`` of the MMS defatt files).
    f_cut : float, Optional
        Cut-off frequency (Hz) of a high-pass filter applied to `inp` first
        (`pyrfu.pyrf.filt`, order 5). Default is None (no filter).
    n_spins : int, Optional
        Number of spins of the moving median. Default is 31.

    Returns
    -------
    out : xarray.DataArray
        Spin epoch model at the times of `inp`, with its coordinates and
        attributes, NaN beyond the times of `phase`.

    Raises
    ------
    TypeError
        If `inp` or `phase` is not a xarray.DataArray.
    ValueError
        If `inp` is not a scalar or vector time series, `phase` not a scalar
        time series with finite values, or `n_spins` not a positive integer.

    Notes
    -----
    The spin phase is unwrapped and a turn added at each step back (data
    gaps), so that it increases. The signal is interpolated onto a grid of
    whole spins in steps of 0.5 degree of phase (NaN beyond the phases of
    `phase` and the times of `inp`), and the model at each phase bin of
    each spin is the median of that bin over the `n_spins` spins around it
    (n_spins // 2 before and (n_spins - 1) // 2 after, fewer at the ends),
    the NaN left out. It is interpolated back to the times of `inp`,
    linearly extrapolated at the ends. Over a gap of more than half a spin,
    the unwrapped phase may miss whole turns.

    See Also
    --------
    pyrfu.pyrf.filt, pyrfu.pyrf.medfilt

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    >>> e_dsl = mms.get_data("e_dsl_edp_fast_l2", tint, 1)
    >>> defatt = mms.load_ancillary("defatt", tint, 1)

    Spin epoch model of the electric field, and the field without it

    >>> e_model = pyrf.spin_epoch(e_dsl, defatt.z_phase)
    >>> e_clean = e_dsl - e_model

    """

    if not isinstance(inp, xr.DataArray) or not isinstance(phase, xr.DataArray):
        raise TypeError("inp and phase must be xarray.DataArray")

    if inp.ndim not in (1, 2):
        raise ValueError("inp must be a scalar or vector time series")

    if phase.ndim != 1 or not np.all(np.isfinite(phase.data)):
        raise ValueError("phase must be a scalar time series with finite values")

    is_int = isinstance(n_spins, (int, np.integer)) and not isinstance(n_spins, bool)

    if not is_int or n_spins < 1:
        raise ValueError(f"n_spins must be a positive integer, not {n_spins!r}")

    if f_cut is not None:
        inp = filt(inp, f_cut, 0.0, 5)

    # Unwrapped phase (deg), a turn added at each step back (data gaps) so
    # that it increases
    pha = np.rad2deg(np.unwrap(np.deg2rad(phase.data.astype(np.float64))))
    pha += 360.0 * np.concatenate([[0], np.cumsum(np.diff(pha) < 0)])

    # Grid of whole spins in steps of 0.5 deg, and its times (s from the
    # first time of inp), NaN beyond the phases measured
    t_0 = inp.time.data[0]
    time = (inp.time.data - t_0) / np.timedelta64(1, "s")
    t_pha = (phase.time.data - t_0) / np.timedelta64(1, "s")
    start = pha[0] - np.fmod(pha[0], 360.0)
    stop = pha[-1] - np.fmod(pha[-1], 360.0) + 360.0
    grid = start + np.arange(round((stop - start) * _STEPS_PER_DEG)) / _STEPS_PER_DEG
    t_grid = np.interp(grid, pha, t_pha, left=np.nan, right=np.nan)

    # Signal on the grid, (spin, phase bin, component), and its median
    data = np.asarray(inp.data, dtype=np.float64).reshape(len(time), -1)
    f_grid = np.stack(
        [np.interp(t_grid, time, comp, left=np.nan, right=np.nan) for comp in data.T],
        axis=1,
    )
    f_grid = f_grid.reshape(-1, 360 * _STEPS_PER_DEG, data.shape[1])
    model = _moving_nanmedian(f_grid, n_spins).reshape(-1, data.shape[1])

    # Back to the times of inp, linearly extrapolated at the ends of the
    # grid, NaN beyond the times of phase
    ok = np.isfinite(t_grid)
    out = interpolate.interp1d(
        t_grid[ok], model[ok], axis=0, fill_value="extrapolate", assume_sorted=True
    )(time)
    out[(time < t_pha[0]) | (time > t_pha[-1])] = np.nan

    return inp.copy(data=out.reshape(inp.shape))
