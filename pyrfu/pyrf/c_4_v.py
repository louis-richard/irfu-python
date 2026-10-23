#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants, interpolate

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _to_seconds(time):
    r"""Convert datetime64 (or float epoch in seconds) to float seconds."""
    time = np.asarray(time)

    if np.issubdtype(time.dtype, np.datetime64):
        return time.astype("datetime64[ns]").astype(np.int64) * 1e-9

    return time.astype(np.float64)


def _get_vol_ten(r_xyz, time):
    r"""Separations of MMS2-4 from MMS1 (rows), at time ``time`` (s)."""
    r_ref = []

    for r_sc in r_xyz:
        r_time = _to_seconds(r_sc.time.data)
        r_ref.append(
            [np.interp(time, r_time, r_sc.data[:, j]) for j in range(3)],
        )

    r_ref = np.array(r_ref)

    # Volumetric tensor with SC1 as center: dR = [R2 - R1; R3 - R1; R4 - R1]
    dr_mat = r_ref[1:] - r_ref[0]

    return dr_mat


def _positions_at(r_xyz, t_ref, offsets):
    r"""Position of each spacecraft at its own time, t_ref (s) plus its
    offset (s, precise below the resolution of t_ref), by a cubic spline
    extrapolated outside the data, as irfu-matlab c_4_v (interp1 'spline',
    'extrap')."""
    r_out = []

    for r_sc, offset in zip(r_xyz, offsets):
        r_time = _to_seconds(r_sc.time.data) - t_ref
        spline = interpolate.CubicSpline(r_time, r_sc.data[:, :3])
        r_out.append(spline(offset))

    return np.array(r_out)


def _shifts_moving(r_xyz, time_center, velocity, delta_t):
    r"""Time shifts of MMS2-4 to MMS1 (crossing at time_center) for a plane
    moving at velocity in the frame of the positions: the fixed point of
    n.(r_a(t_c + dt_a) - r_1(t_c)) / |v|, from the shifts at fixed
    positions."""
    speed = np.linalg.norm(velocity)
    n_hat = velocity / speed
    r_1 = _positions_at(r_xyz[:1], time_center, [0.0])[0]

    for _ in range(100):
        r_a = _positions_at(r_xyz[1:], time_center, delta_t)
        delta_t_new = (r_a - r_1) @ n_hat / speed

        if np.max(np.abs(delta_t_new - delta_t)) < 1e-12:
            return delta_t_new

        delta_t = delta_t_new

    raise ValueError(
        "no crossing times: the spacecraft move along the normal as fast as "
        "the discontinuity"
    )


def c_4_v(r_xyz, time, frame: str = "spacecraft"):
    r"""Calculates velocity or time shift of discontinuity as in [6]_.

    Parameters
    ----------
    r_xyz : list
        Time series of the positions of the 4 spacecraft in km.
    time : list
        Either the crossing times of the 4 spacecraft (datetime64 or epoch in
        seconds), or the reference time followed by the velocity of the
        discontinuity in km/s, [t, v_x, v_y, v_z].
    frame : {"spacecraft", "positions"}, Optional
        Frame of the velocity (see Notes):

        * "spacecraft": relative to the spacecraft, with the four positions
          at the first (reference) time, as if the spacecraft did not move
          between the crossings. Default.
        * "positions": in the frame of the positions (e.g., GSE), with each
          position at its own crossing time (cubic spline), as irfu-matlab
          ``c_4_v``.

    Returns
    -------
    out : numpy.ndarray
        Velocity of the discontinuity in km/s (from crossing times), or time
        shifts in s of the 4 spacecraft with respect to MMS1 (from a velocity).

    Raises
    ------
    ValueError
        If frame is not "spacecraft" or "positions", or if, from a velocity
        with frame="positions", the spacecraft move along the normal as fast
        as the discontinuity (no crossing times).

    Notes
    -----
    The spacecraft move between the crossings. With the positions at their
    crossing times ("positions"), the velocity is that of the discontinuity
    in the frame of the positions. With the positions at one time
    ("spacecraft"), it is the velocity relative to the spacecraft (exact for
    spacecraft moving together): the normal is the same, and the speed
    differs by the velocity of the spacecraft along the normal, e.g., by
    1.5 km/s for a formation moving at 2.7 km/s, which is a few per cent of
    the speed of the magnetopause and much less of a shock. From a velocity,
    the time shifts with frame="positions" are found by iterating the
    positions at the crossing times.

    References
    ----------
    .. [6]	Vogt, J., Haaland, S., and Paschmann, G. (2011) Accuracy
            of multi-point boundary crossing time analysis, Ann.
            Geophys., 29, 2239-2252, doi :
            https://doi.org/10.5194/angeo-29-2239-2011

    """

    if frame not in ("spacecraft", "positions"):
        raise ValueError(f'frame must be "spacecraft" or "positions", not {frame!r}')

    time = list(time)

    if np.issubdtype(np.asarray(time[0]).dtype, np.datetime64) and np.issubdtype(
        np.asarray(time[1]).dtype, np.datetime64
    ):
        flag = "v_from_t"
    elif not np.issubdtype(np.asarray(time[1]).dtype, np.datetime64) and (
        float(time[1]) > constants.c / 1e3
    ):
        # Epoch in seconds (larger than the speed of light in km/s)
        flag = "v_from_t"
    else:
        flag = "dt_from_v"

    if flag == "v_from_t":
        # Time input, velocity output
        time = np.asarray(time)

        if np.issubdtype(time.dtype, np.datetime64):
            # Delays in integer ns before converting to seconds: float seconds
            # since 1970 only resolve ~0.24 us, too coarse for ms delays.
            time = time.astype("datetime64[ns]")
            tau = (time[1:] - time[0]).astype(np.int64) * 1e-9
        else:
            tau = time[1:].astype(np.float64) - float(time[0])

        if frame == "spacecraft":
            dr_mat = _get_vol_ten(r_xyz, _to_seconds(time[0]))
        else:
            # Each position at its own crossing time, as irfu-matlab
            r_cross = _positions_at(
                r_xyz, float(_to_seconds(time[0])), np.hstack([0.0, tau])
            )
            dr_mat = r_cross[1:] - r_cross[0]

        slowness = np.linalg.solve(dr_mat, tau)

        # "1/v vector"
        out = slowness / np.linalg.norm(slowness) ** 2

    else:
        # Time and velocity input, time output
        time_center = float(_to_seconds(time[0]))  # center time
        velocity = np.array(time[1:], dtype=np.float64)  # Input velocity
        slowness = velocity / np.linalg.norm(velocity) ** 2

        dr_mat = _get_vol_ten(r_xyz, time_center)

        delta_t = np.matmul(dr_mat, slowness)

        if frame == "positions":
            # MMS1 crosses at time_center; the others when the plane, moving
            # at velocity, reaches their positions at their crossing times
            delta_t = _shifts_moving(r_xyz, time_center, velocity, delta_t)

        out = np.hstack([0, delta_t])

    return out
