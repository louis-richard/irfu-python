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


def c_4_v(r_xyz, time):
    r"""Calculates velocity or time shift of discontinuity as in [6]_.

    Parameters
    ----------
    r_xyz : list
        Time series of the positions of the 4 spacecraft in km.
    time : list
        Either the crossing times of the 4 spacecraft (datetime64 or epoch in
        seconds), or the reference time followed by the velocity of the
        discontinuity in km/s, [t, v_x, v_y, v_z].

    Returns
    -------
    out : numpy.ndarray
        Velocity of the discontinuity in km/s (from crossing times), or time
        shifts in s of the 4 spacecraft with respect to MMS1 (from a velocity).
        The positions are taken at the first (reference) time.

    References
    ----------
    .. [6]	Vogt, J., Haaland, S., and Paschmann, G. (2011) Accuracy
            of multi-point boundary crossing time analysis, Ann.
            Geophys., 29, 2239-2252, doi :
            https://doi.org/10.5194/angeo-29-2239-2011

    """

    time = list(time)

    if np.issubdtype(np.asarray(time[0]).dtype, np.datetime64) and np.issubdtype(
        np.asarray(time[1]).dtype, np.datetime64
    ):
        flag = "v_from_t"
    elif not np.issubdtype(np.asarray(time[1]).dtype, np.datetime64) and (
        float(time[1]) > 299792.458
    ):
        # Epoch in seconds (larger than the speed of light in km/s)
        flag = "v_from_t"
    else:
        flag = "dt_from_v"

    if flag == "v_from_t":
        # Time input, velocity output
        time = _to_seconds(time)
        dr_mat = _get_vol_ten(r_xyz, time[0])
        tau = time[1:] - time[0]
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
        out = np.hstack([0, delta_t])

    return out
