#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _get_vol_ten(r_xyz, time):
    if len(time) == 1:
        time = np.array([time, time, time, time])

    tckr_x, tckr_y, tckr_z = [[], [], []]

    for i in range(4):
        tckr_x.append(
            interpolate.interp1d(r_xyz[i].time.data, r_xyz[i].data[:, 0]),
        )
        tckr_y.append(
            interpolate.interp1d(r_xyz[i].time.data, r_xyz[i].data[:, 1]),
        )
        tckr_z.append(
            interpolate.interp1d(r_xyz[i].time.data, r_xyz[i].data[:, 2]),
        )

        r_xyz[i] = np.array(
            [tckr_x[i](time[0]), tckr_y[i](time[0]), tckr_z[i](time[0])],
        )

    # Volumetric tensor with SC1 as center.
    dr_mat = (np.vstack(r_xyz[1:]) - np.tile(r_xyz[0], (3, 1))).T

    return dr_mat


def c_4_v(r_xyz, time):
    r"""Calculates velocity or time shift of discontinuity as in [6]_.

    Parameters
    ----------
    r_xyz : list
        Time series of the positions of the 4 spacecraft.
    time : list
        Crossing times or time and velocity.

    Returns
    -------
    out : ndarray
        Discontinuity velocity or time shift with respect to mms1.

    References
    ----------
    .. [6]	Vogt, J., Haaland, S., and Paschmann, G. (2011) Accuracy
            of multi-point boundary crossing time analysis, Ann.
            Geophys., 29, 2239-2252, doi :
            https://doi.org/10.5194/angeo-29-2239-2011

    """

    if isinstance(time, np.ndarray) and time.dtype == np.datetime64:
        flag = "v_from_t"

        time = time.view("i8") * 1e-9
    elif time[1] > 299792.458:
        flag = "v_from_t"
    else:
        flag = "dt_from_v"

    if flag.lower() == "v_from_t":
        # Time input, velocity output
        dr_mat = _get_vol_ten(r_xyz, time)
        tau = np.array(time[1:]) - time[0]
        slowness = np.linalg.solve(dr_mat, tau)

        # "1/v vector"
        out = slowness / np.linalg.norm(slowness) ** 2

    elif flag.lower() == "dt_from_v":
        # Time and velocity input, time output
        time_center = time[0]  # center time
        velocity = np.array(time[1:])  # Input velocity
        slowness = velocity / np.linalg.norm(velocity) ** 2

        dr_mat = _get_vol_ten(r_xyz, time_center)

        delta_t = np.matmul(dr_mat, slowness)
        out = np.hstack([0, delta_t])

    else:
        raise ValueError("Invalid flag")

    return out
