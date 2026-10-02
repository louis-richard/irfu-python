#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.resample import resample
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _shifted(time, data, shift_ns, ref=None):
    # Time series with the time tags shifted, resampled to the reference
    out = ts_scalar(time + np.timedelta64(shift_ns, "ns"), data)
    return out if ref is None else resample(out, ref)


def _valid_runs(time, data):
    # Start and end times of the runs of non-NaN samples
    edges = np.diff(np.r_[0, (~np.isnan(data)).astype(int), 0])
    return time[edges[:-1] == 1], time[np.where(edges == -1)[0] - 1]


def probe_align_times(e_xyz, b_xyz, sc_pot, z_phase):
    r"""Returns times when field-aligned electrostatic waves can be
    characterized using interferometry techniques. The same alignment
    conditions as Graham et al., JGR, 2015 are used: the angle between B and
    the probes in the spin plane is less than 25 degrees, and B is closer to
    the spin plane than to the spin axis. Currently p5-p6 are not used in
    this routine; the analysis is the same as the one used for Cluster.

    Port of mms.probe_align_times in irfu-matlab (without the figure). The
    intervals start at their first valid sample, one sample later than in
    irfu-matlab.

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Electric field in DSL coordinates, brst mode. Not used: only needed
        for the figure in irfu-matlab.
    b_xyz : xarray.DataArray
        Magnetic field in DMPA coordinates.
    sc_pot : xarray.DataArray
        L2 probe potentials (6 probes). Timing corrections are applied in
        this function.
    z_phase : xarray.DataArray
        Spacecraft phase (z_phase) in degrees. Obtained from ancillary_defatt.

    Returns
    -------
    start_time1 : ndarray
        Start times of intervals which satisfy the probe alignment conditions
        for probe combinates p1-p2.
    end_time1 : ndarray
        End times of intervals which satisfy the probe alignment conditions
        for probe combinates p1-p2.
    start_time3 : ndarray
        Start times of intervals which satisfy the probe alignment conditions
        for probe combinates p3-p4.
    end_time3 : ndarray
        End times of intervals which satisfy the probe alignment conditions
        for probe combinates p3-p4.

    """

    del e_xyz  # only used for the figure in irfu-matlab

    # Correct for timing in spacecraft potential data.
    time, pot = [sc_pot.time.data, sc_pot.data]
    v_1 = ts_scalar(time, pot[:, 0])
    v_3 = _shifted(time, pot[:, 2], 7629, v_1)
    v_5 = _shifted(time, pot[:, 4], 15259, v_1)
    e12 = _shifted(time, (pot[:, 0] - pot[:, 1]) / 0.120, 26703, v_1)
    e34 = _shifted(time, (pot[:, 2] - pot[:, 3]) / 0.120, 30518, v_1)
    e56 = _shifted(time, (pot[:, 4] - pot[:, 5]) / 0.0292, 34332, v_1)

    v_all = np.column_stack(
        [
            v_1.data,
            v_1.data - e12.data * 0.120,
            v_3.data,
            v_3.data - e34.data * 0.120,
            v_5.data,
            v_5.data - e56.data * 0.0292,
        ]
    )

    t_limit_long = np.array([time[0], time[-1]]) + np.array(
        [-10, 10], dtype="timedelta64[s]"
    )

    b_xyz = resample(time_clip(b_xyz, t_limit_long), v_1).data

    # Remove repeated z_phase elements and unwrap the phase
    z_phase = time_clip(z_phase, t_limit_long)
    no_repeat = np.r_[True, np.diff(z_phase.time.data) > np.timedelta64(0, "ns")]
    z_phase_data = z_phase.data[no_repeat].astype(np.float64)
    z_phase_data += 360.0 * np.r_[0, np.cumsum(np.diff(z_phase_data) < 0)]
    z_phase = ts_scalar(z_phase.time.data[no_repeat], z_phase_data)
    z_phase = resample(z_phase, v_1).data

    # Angles between probes 1 and 3 and the direction of B in the spin plane
    b_plane = np.sqrt(b_xyz[:, 0] ** 2 + b_xyz[:, 1] ** 2)
    theta_pb = []
    for offset in [np.pi / 6, 2 * np.pi / 3]:
        phase = np.deg2rad(z_phase) + offset
        cos_pb = np.cos(phase) * b_xyz[:, 0] + np.sin(phase) * b_xyz[:, 1]
        theta_pb.append(np.rad2deg(np.arccos(np.abs(cos_pb / b_plane))))

    sc_v12 = (v_all[:, 0] + v_all[:, 1]) / 2
    sc_v34 = (v_all[:, 2] + v_all[:, 3]) / 2

    # Fields between the single probes and the spacecraft
    e_1 = (v_all[:, 0] - sc_v34) * 1e3 / 60
    e_3 = (v_all[:, 2] - sc_v12) * 1e3 / 60

    thresh_ang = 25.0
    idx_b = b_plane < np.abs(b_xyz[:, 2])

    e_1[(theta_pb[0] > thresh_ang) | idx_b] = np.nan
    e_3[(theta_pb[1] > thresh_ang) | idx_b] = np.nan

    start_time1, end_time1 = _valid_runs(v_1.time.data, e_1)
    start_time3, end_time3 = _valid_runs(v_1.time.data, e_3)

    return start_time1, end_time1, start_time3, end_time3
