#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from astropy.time import Time

from ..pyrf import ts_scalar, resample, extend_tint, time_clip, convert_fac


def probe_align_times(e_xyz, b_xyz, sc_pot, z_phase, plot_fig=False):
    """Returns times when field-aligned electrostatic waves can be characterized using
    interferometry techniques. The same alignment conditions as Graham et al., JGR, 2015 are
    used. Optional figure produced  showing E_FAC, probe fields, and probe potentials to view
    time delays between electric fields aligned with B.  Currently p5-p6 are not used in this
    routine; the analysis is the same as the one used for Cluster.

    For the figure the panels are :
        * (a) B in DMPA Coordinates
        * (b) Magnitude of B in and out of the spin plane
        * (c) Angles between B and probes 1 and 3 in the spin plane (angle between 0 and 90 degrees)
        * (d) Spacecraft potential from probes perpendicular to B
        * (e) E fields from p1-p4 and SC for probes closely aligned with B
        * (f) E in field-aligned coordinates
        * (g) E from probes p1-p2 and p3-p4.


    Parameters
    ----------
    e_xyz : xarray.DataArray
        Electric field in DSL coordinates, brst mode.

    b_xyz : xarray.DataArray
        Magnetic field in DMPA coordinates.

    sc_pot : xarray.DataArray
        L2 Spacecraft potential data. Timing corrections are applied in this

    z_phase : xarray.DataArray
        Spacecraft phase (z_phase). Obtained from ancillary_defatt.

    plot_fig : bool, optional
        Flag to plot fields and potentials which satisfy alignment conditions. Default is False

    Returns
    -------
    start_time1 : to fill
        Start times of intervals which satisfy the probe alignment conditions for probe combinates
        p1-p2.

    end_time1 : to fill
        End times of intervals which satisfy the probe alignment conditions for probe combinates
        p1-p2.

    start_time3 : to fill
        Start times of intervals which satisfy the probe alignment conditions for probe combinates
        p3-p4.

    end_time3 : to fill
        End times of intervals which satisfy the probe alignment conditions for probe combinates
        p3-p4.

    """

    # Correct for timing in spacecraft potential data.
    e12 = ts_scalar(sc_pot.time.data, (sc_pot.data[:, 0] - sc_pot.data[:, 1]) / .120)
    e34 = ts_scalar(sc_pot.time.data, (sc_pot.data[:, 2] - sc_pot.data[:, 3]) / .120)
    e56 = ts_scalar(sc_pot.time.data, (sc_pot.data[:, 4] - sc_pot.data[:, 4]) / .0292)

    v1 = ts_scalar(sc_pot.time.data, sc_pot.data[:, 0])
    v3 = ts_scalar(sc_pot.time.data + np.timedelta64(7629, "ns"), sc_pot.data[:, 2])
    v5 = ts_scalar(sc_pot.time.data + np.timedelta64(15259, "ns"), sc_pot.data[:, 4])

    e12.time.data += np.timedelta64(26703, "ns")
    e34.time.data += np.timedelta64(30518, "ns")
    e56.time.data += np.timedelta64(34332, "ns")

    v1, v3, v5 = [resample(v, v1) for v in [v1, v3, v5]]
    e12, e34, e56 = [resample(e, v1) for e in [e12, e34, e56]]

    v2 = v1 - e12 * 0.120
    v4 = v3 - e34 * 0.120
    v6 = v5 - e56 * 0.0292

    sc_pot = np.hstack([v1.data, v2.data, v3.data, v4.data, v5.data, v6.data])

    sc_pot = xr.DataArray(sc_pot, coords=[v1.time.data, np.arange(1, 7)], dims=["time", "probe"])

    t_limit = list(Time(sc_pot.time.data[[0, -1]], format="datetime64").isot)

    t_limit_long = extend_tint(t_limit, [-10, 10])

    b_xyz = time_clip(b_xyz, t_limit_long)
    b_xyz = resample(b_xyz, sc_pot)
    e_xyz = resample(e_xyz, sc_pot)

    z_phase = time_clip(z_phase, t_limit_long)

    # Remove repeated z_phase elements
    n_ph = len(z_phase)
    no_repeat = np.ones(n_ph)

    for i in range(1, n_ph):
        if z_phase.time.data[i] > z_phase.time.data[i - 1]:
            if z_phase.data[i] < z_phase.data[i - 1]:
                z_phase.data[i:] += 360.
        else:
            no_repeat[i] = 0

    z_phase_time = z_phase.time[no_repeat == 1]
    z_phase_data = z_phase.data[no_repeat == 1]

    z_phase = ts_scalar(z_phase_time, z_phase_data)
    z_phase = resample(z_phase, sc_pot)

    # Perform rotation on e_xyz into field - aligned coordinates
    e_fac = convert_fac(e_xyz, b_xyz, [1, 0, 0])

    # Probe angles in DSL or whatever
    idx_a, idx_b = [[1, 7, 2, 5], [6, 6, 3, 3]]
    phase_p = [z_phase.data / 180 * np.pi + i * np.pi / j for i, j in zip(idx_a, idx_b)]

    rp = [np.array([60 * np.cos(phase), 60 * np.sin(phase)]) for phase in phase_p]

    # Calculate angles between probes and direction of B in the spin plane.
    theta_pb = [None, None, None, None]

    for i in [0, 2]:
        theta_pb[i] = rp[i][:, 0] * b_xyz.data[:, 0] + rp[i][:, 1] * b_xyz.data[:, 1]
        theta_pb[i] /= np.sqrt(rp[i][:, 0] ** 2 + rp[i][:, 1] ** 2)
        theta_pb[i] /= np.sqrt(b_xyz[:, 0] ** 2 + b_xyz[:, 1] ** 2)
        theta_pb[i] = np.arccos(abs(theta_pb[i])) * 180 / np.pi

    theta_pb[1] = theta_pb[0]
    theta_pb[3] = theta_pb[2]

    sc_v12 = (sc_pot.data[:, 0] + sc_pot.data[:, 1]) / 2
    sc_v34 = (sc_pot.data[:, 2] + sc_pot.data[:, 3]) / 2

    e1, e3 = [(sc_pot.data[:, i] - sc_v) * 1e3 / 60 for i, sc_v in zip([0, 2], [sc_v34, sc_v12])]
    e2, e4 = [(sc_v - sc_pot.data[:, i]) * 1e3 / 60 for i, sc_v in zip([0, 2], [sc_v34, sc_v12])]

    es = [e1, e2, e3, e4]

    e12 = (sc_pot.data[:, 0] - sc_pot.data[:, 1]) * 1e3 / 120
    e34 = (sc_pot.data[:, 2] - sc_pot.data[:, 3]) * 1e3 / 120

    idx_b = np.sqrt(b_xyz.data[:, 0] ** 2 + b_xyz.data[:, 1] ** 2) < abs(b_xyz.data[:, 2])
    thresh_ang = 25.0

    for e, theta in zip(es, theta_pb):
        e[theta > thresh_ang] = np.nan
        e[idx_b] = np.nan

    sc_v12[theta_pb[2] > thresh_ang] = np.nan
    sc_v34[theta_pb[0] > thresh_ang] = np.nan

    sc_v12[idx_b] = np.nan
    sc_v34[idx_b] = np.nan

    if plot_fig:
        f, ax = plt.subplots(7, sharex="all", figsize=(16, 9))
        f.subplots_adjust(left=.08, right=.92, bottom=.05, top=.95, hspace=0)

        plt.show()

    return
