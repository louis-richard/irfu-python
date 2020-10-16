#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_align_times.py

@author : Louis RICHARD
"""
import numpy as np

from ..pyrf import ts_scalar


def probe_align_times(e_xyz=None, b_xyz=None, sc_pot=None, z_phase=None, plot_fig=False):
    """
    Returns times when field-aligned electrostatic waves can be characterized using interferometry techniques. The same
    alignment conditions as Graham et al., JGR, 2015 are used. Optional figure produced showing E_FAC, probe fields,
    and probe potentials to view  time delays between electric fields aligned with B. Currently p5-p6 are not used in
    this routine; the analysis is the same as the one used for Cluster.

    For the figure the panels are:
    - (a) B in DMPA Coordinates
    - (b) Magnitude of B in and out of the spin plane
    - (c) Angles between B and probes 1 and 3 in the spin plane (angle between 0 and 90 degrees)
    - (d) Spacecraft potential from probes perpendicular to B
    - (e) E fields from p1-p4 and SC for probes closely aligned with B
    - (f) E in field-aligned coordinates
    - (g) E from probes p1-p2 and p3-p4.

    Parameters
    ----------
    e_xyz = xarray.DataArray
        Electric field in DSL coordinates, brst mode.

    b_xyz : xarray.DataArray
        Magnetic field in DMPA coordinates.

    sc_pot : xarray.DataArray
        L2 SCpot data. Timing corrections are applied in this

    z_phase : xarray.DataArray
        Spacecraft phase (zphase). Obtained from ancillary_defatt.

    plot_fig : bool
        (Optional) Flag to plot fields and potentials which satisfy alignment conditions. Default is False

    Returns
    -------
    start_time1 : to fill
        Start times of intervals which satisfy the probe alignment conditions for probe combinates p1-p2.

    end_time1 : to fill
        End times of intervals which satisfy the probe alignment conditions for probe combinates p1-p2.

    start_time3 : to fill
        Start times of intervals which satisfy the probe alignment conditions for probe combinates p3-p4.

    end_time3 : to fill
        End times of intervals which satisfy the probe alignment conditions for probe combinates p3-p4.


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

    return