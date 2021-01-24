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

from ..pyrf import ts_scalar, resample


def correct_edp_probe_timing(v0):
    """Corrects for the channel delays not accounted for in the MMS EDP processing. As described
    in the MMS EDP data products guide.

    Parameters
    ----------
    v0 : xarray.DataArray
        Time series created from L2 sc_pot files, from the variable "mms#_edp_dcv_brst_l2"
        containing individual probe potentials.

    Returns
    -------
    v_corrected : xarray.DataArray
        Time series where the channel delay for each probe have been accounted and corrected for.

    Notes
    -----
    This function is only useful for Burst mode data. For the other telemetry modes (i.e. slow
    and fast) the channel delays are completely negligible and the interpolation and resampling
    applied here will have no effect other than possibly introduce numerical noise.

    """

    e_fact = [.1200, .1200, .0292]

    # Reconstruct E12, E34, E56 as computed in MMS processing
    t = v0.time.data
    es = [ts_scalar(t, np.diff(v0.data[:, i:i + 2]) / fact) for i, fact in zip([0, 2, 4], e_fact)]

    # Correct the time tags to create individual time series
    tau_vs = [np.timedelta64(0, "ns"), np.timedelta64(7629, "ns"), np.timedelta64(15259, "ns")]
    tau_es = [np.timedelta64(26703, "ns"), np.timedelta64(30518, "ns"), np.timedelta64(34332, "ns")]

    # Odds probes potential 1, 3, 5
    vo = [ts_scalar(t + tau, v0.data[:, i]) for tau, i in zip(tau_vs, [0, 2, 4])]

    # Electric field
    es = [ts_scalar(e.time.data + tau, e.data) for e, tau in zip(es, tau_es)]

    # Resample all data to time tags of V1 (i.e. timeOrig).
    vo = [resample(v, v0) for v in vo]
    es = [resample(e, v0) for e in es]

    # Recompute individual even probe potentials 2, 4, 6
    ve = [v - e * fact for v, e, fact in zip(vo, es, e_fact)]

    v = [item.data for tup in zip(ve, vo) for item in tup]

    # Create the new time series with the corrected values
    probe_index = np.arange(1, 7)
    v_corrected = xr.DataArray(v, coords=[t, probe_index], dims=["time", "probe"])

    return v_corrected
