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

"""integrate.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from astropy.time import Time


def integrate(inp, time_step):
    """Integrate time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the variable to integrate.

    time_step : float
        Time steps threshold. All time_steps larger than 3*time_step
        are assumed data gaps, default is that time_step is the
        smallest value of all time_steps of the time series.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time integrated input.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-12-14T01:17:40.200", "2015-12-14T01:17:41.500"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Convert electric field to field aligned coordinates

    >>> e_xyzfac = pyrf.convert_fac(e_xyz, b_xyz, [1, 0, 0])

    """

    time_tmp = Time(inp.time.data, format="datetime64").unix
    data_tmp = inp.data
    unit_tmp = inp.attrs["UNITS"]

    data = np.hstack([time_tmp, data_tmp])

    delta_t = np.hstack([0, np.diff(data[:, 0])])

    if time_step is None:
        time_steps = np.diff(data[:, 0])

        # remove the smallest time step in case some problems
        time_step = np.min(np.delete(time_steps, np.argmin(time_steps)))

    delta_t[delta_t > 3 * time_step] = 0

    x_int = data
    for j in range(1, x_int.shape[1]):
        j_ok = ~np.isnan(x_int[:, j])

        x_int[j_ok, j] = np.cumsum(data[j_ok, j] * delta_t[j_ok])

    out = xr.DataArray(data[:, 1:], coords=inp.coords, dims=inp.dims)
    out.attrs["UNITS"] = unit_tmp + "*s"

    return out
