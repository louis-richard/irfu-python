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

"""gradient.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from .calc_dt import calc_dt


def gradient(inp):
    """Computes time derivative of the input variable.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time derivative of the input variable.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2017-07-18T13:03:34.000", "2017-07-18T13:07:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)

    Time derivative of the magnetic field

    >>> db_dt = pyrf.gradient(b_xyz)

    """

    # guess time step
    delta_t = calc_dt(inp)

    d_inp_dt = np.gradient(inp.data, axis=0) / delta_t

    out = xr.DataArray(d_inp_dt, coords=inp.coords, dims=inp.dims,
                       attrs=inp.attrs)

    if "UNITS" in out.attrs:
        out.attrs["UNITS"] += "/s"

    return out
