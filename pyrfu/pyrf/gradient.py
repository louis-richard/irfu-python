#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gradient.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .calc_dt import calc_dt


def gradient(inp=None):
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
    >>> # Time interval
    >>> tint = ["2017-07-18T13:03:34.000", "2017-07-18T13:07:00.000"]
    >>> # Spacecraft index
    >>> mms_id = 1
    >>> # Load magnetic field
    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> # Time derivative of the magnetic field
    >>> db_dt = pyrf.gradient(b_xyz)

    """

    assert inp is not None and isinstance(inp, xr.DataArray)

    # guess time step
    dt = calc_dt(inp)

    d_inp_dt = np.gradient(inp.data, axis=0) / dt

    out = xr.DataArray(d_inp_dt, coords=inp.coords, dims=inp.dims, attrs=inp.attrs)

    if "UNITS" in out.attrs:
        out.attrs["UNITS"] += "/s"

    return out
