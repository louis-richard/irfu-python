#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def gradient(inp):
    r"""Computes time derivative of the input variable.

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
