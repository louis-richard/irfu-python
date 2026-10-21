#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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
        Time series of the time derivative of the input variable, per second.

    Notes
    -----
    Second-order central differences with the actual time steps (also across
    gaps and for uneven sampling), and first-order differences at the ends (see
    numpy.gradient).

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

    # Times in seconds from the first sample, from the exact time steps in ns (the
    # median time step was wrong across gaps and for jittered times)
    time = inp.time.data.astype("datetime64[ns]")
    time = (time - time[0]).astype(np.int64) * 1e-9

    d_inp_dt = np.gradient(inp.data, time, axis=0)

    out = xr.DataArray(
        d_inp_dt,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs,
    )

    if "UNITS" in out.attrs:
        out.attrs["UNITS"] += "/s"

    return out
