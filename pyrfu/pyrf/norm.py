#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .ts_scalar import ts_scalar


def norm(inp):
    r"""Computes the magnitude of the input field.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field, a vector (time, n) or a scalar (its
        absolute value).

    Returns
    -------
    out : xarray.DataArray
        Time series of the magnitude of the input field.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If inp is not a scalar or a vector (a scalar raised AxisError, a
        tensor ValueError from ts_scalar).

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)

    Compute magnitude of the magnetic field

    >>> b_mag = pyrf.norm(b_xyz)

    """

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if inp.ndim == 1:
        magnitude = np.abs(inp.data)
    elif inp.ndim == 2:
        magnitude = np.linalg.norm(inp.data, axis=1)
    else:
        raise ValueError("inp must be a scalar or a vector time series")

    out = ts_scalar(inp.time.data, magnitude, attrs=inp.attrs)

    return out
