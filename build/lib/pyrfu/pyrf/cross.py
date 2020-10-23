#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def cross(x=None, y=None):
    """Computes cross product of two fields.

    Parameters
    ----------
    x : xarray.DataArray
        Time series of the first field X.

    y : xarray.DataArray
        Time series of the second field Y.

    Returns
    -------
    out : xarray.DataArray
        Time series of the cross product Z = XxY.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Define time interval
    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    >>> # Index of the MMS spacecraft
    >>> mms_id = 1
    >>> # Load magnetic field and electric field
    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)
    >>> # Compute magnitude of the magnetic field
    >>> b_mag = pyrf.norm(b_xyz)
    >>> # Compute ExB drift velocity
    >>> v_xyz_exb = pyrf.cross(e_xyz, b_xyz) / b_mag ** 2

    """

    assert x is not None and isinstance(x, xr.DataArray)
    assert y is not None and isinstance(y, xr.DataArray)

    if len(x) != len(y):
        y = resample(y, x)

    out_data = np.cross(x.data, y.data, axis=1)

    out = ts_vec_xyz(x.time.data, out_data)

    return out
