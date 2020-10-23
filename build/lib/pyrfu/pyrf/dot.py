#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dot.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .ts_scalar import ts_scalar


def dot(x=None, y=None):
    """Computes dot product of two fields.

    Parameters
    ----------
    x : xarray.DataArray
        Time series of the first field X.

    y : xarray.DataArray
        Time series of the second field Y.

    Returns
    -------
    out : xarray.DataArray
        Time series of the dot product Z = X.Y.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    >>> # Load magnetic field, electric field and spacecraft position
    >>> r_mms, b_mms, e_mms = [[] * 4 for _ in range(3)]
    >>> for mms_id in range(1, 5):
    >>>		r_mms.append(mms.get_data("R_gse", tint, mms_id))
    >>> 	b_mms.append(mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id))
    >>>		e_mms.append(mms.get_data("E_gse_edp_fast_l2", tint, mms_id))
    >>>
    >>> j_xyz, div_b, b_avg, jxb, div_t_shear, div_pb = pyrf.c_4_j(r_mms, b_mms)
    >>> # Compute the electric at the center of mass of the tetrahedron
    >>> e_xyz = pyrf.avg_4sc(e_mms)
    >>> # Compute J.E dissipation
    >>> je = pyrf.dot(j_xyz, e_xyz)

    """

    assert x is not None and isinstance(x, xr.DataArray)
    assert y is not None and isinstance(y, xr.DataArray)

    # Resample to first input sampling
    y = resample(y, x)

    # Compute scalar product
    out_data = np.sum(x.data * y.data, axis=1)

    # Output to xarray
    out = ts_scalar(x.time.data, out_data)

    return out
