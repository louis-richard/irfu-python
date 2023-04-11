#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def dot(inp1, inp2):
    r"""Computes dot product of two fields.

    Parameters
    ----------
    inp1 : xarray.DataArray
        Time series of the first field X.
    inp2 : xarray.DataArray
        Time series of the second field Y.

    Returns
    -------
    out : xarray.DataArray
        Time series of the dot product Z = X.Y.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Load magnetic field, electric field and spacecraft position

    >>> r_mms, b_mms, e_mms = [[] * 4 for _ in range(3)]
    >>> for mms_id in range(1, 5):
    >>>		r_mms.append(mms.get_data("R_gse", tint, mms_id))
    >>> 	b_mms.append(mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id))
    >>>		e_mms.append(mms.get_data("E_gse_edp_fast_l2", tint, mms_id))

    Compute current density using curlometer technique

    >>> j_xyz, _, _, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Compute the electric at the center of mass of the tetrahedron

    >>> e_xyz = pyrf.avg_4sc(e_mms)

    Compute J.E dissipation

    >>> je = pyrf.dot(j_xyz, e_xyz)

    """

    # Resample to first input sampling
    inp2 = resample(inp2, inp1)

    # Compute scalar product
    out_data = np.sum(inp1.data * inp2.data, axis=1)

    # Output to xarray
    out = ts_scalar(inp1.time.data, out_data)

    return out
