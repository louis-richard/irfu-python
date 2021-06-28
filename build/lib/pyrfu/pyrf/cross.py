#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def cross(inp1, inp2):
    r"""Computes cross product of two fields.

    Parameters
    ----------
    inp1 : xarray.DataArray
        Time series of the first field X.
    inp2 : xarray.DataArray
        Time series of the second field Y.

    Returns
    -------
    out : xarray.DataArray
        Time series of the cross product Z = XxY.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Define time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of the MMS spacecraft

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute magnitude of the magnetic field

    >>> b_mag = pyrf.norm(b_xyz)

    Compute ExB drift velocity

    >>> v_xyz_exb = pyrf.cross(e_xyz, b_xyz) / b_mag ** 2

    """

    if len(inp1) != len(inp2):
        inp2 = resample(inp2, inp1)

    out_data = np.cross(inp1.data, inp2.data, axis=1)

    out = ts_vec_xyz(inp1.time.data, out_data)

    return out
