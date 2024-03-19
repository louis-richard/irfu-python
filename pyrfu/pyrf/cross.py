#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def cross(inp1: DataArray, inp2: DataArray) -> DataArray:
    r"""Compute cross product of two fields.

    Parameters
    ----------
    inp1 : DataArray
        Time series of the first field X.
    inp2 : DataArray
        Time series of the second field Y.

    Returns
    -------
    DataArray
        Time series of the cross product Z = XxY.

    Raises
    ------
    TypeError
        If inp1 or inp2 are not xarray.DataArray.
    ValueError
        If inp1 or inp2 are not vectors timeseries.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Define time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of the MMS spacecraft

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("e_gse_edp_fast_l2", tint, mms_id)

    Compute magnitude of the magnetic field

    >>> b_mag = pyrf.norm(b_xyz)

    Compute ExB drift velocity

    >>> v_xyz_exb = pyrf.cross(e_xyz, b_xyz) / b_mag ** 2

    """
    # Check type
    if not isinstance(inp1, DataArray):
        raise TypeError("inp1 must be a xarray.DataArray")

    if not isinstance(inp2, DataArray):
        raise TypeError("inp2 must be a xarray.DataArray")

    # Check inputs are vectors
    if inp1.ndim != 2 or inp1.shape[1] != 3:
        raise ValueError("inp1 must be a vector")

    if inp2.ndim != 2 or inp2.shape[1] != 3:
        raise ValueError("inp2 must be a vector")

    if len(inp1) != len(inp2):
        inp2 = resample(inp2, inp1)

    out_data = np.cross(inp1.data, inp2.data, axis=1)

    out = ts_vec_xyz(inp1.time.data, out_data)

    return out
