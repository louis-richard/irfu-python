#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray import DataArray

# Local imports
from pyrfu._core.units import _attrs, _product, _units
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar


def dot(inp1: DataArray, inp2: DataArray) -> DataArray:
    r"""Compute dot product of two fields.

    Parameters
    ----------
    inp1 : DataArray
        Time series of the first field X.
    inp2 : DataArray
        Time series of the second field Y.

    Returns
    -------
    out : DataArray
        Time series of the dot product Z = X.Y.

    Raises
    ------
    TypeError
        If inp1 or inp2 is not a xarray.DataArray.
    ValueError
        If inp1 and inp2 are not vectors with the same number of components.

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

    if not isinstance(inp1, DataArray) or not isinstance(inp2, DataArray):
        raise TypeError("inp1 and inp2 must be xarray.DataArray")

    if inp1.ndim != 2 or inp2.ndim != 2 or inp1.shape[1] != inp2.shape[1]:
        raise ValueError("inp1 and inp2 must be vectors of the same size")

    # Resample to first input sampling
    inp2 = resample(inp2, inp1)

    # Compute scalar product
    out_data = np.sum(inp1.data * inp2.data, axis=1)

    # Output to xarray
    # Product of the units (none)
    out = ts_scalar(
        inp1.time.data, out_data, attrs=_attrs({}, _product(_units(inp1), _units(inp2)))
    )

    return out
