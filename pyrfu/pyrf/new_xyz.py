#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def new_xyz(inp, trans_mat, coordinate_system: Optional[str] = None):
    r"""Transform the input field to the new frame.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field in the original coordinate system.
    trans_mat : array_like
        Transformation matrix, with the unit vectors of the new frame as columns.
    coordinate_system : str, Optional
        Name of the new coordinate system (e.g., "lmn"), stored in the
        ``COORDINATE_SYSTEM`` attribute of the output. Default is None, which
        removes that attribute: the name of an arbitrary frame is unknown, and
        keeping the original one would let functions relying on it, like
        :func:`pyrfu.pyrf.cotrans`, treat the output as still being in the
        original frame.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input in the new frame.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute MVA frame

    >>> b_lmn, l, mva = pyrf.mva(b_xyz)

    Move electric field to the MVA frame

    >>> e_lmn = pyrf.new_xyz(e_xyz, mva, "lmn")

    """

    if inp.data.ndim == 3:
        out_data = np.matmul(np.matmul(trans_mat.T, inp.data), trans_mat)
    else:
        out_data = (trans_mat.T @ inp.data.T).T

    if coordinate_system is not None and not isinstance(coordinate_system, str):
        raise TypeError("coordinate_system must be a string or None")

    # Copy so that the caller's attributes are unchanged
    attrs = dict(inp.attrs)

    if coordinate_system is None:
        attrs.pop("COORDINATE_SYSTEM", None)
    else:
        attrs["COORDINATE_SYSTEM"] = coordinate_system

    out = xr.DataArray(
        out_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=attrs,
    )

    return out
