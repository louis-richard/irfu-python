#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def new_xyz(inp, trans_mat):
    r"""Transform the input field to the new frame.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field in the original coordinate system.
    trans_mat : array_like
        Transformation matrix.

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

    >>> e_lmn = pyrf.new_xyz(e_xyz, mva)

    """

    if inp.data.ndim == 3:
        out_data = np.matmul(np.matmul(trans_mat.T, inp.data), trans_mat)
    else:
        out_data = (trans_mat.T @ inp.data.T).T

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims,
                       attrs=inp.attrs)

    return out
