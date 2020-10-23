#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e_vxb.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def e_vxb(v=None, b=None, flag="vxb"):
    """Computes the convection electric field :math:`\\mathbf{V}\\times\\mathbf{B}` (default) or the
    :math:`\\mathbf{E}\\times\\mathbf{B}/|\\mathbf{B}|^{2}` drift velocity (flag="exb").

    Parameters
    ----------
    v : xarray.DataArray
        Time series of the velocity/electric field.

    b : xarray.DataArray
        Time series of the magnetic field.

    flag : str
        Method flag :
            * "vxb" : computes convection electric field (default).
            * "exb" : computes ExB drift velocity.

    Returns
    -------
    out : xarray.DataArray
        Time series of the convection electric field/ExB drift velocity.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    >>> # Spacecraft index
    >>> mms_id = 1
    >>> # Load magnetic field and electric field
    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)
    >>> # Compute ExB drift velocity
    >>> v_xyz_exb = pyrf.e_vxb(e_xyz, b_xyz,"ExB")

    """

    assert v is not None and isinstance(v, xr.DataArray)
    assert b is not None and isinstance(b, xr.DataArray)

    if flag.lower() == "exb":
        estimate_exb = True
        estimate_vxb = False
    else:
        estimate_exb = False
        estimate_vxb = True

    if v.ndim == 1 and len(v) == 3:
        input_v_cons = True
    else:
        input_v_cons = False

    if estimate_exb:
        e = v

        if len(e) != len(b):
            b = resample(b, e)

        res = 1e3 * np.cross(e.data, b.data, axis=1)
        res /= np.linalg.norm(b.data, axis=1)[:, None] ** 2

        attrs = {"UNITS": "km/s", "FIELDNAM": "Velocity", "LABLAXIS": "V"}

    elif estimate_vxb:
        if input_v_cons:
            res = np.cross(np.tile(v, (len(b), 1)), b.data) * (-1) * 1e-3

        else:
            if len(v) != len(b):
                b = resample(b, v)

            res = np.cross(v.data, b.data) * (-1) * 1e-3

        attrs = {"UNITS": "mV/s", "FIELDNAM": "Electric field", "LABLAXIS": "E"}

    else:
        raise ValueError("Invalid flag")

    out = ts_vec_xyz(b.time.data, res, attrs)

    return out
