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


def e_vxb(v_xyz, b_xyz, flag: str = "vxb"):
    r"""Computes the convection electric field :math:`V\times B` (default)
    or the :math:`E\times B/|B|^{2}` drift velocity (flag="exb").

    Parameters
    ----------
    v_xyz : xarray.DataArray
        Time series of the velocity/electric field.
    b_xyz : xarray.DataArray
        Time series of the magnetic field.
    flag : {"vxb", "exb"}, Optional
        Method flag :
            * "vxb" : computes convection electric field (Default).
            * "exb" : computes ExB drift velocity.

    Returns
    -------
    out : xarray.DataArray
        Time series of the convection electric field/ExB drift velocity.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute ExB drift velocity

    >>> v_xyz_exb = pyrf.e_vxb(e_xyz, b_xyz,"ExB")

    """

    input_v_cons = v_xyz.size == 3
    estimate_exb = flag.lower() == "exb"

    if estimate_exb:
        b_xyz = resample(b_xyz, v_xyz)

        res = 1e3 * np.cross(v_xyz.data, b_xyz.data, axis=1)
        res /= np.linalg.norm(b_xyz.data, axis=1)[:, None] ** 2

        attrs = {"UNITS": "km/s", "FIELDNAM": "Velocity",
                 "LABLAXIS": "V"}

    else:
        if input_v_cons:
            res = np.cross(np.tile(v_xyz, (len(b_xyz), 1)), b_xyz.data)
            res *= (-1) * 1e-3

        else:
            b_xyz = resample(b_xyz, v_xyz)

            res = np.cross(v_xyz.data, b_xyz.data)
            res *= (-1) * 1e-3

        attrs = {"UNITS": "mV/s", "FIELDNAM": "Electric field",
                 "LABLAXIS": "E"}

    out = ts_vec_xyz(b_xyz.time.data, res, attrs)

    return out
