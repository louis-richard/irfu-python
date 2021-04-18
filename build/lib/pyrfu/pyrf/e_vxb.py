#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""e_vxb.py
@author: Louis Richard
"""

import numpy as np

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def e_vxb(v_xyz, b_xyz, flag="vxb"):
    """Computes the convection electric field :math:`\\mathbf{V}\\times
    \\mathbf{B}` (default) or the :math:`\\mathbf{E}\\times\\mathbf{
    B}/|\\mathbf{B}|^{2}` drift velocity (flag="exb").

    Parameters
    ----------
    v_xyz : xarray.DataArray
        Time series of the velocity/electric field.

    b_xyz : xarray.DataArray
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

    estimate_exb = False
    input_v_cons = False

    if flag.lower() == "exb":
        estimate_exb = True

    if v_xyz.size == 3:
        input_v_cons = True

    if estimate_exb:
        if len(v_xyz) != len(b_xyz):
            b_xyz = resample(b_xyz, v_xyz)

        res = 1e3 * np.cross(v_xyz.data, b_xyz.data, axis=1)
        res /= np.linalg.norm(b_xyz.data, axis=1)[:, None] ** 2

        attrs = {"UNITS": "km/s", "FIELDNAM": "Velocity",
                 "LABLAXIS": "V"}

    else:
        if input_v_cons:
            res = np.cross(np.tile(v_xyz, (len(b_xyz), 1)), b_xyz.data) * (-1) * 1e-3

        else:
            if len(v_xyz) != len(b_xyz):
                b_xyz = resample(b_xyz, v_xyz)

            res = np.cross(v_xyz.data, b_xyz.data) * (-1) * 1e-3

        attrs = {"UNITS": "mV/s", "FIELDNAM": "Electric field",
                 "LABLAXIS": "E"}

    out = ts_vec_xyz(b_xyz.time.data, res, attrs)

    return out
