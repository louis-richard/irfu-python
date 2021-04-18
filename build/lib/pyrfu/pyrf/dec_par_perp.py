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

"""dec_par_perp.py
@author: Louis Richard
"""

import numpy as np

from .resample import resample
from .dot import dot


def dec_par_perp(inp, b_bgd, flag_spin_plane=False):
    """Decomposes a vector into par/perp to B components. If
    flagspinplane decomposes components to the projection of ``b0``
    into the XY plane. ``alpha`` gives the angle between ``b0`` and
    the XY. plane.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the field to decompose.

    b_bgd : xarray.DataArray
        Time series of the background magnetic field.

    flag_spin_plane : bool, optional
        Flag if True gives the projection in XY plane.

    Returns
    -------
    a_para : xarray.DataArray
        Time series of the input field parallel to the background
        magnetic field.

    a_perp : xarray.DataArray
        Time series of the input field perpendicular to the background
        magnetic field.

    alpha : xarray.DataArray
        Time series of the angle between the background magnetic field
        and the XY plane.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field (FGM) and electric field (EDP)

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Decompose e_xyz into parallel and perpendicular to b_xyz components

    >>> e_para, e_perp, _ = pyrf.dec_par_perp(e_xyz, b_xyz)

    """

    if not flag_spin_plane:
        b_mag = np.linalg.norm(b_bgd, axis=1, keepdims=True)

        indices = np.where(b_mag < 1e-3)[0]

        if indices.size > 0:
            b_mag[indices] = np.ones(len(indices)) * 1e-3

        b_hat = b_bgd / b_mag
        b_hat = resample(b_hat, inp)

        a_para = dot(b_hat, inp)
        a_perp = inp.data - (b_hat * np.tile(a_para.data, (3, 1)).T)
        alpha = []
    else:
        b_bgd = resample(b_bgd, inp)
        b_tot = np.sqrt(b_bgd[:, 0] ** 2 + b_bgd[:, 1] ** 2)
        b_bgd /= b_tot[:, np.newaxis]

        a_para = inp[:, 0] * b_bgd[:, 0] + inp[:, 1] * b_bgd[:, 1]
        a_perp = inp[:, 0] * b_bgd[:, 1] - inp[:, 1] * b_bgd[:, 0]
        alpha = np.arctan2(b_bgd[:, 2], b_tot)

    return a_para, a_perp, alpha
