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

"""calc_dng.py
@author: Louis Richard
"""

import numpy as np


def calc_dng(p_xyz):
    """Computes agyrotropy coefficient as in [15]_

    .. math::

        D_{ng} = \\frac{\\sqrt{8 (P_{12}^2 + P_{13}^2 + P_{23}^2)}}
        {P_\\parallel + 2 P_\\perp}


    Parameters
    ----------
    p_xyz : xarray.DataArray
        Time series of the pressure tensor

    Returns
    -------
    d_ng : xarray.DataArray
        Time series of the agyrotropy coefficient of the specie.

    References
    ----------
    .. [15] Aunai, N., M. Hesse, and M. Kuznetsova (2013), Electron
            nongyrotropy in the context of collisionless magnetic
            reconnection, Phys. Plasmas, 20(6), 092903,
            doi: https://doi.org/10.1063/1.4820953.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> ic = 1

    Load magnetic field and electron pressure tensor

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> p_xyz_e = mms.get_data("pe_gse_fpi_fast_l2", tint, 1)

    Rotate electron pressure tensor to field aligned coordinates

    >>> p_fac_e_pp = mms.rotate_tensor(p_xyz_e, "fac", b_xyz, "pp")

    Compute agyrotropy coefficient

    >>> d_ng_e = pyrf.calc_dng(p_fac_e_pp)

    """

    # Parallel and perpendicular components
    p_para, p_perp = [p_xyz[:, 0, 0], (p_xyz[:, 1, 1] + p_xyz[:, 2, 2]) / 2]

    # Off-diagonal terms
    p_12, p_13, p_23 = [p_xyz[:, 0, 1], p_xyz[:, 0, 2], p_xyz[:, 1, 2]]

    d_ng = np.sqrt(8 * (p_12 ** 2 + p_13 ** 2 + p_23 ** 2))
    d_ng /= (p_para + 2 * p_perp)

    return d_ng
