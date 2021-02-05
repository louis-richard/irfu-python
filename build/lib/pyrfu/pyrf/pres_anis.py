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

"""pres_anis.py
@author: Louis Richard
"""

import numpy as np

from scipy import constants

from .resample import resample
from ..mms import rotate_tensor


def pres_anis(p_xyz, b_xyz):
    """
    Compute pressure anisotropy factor:

    .. math::

        \\mu_0 \\frac{P_\\parallel - P_\\perp}{|\\mathbf{B}|^2}

    Parameters
    ----------
    p_xyz : xarray.DataArray
        Time series of the pressure tensor.

    b_xyz : xarray.DataArray
        Time series of the background magnetic field.

    Returns
    -------
    p_anis : xarray.DataArray
        Time series of the pressure anisotropy.

    See also
    --------
    pyrfu.mms.rotate_tensor : Rotates pressure or temperature tensor
                                into another coordinate system.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field, ion/electron temperature and number density

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> p_xyz_i = mms.get_data("Pi_gse_fpi_fast_l2", tint, mms_id)

    Compute pressure anistropy

    >>> p_anis = pyrf.pres_anis(p_xyz_i, b_xyz)

    """

    b_xyz = resample(b_xyz, p_xyz)

    # rotate pressure tensor to field aligned coordinates
    p_xyzfac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Get parallel and perpendicular pressure
    p_para = p_xyzfac[:, 0, 0]
    p_perp = (p_xyzfac[:, 1, 1] + p_xyzfac[:, 2, 2]) / 2

    # Load permittivity
    mu0 = constants.mu_0

    # Compute pressure anistropy
    p_anis = 1e9 * mu0 * (p_para - p_perp) / np.linalg.norm(b_xyz) ** 2

    return p_anis
