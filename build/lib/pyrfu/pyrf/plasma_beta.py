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

"""plasma_beta.py
@author: Louis Richard
"""

from ..mms import rotate_tensor

from .norm import norm
from .resample import resample
from .ts_scalar import ts_scalar


def plasma_beta(b_xyz, p_xyz):
    """Computes plasma beta at magnetic field sampling

    .. math::

        \\beta = \\frac{P_{th}}{P_b}

    where : :math:`P_b = B^2 / 2 \\mu_0`

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    p_xyz : xarray.DataArray
        Time series of the pressure tensor.

    Returns
    -------
    beta : xarray.DataArray
        Time series of the plasma beta at magnetic field sampling.

    """

    p_xyz = resample(p_xyz, b_xyz)

    p_fac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Scalar temperature
    p_mag = (p_fac[0, 0] + (p_fac[1, 1] + p_fac[2, 2]) / 2) / 2

    # Magnitude of the magnetic field
    b_mag = norm(b_xyz)

    # Compute plasma beta
    beta = p_mag / (b_mag * 1e-5) ** 2

    time = b_xyz.time.data
    beta = ts_scalar(time, beta)

    return beta
