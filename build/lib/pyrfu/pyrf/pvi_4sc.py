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

"""pvi_4sc.py
@author: Louis Richard
"""

import numpy as np

from .resample import resample
from .norm import norm
from .dot import dot


def pvi_4sc(b_mms):
    """Compute the Partial Variance of Increments (PVI) using the
    definition in [12]_ as

    .. math::

            PVI_{ij}(t) = \\sqrt{\\frac{|\\Delta \\mathbf{B}_{ij}(t)|^2}
            {\\langle|\\Delta\\mathbf{B}_{ij}|^2\\rangle}}

    where :math:`\\Delta \\mathbf{B}_{ij}(t) = \\mathbf{B}_i(t)
    - \\mathbf{B}_i(t)` is the magnetic field increments, the average
    :math:`\\langle . \\rangle` is taken over the whole interval,
    and :math:`i`, :math:`j` = 1,2,3,4 is the MMS spacecraft number.

    In addition, computes, the rotation of the magnetic field between
    two spacecraft, i.e., magnetic field shear angle, as :

    .. math::

        \\alpha_{ij}(t) = cos^{-1} \\frac{\\mathbf{B}_i(t).
        \\mathbf{B}_j(t)}{|\\mathbf{B}_i(t)| |\\mathbf{B}_j(t)|}


    Parameters
    ----------
    b_mms : list of xarray.DataArray
        List of the time series of the background magnetic field for
        the 4 spacecraft.

    Returns
    -------
    pvi_ij : xarray.DataArray
        Time series of the Partial Variance of Increments for the 6
        pairs of spacecraft.

    alpha_ij : xarray.DataArray
        Time series of the magnetic field shear angle for the 6 pairs
        of spacecraft.

    References
    ----------
    .. [12] Chasapis, A., Retinó, A., Sahraoui, F., Vaivads, A.,
            Khotyaintsev, Yu. V., Sundkvist, D., et al. (2015)
            Thin current sheets and associated electron heating in
            turbulent space plasma. Astrophys. J. Lett. 804:L1.
            doi: https://doi.org/10.1088/2041-8205/804/1/L1


    """

    b_mms = [resample(b_xyz, b_mms[0]) for b_xyz in b_mms]

    i_indices = [0, 0, 0, 1, 1, 2]
    j_indices = [1, 2, 3, 2, 3, 3]

    # Compute normalized partial variance of increments
    def pvi(b_i, b_j):
        return np.sqrt(norm(b_i - b_j) ** 2 / np.mean(norm(b_i - b_j) ** 2))

    pvi_ij = [pvi(b_mms[i], b_mms[j]) for i, j in zip(i_indices, j_indices)]

    # Compute magnetic field shear angle
    def shear_angle(b_i, b_j):
        return np.arccos(dot(b_i, b_j) / (norm(b_i) * norm(b_j)))

    theta_ij = []
    for i, j in zip(i_indices, j_indices):
        theta_ij.append(shear_angle(b_mms[i], b_mms[j]))

    return pvi_ij, theta_ij
