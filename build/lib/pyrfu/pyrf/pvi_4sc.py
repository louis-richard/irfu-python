#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pvi_4sc.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .norm import norm


def pvi_4sc(b_mms):
    """Compute the Partial Variance of Increments (PVI) using the definition in [12]_ as

    .. math::

            PVI_{ij}(t) = \\sqrt{\\frac{|\\Delta \\mathbf{B}_{ij}(t)|^2}{\\langle|\\Delta
            \\mathbf{B}_{ij}|^2\\rangle}}

    where :math:`\\Delta \\mathbf{B}_{ij}(t) = \\mathbf{B}_i(t) - \\mathbf{B}_i(t)` is the
    magnetic field increments, the average :math:`\\langle . \\rangle` is taken over the whole
    interval, and :math:`i`, :math:`j` = 1,2,3,4 is the MMS spacecraft number.

    In addition, computes, the rotation of the magnetic field between two spacecraft, i.e.,
    magnetic field shear angle, as :

    .. math::

        \\alpha_{ij}(t) = cos^{-1} \\frac{\\mathbf{B}_i(t).\\mathbf{B}_j(t)}
        {|\\mathbf{B}_i(t)| |\\mathbf{B}_j(t)|}


    Parameters
    ----------
    b_mms : list of xarray.DataArray
        List of the time series of the background magnetic field for the 4 spacecraft.

    Returns
    -------
    pvi_ij : xarray.DataArray
        Time series of the Partial Variance of Increments for the 6 pairs of spacecraft.

    alpha_ij : xarray.DataArray
        Time series of the magnetic field shear angle for the 6 pairs of spacecraft.

    References
    ----------
    .. [12] Chasapis, A., Retinó, A., Sahraoui, F., Vaivads, A., Khotyaintsev, Yu. V.,
            Sundkvist, D., et al. (2015) Thin current sheets and associated electron heating in
            turbulent space plasma. Astrophys. J. Lett. 804:L1.
            doi: https://doi.org/10.1088/2041-8205/804/1/L1


    """

    assert isinstance(b_mms, list) and isinstance(b_mms[0], xr.DataArray) and len(b_mms) == 4

    b_mms = [resample(b_xyz) for b_xyz in b_mms]

    i_indices = [0, 0, 0, 1, 1, 2]
    j_indices = [1, 2, 3, 2, 3, 3]

    delta_b_ij = [b_mms[i] - b_mms[j] for i, j in zip(i_indices, j_indices)]

    pvi_ij = [np.sqrt(norm(delta_b) ** 2 / np.mean(norm(delta_b) ** 2)) for delta_b in delta_b_ij]

    return pvi_ij
