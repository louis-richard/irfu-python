#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from .dot import dot
from .norm import norm

# Local imports
from .resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def pvi_4sc(b_mms):
    r"""Compute the Partial Variance of Increments (PVI) using the
    definition in [1]_ as

    .. math::

            PVI_{ij}(t) = \sqrt{\frac{|\Delta B_{ij}(t)|^2}
            {\langle|\Delta B_{ij}|^2\rangle}}

    where :math:`\Delta B_{ij}(t) = B_i(t) - B_j(t)` is the magnetic field
    increments, the average :math:`\langle . \rangle` is taken over the
    whole interval, and :math:`i`, :math:`j` = 1,2,3,4 is the MMS spacecraft
    number.

    In addition, computes, the rotation of the magnetic field between
    two spacecraft, i.e., magnetic field shear angle, as :

    .. math::

        \alpha_{ij}(t) = cos^{-1} \frac{B_i(t) . B_j(t)}{|B_i(t)| |B_j(t)|}


    Parameters
    ----------
    b_mms : list of xarray.DataArray
        List of the time series of the background magnetic field for
        the 4 spacecraft.

    Returns
    -------
    pvi_ij : list of xarray.DataArray
        Time series of the Partial Variance of Increments for the 6
        pairs of spacecraft (12, 13, 14, 23, 24, 34).
    alpha_ij : list of xarray.DataArray
        Time series of the magnetic field shear angle, in radians, for the
        6 pairs of spacecraft.

    References
    ----------
    .. [1] Chasapis, A., Retinó, A., Sahraoui, F., Vaivads, A.,
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
        # Clipped to [-1, 1]: rounding gave cosines just above 1 for parallel
        # fields, and arccos NaN instead of 0
        cos_ij = dot(b_i, b_j) / (norm(b_i) * norm(b_j))
        return np.arccos(cos_ij.clip(-1.0, 1.0))

    theta_ij = []
    for i, j in zip(i_indices, j_indices):
        theta_ij.append(shear_angle(b_mms[i], b_mms[j]))

    # PVI dimensionless, shear angles in rad (they kept the units of b)
    for pvi_, theta_ in zip(pvi_ij, theta_ij):
        pvi_.attrs["UNITS"], theta_.attrs["UNITS"] = "", "rad"

    return pvi_ij, theta_ij
