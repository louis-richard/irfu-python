#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def plasma_beta(b_xyz, p_xyz):
    r"""Computes plasma beta at magnetic field sampling

    .. math::

        \beta = \frac{P_{th}}{P_b}

    where : :math:`P_{th} = \mathrm{tr}(\mathbf{P}) / 3` and
    :math:`P_b = B^2 / 2 \mu_0`

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field in nT.
    p_xyz : xarray.DataArray
        Time series of the pressure tensor in nPa.

    Returns
    -------
    beta : xarray.DataArray
        Time series of the plasma beta at magnetic field sampling.

    """

    p_xyz = resample(p_xyz, b_xyz)

    # Scalar pressure (the trace is invariant under rotation) in nPa
    p_tot = np.trace(p_xyz.data, axis1=1, axis2=2) / 3

    # Magnetic pressure in nPa (B in nT)
    b_mag = np.linalg.norm(b_xyz.data, axis=1)
    p_mag = 1e-9 * b_mag**2 / (2 * constants.mu_0)

    # Compute plasma beta
    beta = ts_scalar(b_xyz.time.data, p_tot / p_mag)

    return beta
