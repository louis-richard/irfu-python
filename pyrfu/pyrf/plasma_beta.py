#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from ..mms import rotate_tensor
from .resample import resample
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def plasma_beta(b_xyz, p_xyz):
    """Computes plasma beta at magnetic field sampling

    .. math::

        \beta = \frac{P_{th}}{P_b}

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
    p_tot = (p_fac.data[:, 0, 0] + p_fac.data[:, 1, 1] + p_fac.data[:, 2, 2]) / 3

    # Magnitude of the magnetic field
    b_mag = np.linalg.norm(b_xyz.data, axis=1)
    p_mag = 1e-18 * b_mag**2 / (2 * constants.mu_0)

    # Compute plasma beta
    beta = ts_scalar(b_xyz.time.data, p_tot / p_mag)

    return beta
