#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import constants

# Local imports
from .resample import resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def pres_anis(p_fac, b_xyz):
    r"""Compute pressure anisotropy factor:

    .. math::

        \mu_0 \frac{P_\parallel - P_\perp}{|\mathbf{B}|^2}

    Parameters
    ----------
    p_fac : xarray.DataArray
        Time series of the pressure tensor in field aligne coordinates.
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

    Transform pressure tensor to field aligned coordinates
    >>> p_fac_i = mms.rotate_tensor(p_xyz_i, "fac", b_xyz)

    Compute pressure anistropy

    >>> p_anis = pyrf.pres_anis(p_xyz_i, b_xyz)

    """

    b_xyz = resample(b_xyz, p_fac)

    # Get parallel and perpendicular pressure
    p_para = p_fac[:, 0, 0]
    p_perp = (p_fac[:, 1, 1] + p_fac[:, 2, 2]) / 2

    # Load permittivity
    mu0 = constants.mu_0

    # Compute pressure anistropy
    p_anis = 1e9 * mu0 * (p_para - p_perp) / np.linalg.norm(b_xyz) ** 2

    return p_anis
