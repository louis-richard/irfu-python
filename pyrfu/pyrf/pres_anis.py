#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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

    # Get parallel and perpendicular pressure
    p_para = p_fac.data[:, 0, 0]
    p_perp = (p_fac.data[:, 1, 1] + p_fac.data[:, 2, 2]) / 2

    # Compute magnetic pressure
    b_xyz = resample(b_xyz, p_fac)
    b_mag = np.linalg.norm(b_xyz.data, axis=1)
    p_mag = 1e-18 * b_mag**2 / (2 * constants.mu_0)

    # Compute pressure anistropy
    p_anis = (1e-9 * (p_para - p_perp) / 2) / p_mag
    p_anis = ts_scalar(p_fac.time.data, p_anis)

    return p_anis
