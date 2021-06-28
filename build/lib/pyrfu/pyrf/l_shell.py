#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .cotrans import cotrans

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def l_shell(r_xyz):
    r"""Compute spacecraft position L Shell for a dipole magnetic field
    according to IRGF.

    Parameters
    ----------
    r_xyz : xarray.DataArray
        Time series of the spacecraft position. Must have a
        "COORDINATES_SYSTEM" attributes.

    Returns
    -------
    out : xarray.DataArray
        Time series of the spacecraft position L-Shell.

    """

    # Transform spacecraft coordinates to solar magnetic system
    r_sm = cotrans(r_xyz, "sm")

    # Compute Geomagnetic latitude
    lambda_ = np.arctan(r_sm[:, 2] / np.linalg.norm(r_sm[:, :2], axis=1))

    # Compute L shell
    out = np.linalg.norm(r_sm, axis=1) / np.cos(lambda_) ** 2

    return out
