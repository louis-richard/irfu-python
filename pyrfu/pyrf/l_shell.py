#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from pyrfu._core.units import _check_units

# Local imports
from ..constants import R_E
from .cotrans import cotrans
from .ts_scalar import ts_scalar


def l_shell(r_xyz):
    r"""Compute spacecraft position L Shell for a dipole magnetic field
    according to IGRF.

    .. math::

        L = \frac{r}{R_E \cos^2{\lambda}}

    where :math:`r` is the radial distance, :math:`\lambda` the magnetic
    latitude in the solar magnetic (SM) system, which uses the IGRF dipole
    axis, and :math:`R_E` = 6371.2 km the IGRF reference radius
    (:data:`pyrfu.constants.R_E`).

    Parameters
    ----------
    r_xyz : xarray.DataArray
        Time series of the spacecraft position [km] (a warning for other
        ``UNITS``). Must have a "COORDINATE_SYSTEM" attribute.

    Returns
    -------
    out : xarray.DataArray
        Time series of the spacecraft position L-Shell [R_E].

    """

    # Positions in R_E gave L = r / (6371.2 R_E), with no message
    _check_units(r_xyz, "km", "r_xyz")

    # Transform spacecraft coordinates to solar magnetic system
    r_sm = cotrans(r_xyz, "sm").data

    # Compute Geomagnetic latitude
    lambda_ = np.arctan2(r_sm[:, 2], np.linalg.norm(r_sm[:, :2], axis=1))

    # Compute L shell
    l_sh = np.linalg.norm(r_sm, axis=1) / (R_E * np.cos(lambda_) ** 2)

    out = ts_scalar(r_xyz.time.data, l_sh, {"UNITS": "R_E"})

    return out
