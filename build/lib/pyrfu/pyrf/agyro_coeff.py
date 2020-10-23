#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agyro_coeff.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def agyro_coeff(p_xyz=None):
    """Computes agyrotropy coefficient as in [1]_

    .. math::

        Q = \\frac{P_{12}^2 + P_{13}^2 + P_{23}^2}{P_\\perp^2 + 2 P_\\perp P_\\parallel}


    Parameters
    ----------
    p_xyz : xarray.DataArray
        Time series of the pressure tensor

    Returns
    -------
    sqrt_q : xarray.DataArray
        Time series of the agyrotropy coefficient of the specie

    References
    ----------
    .. [1]  Swisdak, M. (2016), Quantifying gyrotropy in magnetic reconnection, Geophys. Res.
            Lett., 43, 43–49, doi: https://doi.org/10.1002/2015GL066980.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
    >>> # Spacecraft index
    >>> ic = 1
    >>> # Load magnetic field and electron pressure tensor
    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
    >>> p_xyz_e = mms.get_data("Pe_gse_fpi_fast_l2",Tint,1)
    >>> # Rotate electron pressure tensor to field aligned coordinates
    >>> p_xyzfac_e = mms.rotate_tensor(p_xyz_e,"fac",b_xyz,"pp")
    >>> # Compute agyrotropy coefficient
    >>> sqrt_q_e = pyrf.agyro_coeff(p_xyzfac_e)

    """

    if p_xyz is None:
        raise ValueError("agyro_coeff requires at least one argument")

    if not isinstance(p_xyz, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    if p_xyz.ndim != 3:
        raise TypeError("Input must be a second order tensor")

    # Parallel and perpendicular components
    p_para, p_perp = [p_xyz[:, 0, 0], (p_xyz[:, 1, 1] + p_xyz[:, 2, 2]) / 2]

    # Off-diagonal terms
    p_12, p_13, p_23 = [p_xyz[:, 0, 1], p_xyz[:, 0, 2], p_xyz[:, 1, 2]]

    sqrt_q = np.sqrt((p_12 ** 2 + p_13 ** 2 + p_23 ** 2) / (p_perp ** 2 + 2 * p_perp * p_para))

    return sqrt_q
