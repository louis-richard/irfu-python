# -*- coding: utf-8 -*-
"""
agyro_coeff.py
@author : Louis RICHARD
"""

import xarray as xr


def agyro_coeff(p=None):
    """
    Computes agyrotropy coefficient (Swidak2016 https://doi.org/10.1002/2015GL066980)

    Parameters :
        P : DataArray
            Time series of the pressure tensor

    Returns :
        Q : DataArray
            Time series of the agyrotropy coefficient of the specie

    Example :
        >>> from pyrfu import mms, pyrf
        >>> # Time interval
        >>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
        >>> # Spacecraft index
        >>> ic = 1
        >>> # Load magnetic field and electron pressure tensor
        >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
        >>> p_xyz_e = mms.get_data("Pe_gse_fpi_fast_l2",Tint,1)
        >>> # Rotate electron pressure tensor to field aligned coordinates
        >>> p_xyzfac_e = pyrf.rotate_tensor(p_xyz_e,"fac",b_xyz,"pp")
        >>> # Compute agyrotropy coefficient
        >>> q_e = pyrf.agyro_coeff(p_xyzfac_e)

    """

    if p is None:
        raise ValueError("agyro_coeff requires at least one argument")

    if not isinstance(p, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    if p.ndim != 3:
        raise TypeError("Input must be a second order tensor")

    # Parallel and perpandicular components
    p_para, p_perp = [p[:, 0, 0], (p[:, 1, 1] + p[:, 2, 2]) / 2]

    # Off-diagonal terms
    p_12, p_13, p_23 = [p[:, 0, 1], p[:, 0, 2], p[:, 1, 2]]

    q = (p_12 ** 2 + p_13 ** 2 + p_23 ** 2) / (p_perp ** 2 + 2 * p_perp * p_para)

    return q
