#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from xarray import DataArray

# Local imports
from pyrfu.pyrf.ts_scalar import ts_scalar


def calc_sqrtq(p_xyz: DataArray) -> DataArray:
    r"""Computes the square root of the agyrotropy coefficient Q of [1]_

    .. math::

        \sqrt{Q} = \sqrt{\frac{P_{12}^2 + P_{13}^2 + P_{23}^2}
        {P_\perp^2 + 2 P_\perp P_\parallel}}

    with :math:`P_\parallel = P_{11}` and
    :math:`P_\perp = (P_{22} + P_{33}) / 2`.

    Parameters
    ----------
    p_xyz : DataArray
        Time series of the pressure tensor in field-aligned coordinates, the
        parallel direction first (e.g., mms.rotate_tensor(p, "fac", b, "pp")).

    Returns
    -------
    DataArray
        Time series of sqrt(Q) of the specie, between 0 (gyrotropic) and 1.

    Raises
    ------
    TypeError
        If input is not a xarray.DataArray.
    ValueError
        If input is not a time series of a tensor (n_time, 3, 3).

    References
    ----------
    .. [1]  Swisdak, M. (2016), Quantifying gyrotropy in magnetic
            reconnection, Geophys. Res.Lett., 43, 43–49, doi:
            https://doi.org/10.1002/2015GL066980.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> ic = 1

    Load magnetic field and electron pressure tensor

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> p_xyz_e = mms.get_data("pe_gse_fpi_fast_l2", tint, 1)

    Rotate electron pressure tensor to field aligned coordinates

    >>> p_fac_e_pp = mms.rotate_tensor(p_xyz_e, "fac", b_xyz, "pp")

    Compute agyrotropy coefficient

    >>> sqrt_q_e = pyrf.calc_sqrtq(p_fac_e_pp)

    """

    # Check input type
    if not isinstance(p_xyz, xr.DataArray):
        raise TypeError("p_xyz must be a xarray.DataArray")

    # Check import shape
    if not p_xyz.data.ndim == 3 or p_xyz.shape[1] != 3 or p_xyz.shape[2] != 3:
        raise ValueError("p_xyz must be a time series of a tensor")

    # Parallel and perpendicular components
    p_para: np.ndarray = p_xyz.data[:, 0, 0]
    p_perp: np.ndarray = (p_xyz.data[:, 1, 1] + p_xyz.data[:, 2, 2]) / 2

    # Off-diagonal terms
    p_12: np.ndarray = p_xyz.data[:, 0, 1]
    p_13: np.ndarray = p_xyz.data[:, 0, 2]
    p_23: np.ndarray = p_xyz.data[:, 1, 2]

    sqrt_q: np.ndarray = np.sqrt(p_12**2 + p_13**2 + p_23**2)
    sqrt_q /= np.sqrt(p_perp**2 + 2 * p_perp * p_para)
    # Dimensionless (no UNITS)
    sqrt_q_ts: DataArray = ts_scalar(p_xyz.time.data, sqrt_q, {"UNITS": ""})

    return sqrt_q_ts
