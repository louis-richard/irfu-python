#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def calc_dng(p_xyz: DataArray) -> DataArray:
    r"""Computes Aunai's agyrotropy coefficient.

     Aunai's agyrotropy is [15]_

    .. math::

        D_{ng} = \frac{\sqrt{8 (P_{12}^2 + P_{13}^2 + P_{23}^2)}}
        {P_\parallel + 2 P_\perp}


    Parameters
    ----------
    p_xyz : DataArray
        Time series of the pressure tensor

    Returns
    -------
    DataArray
        Time series of the agyrotropy coefficient of the specie.

    Raises
    ------
    TypeError
        If input is not a xarray.DataArray.
    ValueError
        If input is not a time series of a tensor (n_time, 3, 3).

    References
    ----------
    .. [15] Aunai, N., M. Hesse, and M. Kuznetsova (2013), Electron
            nongyrotropy in the context of collisionless magnetic
            reconnection, Phys. Plasmas, 20(6), 092903,
            doi: https://doi.org/10.1063/1.4820953.

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

    >>> d_ng_e = pyrf.calc_dng(p_fac_e_pp)

    """
    # Check input type
    if not isinstance(p_xyz, xr.DataArray):
        raise TypeError("p_xyz must be a xarray.DataArray")

    # Check input shape
    if p_xyz.data.ndim != 3 or p_xyz.shape[1] != 3 or p_xyz.shape[2] != 3:
        raise ValueError("p_xyz must be a time series of a tensor")

    # Parallel and perpendicular components
    p_para: np.ndarray = p_xyz.data[:, 0, 0]
    p_perp: np.ndarray = (p_xyz.data[:, 1, 1] + p_xyz.data[:, 2, 2]) / 2

    # Off-diagonal terms
    p_12: np.ndarray = p_xyz.data[:, 0, 1]
    p_13: np.ndarray = p_xyz.data[:, 0, 2]
    p_23: np.ndarray = p_xyz.data[:, 1, 2]

    d_ng: np.ndarray = np.sqrt(8 * (p_12**2 + p_13**2 + p_23**2))
    d_ng /= p_para + 2 * p_perp
    d_ng_ts: DataArray = ts_scalar(p_xyz.time.data, d_ng)

    return d_ng_ts
