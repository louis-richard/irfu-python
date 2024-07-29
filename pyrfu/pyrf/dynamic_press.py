#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def dynamic_press(
    n_s: DataArray, v_xyz: DataArray, specie: Optional[str] = "ions"
) -> DataArray:
    r"""Computes dynamic pressure.

    Parameters
    ----------
    n_s : DataArray
        Time series of the number density of the specie.
    v_xyz : DataArray
        Time series of the bulk velocity of the specie.
    specie : str, Optional
        Specie 'ions' or 'electrons'. Default 'ions'.

    Returns
    -------
    DataArray
        Time series of the dynamic pressure of the specie.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load ion bulk velocity and remove spintone

    >>> v_xyz_i = mms.get_data("vi_gse_fpi_fast_l2", tint, mms_id)
    >>> st_xyz_i = mms.get_data("sti_gse_fpi_fast_l2", tint, mms_id)
    >>> v_xyz_i = v_xyz_i - st_xyz_i

    Ion number density

    >>> n_i = mms.get_data("ni_fpi_fast_l2", tint, mms_id)

    Compute dynamic pressure

    >>> p = pyrf.dynamic_press(n_i, v_xyz_i, specie="ions")

    """

    # Check input
    if not isinstance(n_s, xr.DataArray):
        raise TypeError("n_s must be a xarray.DataArray")

    if not isinstance(v_xyz, xr.DataArray):
        raise TypeError("v_xyz must be a xarray.DataArray")

    if not isinstance(specie, str):
        raise TypeError("specie must be a string")

    # Check n_s and v_xyz shapes
    if n_s.ndim != 1:
        raise ValueError("n_s must be a scalar")

    if v_xyz.ndim != 2 or v_xyz.shape[1] != 3:
        raise ValueError("v_xyz must be a vector")

    # Check specie
    if specie.lower() not in ["ions", "electrons"]:
        raise ValueError("specie must be 'ions' or 'electrons'")

    if specie.lower() == "ions":
        mass = constants.proton_mass
    else:
        mass = constants.electron_mass

    # Get data
    n_s_data: np.ndarray = n_s.data
    v_xyz_data: np.ndarray = v_xyz.data

    # Compute dynamic pressure
    p_dyn: np.ndarray = n_s_data * mass * np.linalg.norm(v_xyz_data, axis=1) ** 2
    p_dyn_ts: DataArray = ts_scalar(n_s.time.data, p_dyn)

    return p_dyn_ts
