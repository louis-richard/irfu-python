#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dynamic_press(n_s, v_xyz, specie: str = "ions"):
    r"""Computes dynamic pressure.

    Parameters
    ----------
    n_s : xarray.DataArray
        Time series of the number density of the specie.
    v_xyz : xarray.DataArray
        Time series of the bulk velocity of the specie.
    specie : {"ions", "electrons"}, Optional
        Specie. Default "ions".

    Returns
    -------
    p_dyn : xarray.DataArray
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
    assert isinstance(n_s, xr.DataArray), "n_s must be a xarray.DataArray"
    assert isinstance(v_xyz, xr.DataArray), "v_xyz must be a xarray.DataArray"
    assert isinstance(specie, str), "specie must be a str"
    assert specie.lower() in ["ions", "electrons"], "specie must be ions or electrons"

    # Check n_s and v_xyz shapes
    assert n_s.ndim == 1, "n_s must be a scalar"
    assert v_xyz.ndim == 2 and v_xyz.shape[1] == 3, "v_xyz must be a vector"

    if specie.lower() == "ions":
        mass = constants.proton_mass
    else:
        mass = constants.electron_mass

    p_dyn = n_s * mass * np.linalg.norm(v_xyz, axis=1) ** 2

    return p_dyn
