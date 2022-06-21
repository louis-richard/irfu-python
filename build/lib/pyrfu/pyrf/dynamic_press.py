#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def dynamic_press(n_s, v_xyz, specie: str = "i"):
    r"""Computes dynamic pressure.

    Parameters
    ----------
    n_s : xarray.DataArray
        Time series of the number density of the specie.
    v_xyz : xarray.DataArray
        Time series of the bulk velocity of the specie.
    specie : {"i", "e"}, Optional
        Specie. default "i".

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

    >>> v_xyz_i = mms.get_data("Vi_gse_fpi_fast_l2", tint, mms_id)
    >>> st_xyz_i = mms.get_data("STi_gse_fpi_fast_l2", tint, mms_id)
    >>> v_xyz_i = v_xyz_i - st_xyz_i

    Ion number density

    >>> n_i = mms.get_data("Ni_fpi_fast_l2", tint, mms_id)

    Compute dynamic pressure

    >>> p = pyrf.dynamic_press(n_i, v_xyz_i, specie="i")

    """

    if specie == "i":
        mass = constants.proton_mass
    elif specie == "e":
        mass = constants.electron_mass
    else:
        raise ValueError("Unknown specie")

    p_dyn = n_s * mass * np.linalg.norm(v_xyz, axis=0) ** 2

    return p_dyn
