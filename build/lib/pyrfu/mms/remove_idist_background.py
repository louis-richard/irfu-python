#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import constants

# Local imports
from ..pyrf import ts_tensor_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def remove_idist_background(n_i, v_gse_i, p_gse_i, n_bg_i, p_bg_i):
    r"""Removes penetrating radiation background from ion moments.

    Parameters
    ----------
    n_i : xarray.DataArray
        Time series of the ion density.
    v_gse_i : xarray.DataArray
        Time series of the ion bulk velocity.
    p_gse_i : xarray.DataArray
        Time series of the ion pressure tensor.
    n_bg_i : xarray.DataArray
        Time series of the background ion number density.
    p_bg_i : xarray.DataArray
        Time series of the background ion pressure scalar.

    Returns
    -------
    n_i_new : xarray.DataArray
        Time series of the corrected ion number density.
    v_gse_i_new : xarray.DataArray
        Time series of the corrected ion bulk velocity.
    p_gse_i : xarray.DataArray
        Time series of the corrected ion pressure tensor.

    References
    ----------
    MMS DIS Penetrating radiation correction methods.

    """

    m_p = constants.proton_mass

    # Number density
    n_i_new = n_i - n_bg_i.data

    # Bulk velocity
    v_gse_i_new = v_gse_i.copy()
    v_gse_i_new.data *= n_i.data[:, None] / n_i_new.data[:, None]

    # Pressure tensor
    p_gse_i_new = np.zeros(p_gse_i.shape)
    n_old, v_old = [n_i.data, v_gse_i.data]
    n_new, v_new = [n_i_new.data, v_gse_i_new.data]

    for i, j in zip([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]):
        p_gse_i_new[:, i, j] += p_gse_i.data[:, 0, 0]
        p_gse_i_new[:, i, j] -= p_bg_i.data
        p_gse_i_new[:, i, j] += m_p * n_old * np.multiply(v_old[:, i],
                                                          v_old[:, j])
        p_gse_i_new[:, i, j] -= m_p * n_new * np.multiply(v_new[:, i],
                                                          v_new[:, j])

    # Pressure tensor is symmetric
    p_gse_i_new[:, 1, 0] = p_gse_i_new[:, 0, 1]
    p_gse_i_new[:, 2, 0] = p_gse_i_new[:, 0, 2]
    p_gse_i_new[:, 2, 1] = p_gse_i_new[:, 1, 2]

    p_gse_i_new = ts_tensor_xyz(p_gse_i.time.data, p_gse_i_new)

    return n_i_new, v_gse_i_new, p_gse_i_new
