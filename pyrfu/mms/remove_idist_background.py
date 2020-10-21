#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remove_idist_background.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from astropy import constants

from ..pyrf.ts_tensor_xyz import ts_tensor_xyz


def remove_idist_background(n_i=None, v_gse_i=None, p_gse_i=None, n_bg_i=None, p_bg_i=None):
    """Removes penetrating radiation background from ion moments.

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
    MMS DIS Penetrating radiation correction methods :
    https://lasp.colorado.edu/galaxy/display/MFDPG/Penetrating+Radiation+in+DIS+Data

    """

    assert n_i is not None and isinstance(n_i, xr.DataArray)
    assert v_gse_i is not None and isinstance(v_gse_i, xr.DataArray)
    assert p_gse_i is not None and isinstance(p_gse_i, xr.DataArray)
    assert n_bg_i is not None and isinstance(n_bg_i, xr.DataArray)
    assert p_bg_i is not None and isinstance(p_bg_i, xr.DataArray)

    mp = constants.m_p.value

    # Number density
    n_i_new = n_i - n_bg_i.data

    # Bulk velocity
    v_gse_i_new = v_gse_i
    v_gse_i_new *= n_i / n_i_new

    # Pressure tensor
    p_gse_i_new_data = np.zeros(p_gse_i.shape)

    # P_xx_i
    p_gse_i_new_data[:, 0, 0] += p_gse_i.data[:, 0, 0]
    p_gse_i_new_data[:, 0, 0] -= p_bg_i.data
    p_gse_i_new_data[:, 0, 0] += mp * n_i.data * v_gse_i.data[:, 0] * v_gse_i.data[:, 0]
    p_gse_i_new_data[:, 0, 0] -= mp * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 0]

    # P_yy_i
    p_gse_i_new_data[:, 1, 1] += p_gse_i.data[:, 1, 1]
    p_gse_i_new_data[:, 1, 1] -= p_bg_i.data
    p_gse_i_new_data[:, 1, 1] += mp * n_i.data * v_gse_i.data[:, 1] * v_gse_i.data[:, 1]
    p_gse_i_new_data[:, 1, 1] -= mp * n_i_new.data * v_gse_i_new.data[:, 1] * v_gse_i_new.data[:, 1]

    # P_zz_i
    p_gse_i_new_data[:, 2, 2] += p_gse_i.data[:, 2, 2]
    p_gse_i_new_data[:, 2, 2] -= p_bg_i.data
    p_gse_i_new_data[:, 2, 2] += mp * n_i.data * v_gse_i.data[:, 2] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 2, 2] -= mp * n_i_new.data * v_gse_i_new.data[:, 2] * v_gse_i_new.data[:, 2]

    # P_xy_i & P_yx_i
    p_gse_i_new_data[:, 0, 1] += p_gse_i.data[:, 0, 1]
    p_gse_i_new_data[:, 0, 1] += mp * n_i.data * v_gse_i.data[:, 0] * v_gse_i.data[:, 1]
    p_gse_i_new_data[:, 0, 1] -= mp * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 1]
    p_gse_i_new_data[:, 1, 0] = p_gse_i_new_data[:, 0, 1]

    # P_xz_i & P_zx_i
    p_gse_i_new_data[:, 0, 2] += p_gse_i.data[:, 0, 2]
    p_gse_i_new_data[:, 0, 2] += mp * n_i.data*v_gse_i.data[:, 0] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 0, 2] -= mp * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 2]
    p_gse_i_new_data[:, 2, 0] = p_gse_i_new_data[:, 0, 2]

    # P_yz_i & P_zy_i
    p_gse_i_new_data[:, 1, 2] += p_gse_i.data[:, 1, 2]
    p_gse_i_new_data[:, 1, 2] += mp * n_i.data * v_gse_i.data[:, 1] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 1, 2] -= mp * n_i_new.data * v_gse_i_new.data[:, 1] * v_gse_i_new.data[:, 2]
    p_gse_i_new_data[:, 2, 1] = p_gse_i_new_data[:, 1, 2]

    p_gse_i_new = ts_tensor_xyz(p_gse_i.time.data, p_gse_i_new_data)

    return n_i_new, v_gse_i_new, p_gse_i_new
