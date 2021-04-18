#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import numpy as np

from scipy import constants

from ..pyrf import ts_tensor_xyz


def remove_idist_background(n_i, v_gse_i, p_gse_i, n_bg_i, p_bg_i):
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

    m_p = constants.proton_mass

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
    p_gse_i_new_data[:, 0, 0] += m_p * n_i.data * v_gse_i.data[:, 0] * v_gse_i.data[:, 0]
    p_gse_i_new_data[:, 0, 0] -= m_p * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 0]

    # P_yy_i
    p_gse_i_new_data[:, 1, 1] += p_gse_i.data[:, 1, 1]
    p_gse_i_new_data[:, 1, 1] -= p_bg_i.data
    p_gse_i_new_data[:, 1, 1] += m_p * n_i.data * v_gse_i.data[:, 1] * v_gse_i.data[:, 1]
    p_gse_i_new_data[:, 1, 1] -= m_p * n_i_new.data * v_gse_i_new.data[:, 1] * v_gse_i_new.data[:, 1]

    # P_zz_i
    p_gse_i_new_data[:, 2, 2] += p_gse_i.data[:, 2, 2]
    p_gse_i_new_data[:, 2, 2] -= p_bg_i.data
    p_gse_i_new_data[:, 2, 2] += m_p * n_i.data * v_gse_i.data[:, 2] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 2, 2] -= m_p * n_i_new.data * v_gse_i_new.data[:, 2] * v_gse_i_new.data[:, 2]

    # P_xy_i & P_yx_i
    p_gse_i_new_data[:, 0, 1] += p_gse_i.data[:, 0, 1]
    p_gse_i_new_data[:, 0, 1] += m_p * n_i.data * v_gse_i.data[:, 0] * v_gse_i.data[:, 1]
    p_gse_i_new_data[:, 0, 1] -= m_p * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 1]
    p_gse_i_new_data[:, 1, 0] = p_gse_i_new_data[:, 0, 1]

    # P_xz_i & P_zx_i
    p_gse_i_new_data[:, 0, 2] += p_gse_i.data[:, 0, 2]
    p_gse_i_new_data[:, 0, 2] += m_p * n_i.data*v_gse_i.data[:, 0] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 0, 2] -= m_p * n_i_new.data * v_gse_i_new.data[:, 0] * v_gse_i_new.data[:, 2]
    p_gse_i_new_data[:, 2, 0] = p_gse_i_new_data[:, 0, 2]

    # P_yz_i & P_zy_i
    p_gse_i_new_data[:, 1, 2] += p_gse_i.data[:, 1, 2]
    p_gse_i_new_data[:, 1, 2] += m_p * n_i.data * v_gse_i.data[:, 1] * v_gse_i.data[:, 2]
    p_gse_i_new_data[:, 1, 2] -= m_p * n_i_new.data * v_gse_i_new.data[:, 1] * v_gse_i_new.data[:, 2]
    p_gse_i_new_data[:, 2, 1] = p_gse_i_new_data[:, 1, 2]

    p_gse_i_new = ts_tensor_xyz(p_gse_i.time.data, p_gse_i_new_data)

    return n_i_new, v_gse_i_new, p_gse_i_new
