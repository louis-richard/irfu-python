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

"""st_diff.py
@author: Louis Richard
"""

import numpy as np

from .c_4_grad import c_4_grad
from .gradient import gradient
from .avg_4sc import avg_4sc
from .ts_vec_xyz import ts_vec_xyz


def st_diff(r_mms, b_mms, lmn):
    """Computes velocity of the structure using spatio-temporal
    derivative method [13]_ [14]_ as

    .. math::
        \\mathbf{V}_{str}^{\\mathbf{LMN}} = -\\textrm{d}_t
        \\mathbf{B}^{\\mathbf{LMN}}\\left [\\nabla^{\\mathbf{LMN}}
        \\mathbf{B}^{\\mathbf{LMN}}\\right]^T \\left [
        \\mathbf{S}^{\\mathbf{LMN}}\\right ]^{-1}

    where :math:`\\mathbf{B}^{\\mathbf{LMN}}`, :math:`\\nabla^{\\mathbf{
    LMN}}\\mathbf{B}^{\\mathbf{LMN}}`, :math:`\\mathbf{S}^{\\mathbf{LMN}}`
    and :math:`\\mathbf{V}_{str}^{\\mathbf{LMN}}` are namely the magnetic
    field, its gradient, its rotation rate tensor and the velocity of the
    structure in the LMN coordinates system.

    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Spacecraft positions.

    b_mms : list of xarray.DataArray
        Background magnetic field.

    lmn : ndarray
        Structure coordinates system.

    Returns
    -------
    v_str : xarray.DataArray
        Velocity of the structure in its coordinates system.

    References
    ----------
    .. [13] Shi, Q. Q., Shen, C., Pu, Z. Y., Dunlop, M. W., Zong, Q. G.,
            Zhang, H., et al. (2005), Dimensional analysis of observed
            structures using multipoint magnetic field measurements:
            Application to Cluster. Geophysical Research Letters, 32,
            L12105. doi : https://doi.org/10.1029/2005GL022454.

    .. [14] Shi, Q. Q., Shen, C., Dunlop, M. W., Pu, Z. Y., Zong, Q. G.,
            Liu, Z. X., et al. (2006), Motion of observed structures
            calculated from multi‐point magnetic field measurements:
            Application to Cluster. Geophysical Research Letters, 33,
            L08109. doi : https://doi.org/10.1029/2005GL025073.
    """

    # Compute magnetic field at the center of mass of the tetrahedron
    b_xyz = avg_4sc(b_mms)

    # Gradient of the magnetic field
    grad_b = c_4_grad(r_mms, b_mms)

    # Time derivative of the magnetic field at the center of mass of the
    # tetrahedron
    db_dt = gradient(b_xyz)

    # Transform gradient to LMN frame
    l_grad_b = np.matmul(grad_b.data, lmn[:, 0])
    m_grad_b = np.matmul(grad_b.data, lmn[:, 1])
    n_grad_b = np.matmul(grad_b.data, lmn[:, 2])

    # Compute velocity of the structure using MDD
    v_str = np.zeros(db_dt.shape)
    v_str[:, 0] = np.sum(db_dt * l_grad_b, axis=1)
    v_str[:, 0] /= np.linalg.norm(l_grad_b, axis=1) ** 2
    v_str[:, 1] = np.sum(db_dt * m_grad_b, axis=1)
    v_str[:, 1] /= np.linalg.norm(m_grad_b, axis=1) ** 2
    v_str[:, 2] = np.sum(db_dt * n_grad_b, axis=1)
    v_str[:, 2] /= np.linalg.norm(n_grad_b, axis=1) ** 2

    # To time series
    v_str_xyz = ts_vec_xyz(b_xyz.time.data, -v_str)

    return v_str_xyz
