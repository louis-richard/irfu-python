#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from .avg_4sc import avg_4sc

# Local imports
from .c_4_grad import c_4_grad
from .gradient import gradient
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def st_diff(r_mms, b_mms, lmn):
    r"""Computes velocity of the structure using spatio-temporal
    derivative method [13]_ [14]_ as

    .. math::

        V_k = - \frac{\partial_t \mathbf{B} \cdot \partial_k
        \mathbf{B}}{\left | \partial_k \mathbf{B} \right |^2},
        \quad \partial_k \mathbf{B} = (\mathbf{k} \cdot \nabla)
        \mathbf{B}, \quad \mathbf{k} = \mathbf{l}, \mathbf{m},
        \mathbf{n}

    with :math:`\partial_t \mathbf{B}` the time derivative of the field at
    the barycentre and :math:`\nabla \mathbf{B}` its gradient
    (``c_4_grad``). This is the solution of :math:`\partial_t \mathbf{B} +
    (\mathbf{V} \cdot \nabla) \mathbf{B} = 0` when the derivatives along
    l, m and n are orthogonal, i.e., in the frame of the minimum directional
    derivative (MDD) [13]_, and only along the directions with a significant
    gradient (the dimensions of the structure).

    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Spacecraft positions.
    b_mms : list of xarray.DataArray
        Background magnetic field.
    lmn : ndarray
        Structure coordinates system: the l, m and n unit vectors as the
        columns, e.g., the eigenvectors of the average of grad_b^T grad_b
        (MDD), with grad_b = c_4_grad(r_mms, b_mms).data at [time, j, i] =
        d B_j / d x_i.

    Returns
    -------
    v_str : xarray.DataArray
        Velocity of the structure along l, m and n (``COORDINATE_SYSTEM``
        "lmn"), in the unit of the positions per second (km/s for MMS).
        Meaningless (large or NaN) along a direction without gradient.

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

    # To time series, in the LMN coordinates system
    v_str_xyz = ts_vec_xyz(b_xyz.time.data, -v_str, {"COORDINATE_SYSTEM": "lmn"})

    return v_str_xyz
