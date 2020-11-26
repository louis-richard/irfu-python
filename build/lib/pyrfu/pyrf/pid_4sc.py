#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pid_4sc.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from ..mms import rotate_tensor

from .c_4_grad import c_4_grad
from .avg_4sc import avg_4sc
from .trace import trace
from .ts_tensor_xyz import ts_tensor_xyz
from .ts_scalar import ts_scalar


def pid_4sc(r_mms=None, v_mms=None, p_mms=None, b_mms=None):
    """
    Compute Pi-D term using definition of [10]_ as :

    .. math::

        \\textrm{Pi}-\\textrm{D} = - \\mathbf{\\Pi}_{ij}\\mathbf{D}_{ij}

    with :math:`\\mathbf{\\Pi}_{ij}` the deviatoric part of the pressure tensor :

    .. math::

        \\mathbf{\\Pi}_{ij} = \\mathbf{P}_{ij} - p\\delta_{ij}

        p = \\frac{1}{3}\\mathbf{P}_{ii}


    and :math:`\\mathbf{D}_{ij}` the deviatoric part of the strain tensor :

    .. math::

        \\mathbf{D}_{ij} =
        \\frac{1}{2}\\left ( \\partial_i \\mathbf{u}_j + \\partial_j \\mathbf{u}_i\\right )
        - \\frac{1}{3}\\theta\\delta_{ij}

        \\theta = \\nabla . \\mathbf{u}


    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Time series of the position of the 4 spacecraft.

    v_mms : list of xarray.DataArray
        Time series of the bulk velocities of the 4 spacecraft.

    p_mms : list of xarray.DataArray
        Time series of the pressure tensor of the 4 spacecraft.

    b_mms : list of xarray.DataArray
        Time series of the background magnetic field of the 4 spacecraft.

    Returns
    -------
    pid : xarray.DataArray
        Time series of the Pi-D.

    References
    ----------
    .. [10]  Yang, Y., Matthaeus, W. H., Parashar, T. N., Wu, P., Wan, M., Shi, Y.,
            et al. (2017). Energy transfer channels and turbulence cascade in Vlasov-Maxwell
            turbulence. Physical Review E, 95, 061201.
            doi : https://doi.org/10.1103/PhysRevE.95.061201

    """

    # Check input
    assert r_mms is not None and isinstance(r_mms, list) and isinstance(r_mms[0], xr.DataArray)
    assert v_mms is not None and isinstance(v_mms, list) and isinstance(v_mms[0], xr.DataArray)
    assert p_mms is not None and isinstance(p_mms, list) and isinstance(p_mms[0], xr.DataArray)
    assert b_mms is not None and isinstance(b_mms, list) and isinstance(b_mms[0], xr.DataArray)

    time = v_mms[0].time.data

    # Compute pressure tensor and background magnetic field at the center of mass of the tetrahedron
    p_xyz = avg_4sc(p_mms)
    b_xyz = avg_4sc(b_mms)

    # Compute divergence and gradient tensor of the bulk velocity. Yang's notation
    theta, grad_u = [c_4_grad(r_mms, v_mms, method) for method in ["div", "grad"]]

    # Define identity tensor
    identity_3d = np.zeros((len(grad_u), 3, 3))
    np.einsum("jii->ji", identity_3d)[:] = 1

    # strain tensor
    eps_xyz = (grad_u.data + np.transpose(grad_u.data, [0, 2, 1])) / 2

    # Deviatoric part of the strain tensor
    d_xyz = eps_xyz - theta.data[:, np.newaxis, np.newaxis] * identity_3d / 2
    d_xyz = ts_tensor_xyz(time, d_xyz)

    # Convert tensors to field aligned coordinates
    d_fac, p_fac = [rotate_tensor(tensor, "fac", b_xyz, "pp") for tensor in [d_xyz, p_xyz]]

    # Isotropic scalar pressure
    p = trace(p_fac) / 3

    # Deviatoric part of the pressure tensor
    pi_fac = p_fac.data - p.data[:, np.newaxis, np.newaxis] * identity_3d
    pi_fac = ts_tensor_xyz(time, pi_fac)

    # Compute Pi-D
    pid = np.sum(np.sum(pi_fac.data * d_fac.data, axis=1), axis=1)
    pid = ts_scalar(time, pid)

    # Flatten tensors
    idx_flat = [0, 4, -1, 1, 2, 5]
    d_flat, pi_flat = [np.reshape(tensor.data, [len(tensor), 9]) for tensor in [d_fac, pi_fac]]
    d_flat, pi_flat = [tensor[:, idx_flat] for tensor in [d_flat, pi_flat]]
    # Compute components of the double contraction sum
    d_coeff = d_flat * pi_flat

    times = p_xyz.time.data
    index = np.arange(1, 7)
    d_coeff = xr.DataArray(d_coeff, coords=[times, index], dims=["time", "index"])

    return pid, d_coeff