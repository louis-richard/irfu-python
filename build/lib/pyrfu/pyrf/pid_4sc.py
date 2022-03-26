#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..mms import rotate_tensor

from .c_4_grad import c_4_grad
from .avg_4sc import avg_4sc
from .trace import trace
from .ts_tensor_xyz import ts_tensor_xyz
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def pid_4sc(r_mms, v_mms, p_mms, b_mms):
    r"""Compute Pi-D term using definition of [1]_ as :

    .. math::

        Pi-D = - \Pi_{ij}D_{ij}

    with :math:`\Pi_{ij}` the deviatoric part of the pressure
    tensor :

    .. math::

        \Pi_{ij} = P_{ij} - p\delta_{ij}

        p = \frac{1}{3}P_{ii}


    and :math:`D_{ij}` the deviatoric part of the strain tensor :

    .. math::

        D_{ij} = \frac{1}{2}\left ( \partial_i u_j + \partial_j u_i \right )
        - \frac{1}{3}\theta\delta_{ij}

        \theta = \nabla . u


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
    .. [1]  Yang, Y., Matthaeus, W. H., Parashar, T. N., Wu, P., Wan, M.,
            Shi, Y., et al. (2017). Energy transfer channels and turbulence
            cascade in Vlasov-Maxwell turbulence. Physical Review E, 95,
            061201. doi : https://doi.org/10.1103/PhysRevE.95.061201

    """

    # Compute pressure tensor and background magnetic field at the center of
    # mass of the tetrahedron
    p_xyz = avg_4sc(p_mms)
    b_xyz = avg_4sc(b_mms)

    # Compute divergence and gradient tensor of the bulk velocity. Yang's
    # notation
    theta = c_4_grad(r_mms, v_mms, "div")
    grad_u = c_4_grad(r_mms, v_mms, "grad")

    # Define identity tensor
    identity_3d = np.zeros((len(grad_u), 3, 3))
    np.einsum("jii->ji", identity_3d)[:] = 1

    # strain tensor
    eps_xyz = (grad_u.data + np.transpose(grad_u.data, [0, 2, 1])) / 2

    # Deviatoric part of the strain tensor
    d_xyz = eps_xyz - theta.data[:, np.newaxis, np.newaxis] * identity_3d / 2
    d_xyz = ts_tensor_xyz(v_mms[0].time.data, d_xyz)

    # Convert tensors to field aligned coordinates
    d_fac = rotate_tensor(d_xyz, "fac", b_xyz, "pp")
    p_fac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Isotropic scalar pressure
    press = trace(p_fac) / 3

    # Deviatoric part of the pressure tensor
    pi_fac = p_fac.data - press.data[:, np.newaxis, np.newaxis] * identity_3d
    pi_fac = ts_tensor_xyz(v_mms[0].time.data, pi_fac)

    # Compute Pi-D
    pid = np.sum(np.sum(pi_fac.data * d_fac.data, axis=1), axis=1)
    pid = ts_scalar(v_mms[0].time.data, pid)

    # Flatten tensors
    d_flat = np.reshape(d_fac.data, [len(d_fac), 9])
    pi_flat = np.reshape(pi_fac.data, [len(pi_fac), 9])
    d_flat = d_flat[:, [0, 4, -1, 1, 2, 5]]
    pi_flat = pi_flat[:, [0, 4, -1, 1, 2, 5]]

    # Compute components of the double contraction sum
    d_coeff = xr.DataArray(d_flat * pi_flat,
                           coords=[p_xyz.time.data, np.arange(1, 7)],
                           dims=["time", "index"])

    return pid, d_coeff
