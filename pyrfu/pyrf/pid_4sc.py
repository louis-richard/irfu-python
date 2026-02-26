#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.pyrf.avg_4sc import avg_4sc
from pyrfu.pyrf.c_4_grad import c_4_grad
from pyrfu.pyrf.c_4_k import c_4_k
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.trace import trace
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_tensor_xyz import ts_tensor_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["pid_4sc", "pid_4sc_err"]


def pid_4sc(r_mms, v_mms, p_mms):
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
    d_xyz = eps_xyz - theta.data[:, np.newaxis, np.newaxis] * identity_3d / 3
    d_xyz = ts_tensor_xyz(v_mms[0].time.data, d_xyz)

    # Isotropic scalar pressure
    press = trace(p_xyz) / 3

    # Deviatoric part of the pressure tensor
    pi_xyz = p_xyz.data - press.data[:, np.newaxis, np.newaxis] * identity_3d
    pi_xyz = ts_tensor_xyz(v_mms[0].time.data, pi_xyz)

    # Compute p\theta
    ptheta = press.data * theta.data
    ptheta = ts_scalar(v_mms[0].time.data, ptheta)

    # Compute Pi-D
    pid = -np.sum(np.sum(pi_xyz.data * d_xyz.data, axis=1), axis=1)
    pid = ts_scalar(v_mms[0].time.data, pid)

    return d_xyz, pi_xyz, ptheta, pid


def pid_4sc_err(r_mms, v_mms, p_mms, errv_mms, errp_mms):
    r"""Compute error propagation for Pi-D term using definition of [1]_ as :

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
    errv_mms : list of xarray.DataArray
        Time series of the error on the bulk velocities of the 4 spacecraft.
    errp_mms : list of xarray.DataArray
        Time series of the error on the pressure tensor of the 4 spacecraft.

    Returns
    -------
    sigma_pid : xarray.DataArray
        Time series of the error on Pi-D.

    References
    ----------
    .. [1]  Roberts, O. W., Vörös, Z., Torkar, K., Stawarz, J.,
            Bandyopadhyay, R., Gershman, D. J., et al. (2023).
            Estimation of the error in the calculation of the
            pressure-strain term: Application in the terrestrial
            magnetosphere. Journal of Geophysical Research: Space Physics,
            128, e2023JA031565. https://doi.org/10.1029/2023JA031565

    """

    # Compute Pi-D and its components
    d_xyz, pi_xyz, _, pid = pid_4sc(r_mms, v_mms, p_mms)

    # Compute error propagation for the gradient of the velocity field (see c_4_k)
    errv_mms = [resample(errv_i, errv_mms[0]) for errv_i in errv_mms]
    errp_mms = [resample(errp_i, errv_mms[0]) for errp_i in errp_mms]
    r_mms = [resample(r, errv_mms[0]) for r in r_mms]
    p_mms = [resample(p_i, errv_mms[0]) for p_i in p_mms]

    # Compute reciprocal vectors in barycentric coordinates (see c_4_k)
    k_list = c_4_k(r_mms)

    sigma_grad_v_xyz = np.zeros((k_list[0].shape[0], 3, 3))

    for i in range(3):
        for j in range(3):
            for alpha in range(4):
                sigma_grad_v_xyz[:, i, j] += (
                    k_list[alpha].data[:, i] ** 2 * errv_mms[alpha].data[:, j] ** 2
                )

    sigma_grad_v_xyz = np.sqrt(sigma_grad_v_xyz)

    # Compute error propagation for the symmetric part of the velocity gradient tensor
    sigma_d_xyz = np.zeros_like(sigma_grad_v_xyz)

    # Diagonal components (Eq. 15 in Roberts et al., 2023)
    sigma_d_xyz[:, 0, 0] = (
        (4 / 9) * sigma_grad_v_xyz[:, 0, 0] ** 2
        + (1 / 9) * sigma_grad_v_xyz[:, 1, 1] ** 2
        + (1 / 9) * sigma_grad_v_xyz[:, 2, 2] ** 2
    )
    sigma_d_xyz[:, 1, 1] = (
        (1 / 9) * sigma_grad_v_xyz[:, 0, 0] ** 2
        + (4 / 9) * sigma_grad_v_xyz[:, 1, 1] ** 2
        + (1 / 9) * sigma_grad_v_xyz[:, 2, 2] ** 2
    )
    sigma_d_xyz[:, 2, 2] = (
        (1 / 9) * sigma_grad_v_xyz[:, 0, 0] ** 2
        + (1 / 9) * sigma_grad_v_xyz[:, 1, 1] ** 2
        + (4 / 9) * sigma_grad_v_xyz[:, 2, 2] ** 2
    )

    # Off-diagonal components (Eq. 14 in Roberts et al., 2023)
    sigma_d_xyz[:, 0, 1] = (1 / 4) * sigma_grad_v_xyz[:, 0, 1] ** 2 + (
        1 / 4
    ) * sigma_grad_v_xyz[:, 1, 0] ** 2
    sigma_d_xyz[:, 0, 2] = (1 / 4) * sigma_grad_v_xyz[:, 0, 2] ** 2 + (
        1 / 4
    ) * sigma_grad_v_xyz[:, 2, 0] ** 2
    sigma_d_xyz[:, 1, 2] = (1 / 4) * sigma_grad_v_xyz[:, 1, 2] ** 2 + (
        1 / 4
    ) * sigma_grad_v_xyz[:, 2, 1] ** 2
    sigma_d_xyz[:, 1, 0] = sigma_d_xyz[:, 0, 1]  # Symmetry
    sigma_d_xyz[:, 2, 0] = sigma_d_xyz[:, 0, 2]  # Symmetry
    sigma_d_xyz[:, 2, 1] = sigma_d_xyz[:, 1, 2]  # Symmetry
    sigma_d_xyz = np.sqrt(sigma_d_xyz)

    # Error in the center of mass pressure tensor
    sigma_p_xyz = (1 / 4) * np.sqrt(
        errp_mms[0].data ** 2
        + errp_mms[1].data ** 2
        + errp_mms[2].data ** 2
        + errp_mms[3].data ** 2
    )

    # Error in the traceless pressure tensor
    sigma_pi_xyz = np.zeros_like(sigma_p_xyz)
    sigma_pi_xyz[:, 0, 0] = (
        (4 / 9) * sigma_p_xyz[:, 0, 0] ** 2
        + (1 / 9) * sigma_p_xyz[:, 1, 1] ** 2
        + (1 / 9) * sigma_p_xyz[:, 2, 2] ** 2
    )
    sigma_pi_xyz[:, 1, 1] = (
        (1 / 9) * sigma_p_xyz[:, 0, 0] ** 2
        + (4 / 9) * sigma_p_xyz[:, 1, 1] ** 2
        + (1 / 9) * sigma_p_xyz[:, 2, 2] ** 2
    )
    sigma_pi_xyz[:, 2, 2] = (
        (1 / 9) * sigma_p_xyz[:, 0, 0] ** 2
        + (1 / 9) * sigma_p_xyz[:, 1, 1] ** 2
        + (4 / 9) * sigma_p_xyz[:, 2, 2] ** 2
    )
    sigma_pi_xyz[:, 0, 1] = sigma_p_xyz[:, 0, 1] ** 2
    sigma_pi_xyz[:, 0, 2] = sigma_p_xyz[:, 0, 2] ** 2
    sigma_pi_xyz[:, 1, 2] = sigma_p_xyz[:, 1, 2] ** 2
    sigma_pi_xyz[:, 1, 0] = sigma_pi_xyz[:, 0, 1]  # Symmetry
    sigma_pi_xyz[:, 2, 0] = sigma_pi_xyz[:, 0, 2]  # Symmetry
    sigma_pi_xyz[:, 2, 1] = sigma_pi_xyz[:, 1, 2]  # Symmetry
    sigma_pi_xyz = np.sqrt(sigma_pi_xyz)

    sigma_pid_xyz_rel = np.zeros_like(pid.data)
    sigma_pid_xyz_rel = np.sqrt(
        (sigma_d_xyz / d_xyz.data) ** 2 + (sigma_pi_xyz / pi_xyz.data) ** 2
    )
    sigma_pid_xyz = np.abs(pi_xyz.data * d_xyz.data) * sigma_pid_xyz_rel

    sigma_pid = np.sqrt(
        sigma_pid_xyz[:, 0, 0] ** 2
        + sigma_pid_xyz[:, 1, 1] ** 2
        + sigma_pid_xyz[:, 2, 2] ** 2
        + 4 * sigma_pid_xyz[:, 0, 1] ** 2
        + 4 * sigma_pid_xyz[:, 0, 2] ** 2
        + 4 * sigma_pid_xyz[:, 1, 2] ** 2
    )
    sigma_pid = ts_scalar(v_mms[0].time.data, sigma_pid)

    return sigma_pid
