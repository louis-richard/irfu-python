#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

from ..pyrf.dec_par_perp import dec_par_perp
from ..pyrf.norm import norm
from ..pyrf.resample import resample
from ..pyrf.trace import trace
from ..pyrf.ts_scalar import ts_scalar

# Local imports
from .rotate_tensor import rotate_tensor

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def make_model_vdf(
    vdf,
    b_xyz,
    sc_pot,
    n_s,
    v_xyz,
    t_xyz,
    isotropic: bool = False,
):
    r"""Make a general bi-Maxwellian distribution function based on particle
    moment data in the same format as PDist.

    Parameters
    ----------
    vdf : xarray.Dataset
        Particle distribution (skymap).
    b_xyz : xarray.DataArray
        Time series of the background magnetic field.
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.
    n_s : xarray.DataArray
        Time series of the number density of specie s.
    v_xyz : xarray.DataArray
        Time series of the bulk velocity.
    t_xyz : xarray.DataArray
        Time series of the temperature tensor.
    isotropic : bool, Optional
        Flag to make an isotropic model distribution. Default is False.

    Returns
    -------
    model_vdf : xarray.Dataset
        Distribution function in the same format as vdf.

    See also
    --------
    pyrfu.mms.calculate_epsilon : Calculates epsilon parameter using model distribution.

    Examples
    --------
    >>> from pyrfu.mms import get_data, make_model_vdf

    Define time interval

    >>> tint_brst = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Load magnetic field and spacecraft potential

    >>> b_dmpa = get_data("b_dmpa_fgm_brst_l2", tint_brst, 1)
    >>> scpot = get_data("V_edp_brst_l2", tint_brst, 1)

    Load electron velocity distribution function

    >>> vdf_e = get_data("pde_fpi_brst_l2", tint_brst, 1)

    Load moments of the electron velocity distribution function

    >>> n_e = get_data("ne_fpi_brst_l2", tint_brst, 1)
    >>> v_xyz_e = get_data("ve_dbcs_fpi_brst_l2", tint_brst, 1)
    >>> t_xyz_e = get_data("te_dbcs_fpi_brst_l2", tint_brst, 1)

    Compute model electron velocity distribution function

    >>> vdf_m_e = make_model_vdf(vdf_e, b_xyz, scpot, n_e, v_xyz_e, t_xyz_e)

    """

    assert vdf.attrs["species"][0].lower() in ["i", "e"], "Invalid specie"

    # Check that VDF and moments have the same timeline
    message = "VDF and moments have different times."
    assert np.abs(np.median(np.diff(vdf.time.data - n_s.time.data))) == 0, message

    # Resample b_xyz and sc_pot to particle data resolution
    b_xyz, sc_pot = [resample(b_xyz, n_s), resample(sc_pot, n_s)]

    # Define directions based on b_xyz and v_xyz, calculate relevant
    # temperatures. N.B makes final distribution gyrotropic
    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")

    if isotropic:
        t_para = trace(t_xyzfac) / 3
        t_ratio = ts_scalar(
            t_xyzfac.time.data,
            np.ones(len(t_xyzfac.time.data)),
        )
    else:
        t_para = t_xyzfac[:, 0, 0]
        t_ratio = t_xyzfac[:, 0, 0] / t_xyzfac[:, 1, 1]

    v_para, v_perp, _ = dec_par_perp(v_xyz, b_xyz)

    v_perp_mag, b_xyz_mag = [norm(v_perp), norm(b_xyz)]
    v_perp_dir, b_xyz_dir = [v_perp / v_perp_mag, b_xyz / b_xyz_mag]

    # Define constants
    q_e = constants.elementary_charge

    # Check whether particles are electrons or ions
    if vdf.attrs["species"][0].lower() == "e":
        p_mass = constants.electron_mass
    else:
        p_mass = constants.proton_mass
        sc_pot.data = -1.0 * sc_pot.data

    # Convert moments to SI units
    vth_para = np.sqrt(2 * t_para.data * q_e / p_mass)

    v_perp_mag_data = 1e3 * v_perp_mag.data
    v_para_data = 1e3 * v_para.data
    n_s_data = 1e6 * n_s.data

    # Defines dimensions of array below
    n_ti = len(vdf.time)
    n_en = len(vdf.energy.data[0, :])
    n_ph, n_th = [len(angle) for angle in [vdf.phi[0, :], vdf.theta]]

    # Get energy array
    energy = vdf.energy

    # Define Cartesian coordinates
    x_mat, y_mat, z_mat = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    r_mat = np.zeros((n_ti, n_en))

    for i in range(n_ti):
        x_mat[i, ...] = np.outer(
            -np.cos(np.deg2rad(vdf.phi.data[i, :])),
            np.sin(np.deg2rad(vdf.theta.data)),
        )
        y_mat[i, ...] = np.outer(
            -np.sin(np.deg2rad(vdf.phi.data[i, :])),
            np.sin(np.deg2rad(vdf.theta.data)),
        )
        z_mat[i, ...] = np.outer(
            -np.ones(n_ph),
            np.cos(np.deg2rad(vdf.theta.data)),
        )
        r_mat[i, ...] = np.real(
            np.sqrt(2 * (energy[i, :] - sc_pot.data[i]) * q_e / p_mass),
        )

    r_mat[r_mat == 0] = 0.0

    # Define rotation vectors based on B and Ve directions
    r_x = v_perp_dir.data
    r_y = np.cross(b_xyz_dir.data, v_perp_dir.data)
    r_z = b_xyz_dir.data

    # Rotated coordinate system for computing bi-Maxwellian distribution
    x_p, y_p, z_p = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    for i in range(n_ti):
        x_p[i, ...] = (
            x_mat[i, ...] * r_x[i, 0]
            + y_mat[i, ...] * r_x[i, 1]
            + z_mat[i, ...] * r_x[i, 2]
        )

        y_p[i, ...] = (
            x_mat[i, ...] * r_y[i, 0]
            + y_mat[i, ...] * r_y[i, 1]
            + z_mat[i, ...] * r_y[i, 2]
        )

        z_p[i, ...] = (
            x_mat[i, ...] * r_z[i, 0]
            + y_mat[i, ...] * r_z[i, 1]
            + z_mat[i, ...] * r_z[i, 2]
        )

    # Make 4D position matrix
    x_p = np.transpose(np.tile(x_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    y_p = np.transpose(np.tile(y_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    z_p = np.transpose(np.tile(z_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    r_mat = np.transpose(np.tile(r_mat, [n_ph, n_th, 1, 1]), [2, 3, 0, 1])

    # Construct bi-Maxwellian distribution function
    bi_max_dist = np.zeros(r_mat.shape)

    for i in range(n_ti):
        coeff = (
            n_s_data[i] * t_ratio.data[i] / (np.sqrt(np.pi**3) * vth_para.data[i] ** 3)
        )

        bi_max_temp = coeff * np.exp(
            -((x_p[i, ...] * r_mat[i, ...] - v_perp_mag_data[i]) ** 2)
            / (vth_para.data[i] ** 2)
            * t_ratio.data[i],
        )
        bi_max_temp = bi_max_temp * np.exp(
            -((y_p[i, ...] * r_mat[i, ...]) ** 2)
            / (vth_para.data[i] ** 2)
            * t_ratio.data[i],
        )
        bi_max_temp = bi_max_temp * np.exp(
            -((z_p[i, ...] * r_mat[i, ...] - v_para_data[i]) ** 2)
            / (vth_para.data[i] ** 2),
        )

        bi_max_dist[i, ...] = bi_max_temp

    # Make modelPDist file for output
    model_vdf = vdf.copy()
    model_vdf.data.data = bi_max_dist
    model_vdf.data.data *= 1e18
    model_vdf.attrs["UNITS"] = "s^3/km^6"

    return model_vdf
