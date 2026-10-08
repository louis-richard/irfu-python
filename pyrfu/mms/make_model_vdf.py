#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

from ..pyrf.dec_par_perp import dec_par_perp
from ..pyrf.norm import norm
from ..pyrf.resample import resample
from ..pyrf.trace import trace

# Local imports
from .rotate_tensor import rotate_tensor

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _perp_direction(v_perp, b_hat):
    r"""Unit vectors along v_perp, or any direction perpendicular to b_hat
    where v_perp is zero (the model is gyrotropic)."""
    v_perp_mag = np.linalg.norm(v_perp, axis=1, keepdims=True)

    # Perpendicular to B from x (or y if B is along x)
    ref = np.where(np.abs(b_hat[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    any_perp = np.cross(b_hat, ref)
    any_perp /= np.linalg.norm(any_perp, axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(v_perp_mag > 0, v_perp / v_perp_mag, any_perp)

    return out


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
        Distribution function in the same format as vdf, in s^3/km^6. The
        channels whose energy is below the spacecraft potential correspond to
        no particle velocity and are NaN (irfu-matlab gives the value at
        v = 0). Their weight is zero in velocity-space integrals (v^2 dv), so
        :func:`pyrfu.mms.calculate_epsilon` and moments are unaffected.

    Raises
    ------
    ValueError
        If the species is not ions or electrons, or if vdf and the moments
        have different times.

    See also
    --------
    pyrfu.mms.calculate_epsilon : Calculates epsilon parameter using model distribution.

    Notes
    -----
    The model is gyrotropic: when the bulk velocity is parallel to the
    magnetic field (or zero), any direction perpendicular to it is used for
    the perpendicular axis.

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

    species = str(vdf.attrs["species"]).lower()

    if species[:1] not in ["i", "e"]:
        raise ValueError(f"Invalid species {vdf.attrs['species']!r}")

    # Check that VDF and moments have the same timeline
    d_times = vdf.time.data - n_s.time.data if len(vdf.time) == len(n_s.time) else None

    if d_times is None or (
        len(d_times) > 1 and np.median(np.diff(d_times)) != np.timedelta64(0, "ns")
    ):
        raise ValueError("VDF and moments have different times.")

    # Resample b_xyz and sc_pot to particle data resolution
    b_xyz, sc_pot = [resample(b_xyz, n_s), resample(sc_pot, n_s)]

    # Define directions based on b_xyz and v_xyz, calculate relevant
    # temperatures. N.B makes final distribution gyrotropic
    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")

    if isotropic:
        t_para = trace(t_xyzfac).data / 3
        t_ratio = np.ones(len(t_xyzfac.time.data))
    else:
        t_para = t_xyzfac.data[:, 0, 0]
        t_ratio = t_xyzfac.data[:, 0, 0] / t_xyzfac.data[:, 1, 1]

    v_para, v_perp, _ = dec_par_perp(v_xyz, b_xyz)

    # Rotation vectors based on B and the perpendicular bulk velocity
    r_z = (b_xyz / norm(b_xyz)).data
    r_x = _perp_direction(v_perp.data, r_z)
    r_y = np.cross(r_z, r_x)

    # Check whether particles are electrons or ions
    q_e = constants.elementary_charge

    if species[0] == "e":
        p_mass = constants.electron_mass
        sc_pot = sc_pot.data
    else:
        p_mass = constants.proton_mass
        sc_pot = -sc_pot.data

    # Convert moments to SI units
    vth_para = np.sqrt(2 * t_para * q_e / p_mass)
    v_perp_mag = 1e3 * norm(v_perp).data
    v_para = 1e3 * v_para.data
    n_s = 1e6 * n_s.data

    # Directions of the particle velocities (minus the look directions),
    # (time, phi, theta)
    phi = np.deg2rad(vdf.phi.data)[:, :, None]
    theta = np.deg2rad(vdf.theta.data)[None, None, :]
    dirs = [
        -np.cos(phi) * np.sin(theta),
        -np.sin(phi) * np.sin(theta),
        -np.cos(theta) * np.ones_like(phi),
    ]

    # Components in the (v_perp, B x v_perp, B) frame, (time, 1, phi, theta)
    x_p, y_p, z_p = [
        sum(d * r_[:, i, None, None] for i, d in enumerate(dirs))[:, None, ...]
        for r_ in [r_x, r_y, r_z]
    ]

    # Speeds corrected for the spacecraft potential, (time, energy, 1, 1)
    energy = vdf.energy.data.astype(np.float64) - sc_pot[:, None]
    below = energy < 0
    speed = np.sqrt(2 * np.clip(energy, 0.0, None) * q_e / p_mass)[..., None, None]

    # Bi-Maxwellian distribution function
    def _col(x):
        return x[:, None, None, None]

    coeff = n_s * t_ratio / (np.sqrt(np.pi**3) * vth_para**3)
    exponent = (x_p * speed - _col(v_perp_mag)) ** 2 + (y_p * speed) ** 2
    exponent *= _col(t_ratio)
    exponent += (z_p * speed - _col(v_para)) ** 2
    bi_max_dist = _col(coeff) * np.exp(-exponent / _col(vth_para**2))

    # No particle velocity below the spacecraft potential
    bi_max_dist[below] = np.nan

    # Make modelPDist file for output, in s^3/km^6
    model_vdf = vdf.copy()
    model_vdf["data"] = vdf.data.copy(data=bi_max_dist * 1e18)
    model_vdf.data.attrs = {**vdf.data.attrs, "UNITS": "s^3/km^6"}

    return model_vdf
