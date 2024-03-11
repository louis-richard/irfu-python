#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz
from ..pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


@numba.jit(cache=True, fastmath=True, nogil=True, parallel=True, nopython=True)
def _moms(
    energy,
    delta_v,
    q_e,
    sc_pot,
    p_mass,
    flag_inner_electron,
    w_inner_electron,
    phi,
    theta,
    int_energies,
    vdf,
    delta_ang,
):
    n_psd = np.zeros(vdf.shape[0])
    v_psd = np.zeros((vdf.shape[0], 3))
    p_psd = np.zeros((vdf.shape[0], 3, 3))
    h_psd = np.zeros((vdf.shape[0], 3))

    for i_t in numba.prange(vdf.shape[0]):
        energy_correct = energy[i_t, :] - sc_pot[i_t]

        velocity = np.sqrt(2 * q_e * energy_correct / p_mass)
        velocity[energy_correct < flag_inner_electron * w_inner_electron] = 0

        phi_i = np.deg2rad(phi[i_t, :, :])
        theta_i = np.deg2rad(theta[i_t, :, :])

        psd2n_mat = np.ones(theta_i.shape) * np.sin(theta_i)

        # Particle flux and heat flux vector
        psd2v_x_mat = -np.cos(phi_i) * np.sin(theta_i) ** 2
        psd2v_y_mat = -np.sin(phi_i) * np.sin(theta_i) ** 2
        psd2v_z_mat = -np.ones(theta_i.shape) * np.sin(theta_i) * np.cos(theta_i)

        psd2p_xx_mat = np.cos(phi_i) ** 2.0 * np.sin(theta_i) ** 3
        psd2p_yy_mat = np.sin(phi_i) ** 2.0 * np.sin(theta_i) ** 3
        psd2p_zz_mat = np.ones(theta_i.shape) * np.sin(theta_i) * np.cos(theta_i) ** 2
        psd2p_xy_mat = np.cos(phi_i) * np.sin(phi_i) * np.sin(theta_i) ** 3
        psd2p_xz_mat = np.cos(phi_i) * np.sin(phi_i) ** 2 * np.cos(theta_i)
        psd2p_yz_mat = np.sin(phi_i) * np.sin(phi_i) ** 2 * np.cos(theta_i)

        for i_e in int_energies:
            tmp = vdf[i_t, i_e, :, :]
            # n_psd_tmp1 = tmp .* psd2_n_mat * v(ii)^2 * delta_v(ii) * delta_ang;
            # n_psd_e32_phi_theta(nt, ii, :, :) = n_psd_tmp1;
            # n_psd_e32(nt, ii) = n_psd_tmp

            # number density
            n_psd_tmp = np.nansum(tmp * psd2n_mat * delta_ang[i_t])
            n_psd_tmp *= delta_v[i_t, i_e] * velocity[i_e] ** 2
            n_psd[i_t] += n_psd_tmp

            # Bulk velocity
            v_temp_x = np.nansum(tmp * psd2v_x_mat * delta_ang[i_t])
            v_temp_x *= delta_v[i_t, i_e] * velocity[i_e] ** 3

            v_temp_y = np.nansum(tmp * psd2v_y_mat * delta_ang[i_t])
            v_temp_y *= delta_v[i_t, i_e] * velocity[i_e] ** 3

            v_temp_z = np.nansum(tmp * psd2v_z_mat * delta_ang[i_t])
            v_temp_z *= delta_v[i_t, i_e] * velocity[i_e] ** 3

            v_psd[i_t, 0] += v_temp_x
            v_psd[i_t, 1] += v_temp_y
            v_psd[i_t, 2] += v_temp_z

            # Pressure tensor
            p_temp_xx = np.nansum(tmp * psd2p_xx_mat * delta_ang[i_t])
            p_temp_xx *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_temp_xy = np.nansum(tmp * psd2p_xy_mat * delta_ang[i_t])
            p_temp_xy *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_temp_xz = np.nansum(tmp * psd2p_xz_mat * delta_ang[i_t])
            p_temp_xz *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_temp_yy = np.nansum(tmp * psd2p_yy_mat * delta_ang[i_t])
            p_temp_yy *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_temp_yz = np.nansum(tmp * psd2p_yz_mat * delta_ang[i_t])
            p_temp_yz *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_temp_zz = np.nansum(tmp * psd2p_zz_mat * delta_ang[i_t])
            p_temp_zz *= delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_psd[i_t, 0, 0] += p_temp_xx
            p_psd[i_t, 0, 1] += p_temp_xy
            p_psd[i_t, 0, 2] += p_temp_xz
            p_psd[i_t, 1, 1] += p_temp_yy
            p_psd[i_t, 1, 2] += p_temp_yz
            p_psd[i_t, 2, 2] += p_temp_zz

            h_psd[i_t, 0] = v_temp_x * velocity[i_e] ** 2
            h_psd[i_t, 1] = v_temp_y * velocity[i_e] ** 2
            h_psd[i_t, 2] = v_temp_z * velocity[i_e] ** 2

    return n_psd, v_psd, p_psd, h_psd


def psd_moments(vdf, sc_pot, **kwargs):
    r"""Computes moments from the FPI particle phase-space densities.

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution.
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.

    Returns
    -------
    n_psd : xarray.DataArray
        Time series of the number density (1rst moment).
    v_psd : xarray.DataArray
        Time series of the bulk velocity (2nd moment).
    p_psd : xarray.DataArray
        Time series of the pressure tensor (3rd moment).
    p2_psd : xarray.DataArray
        Time series of the pressure tensor.
    t_psd : xarray.DataArray
        Time series of the temperature tensor.
    h_psd : xarray.DataArray
        to fill.

    Other Parameters
    ----------------
    energy_range : array_like
        Set energy range in eV to integrate over [E_min E_max]. Energy range
        is applied to energy0 and the same elements are used for energy1 to
        ensure that the same number of points are integrated over.
    no_sc_pot : bool
        Set to 1 to set spacecraft potential to zero. Calculates moments
        without correcting for spacecraft potential.
    en_channels : array_like
        Set energy channels to integrate over [min max]; min and max between
        must be between 1 and 32.
    partial_moments : numpy.ndarray or xarray.DataArray
        Use a binary array to select which psd points are used in the moments
        calculation. `partial_moments` must be a binary array (1s and 0s,
        1s correspond to points used). Array (or data of Dataarray) must be the same
        size as vdf.data.
    inner_electron : {"on", "off"}
        inner_electrontron potential for electron moments.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Load magnetic field and spacecraft potential

    >>> scpot = mms.get_data("V_edp_brst_l2", tint_brst, 1)

    Load electron velocity distribution function

    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint_brst, 1)

    Compute moments

    >>> options = dict(energy_range=[1, 1000])
    >>> moments_e = mms.psd_moments(vdf_e, scpot, **options)

    """

    # [eV] sc_pot + w_inner_electron for electron moments calculation
    w_inner_electron = 3.5

    # Check if data is fast or burst resolution
    if "brst" in vdf.data.attrs["FIELDNAM"].lower():
        is_brst_data = True
        logging.info("Burst resolution data is used")
    elif "fast" in vdf.data.attrs["FIELDNAM"].lower():
        is_brst_data = False
        logging.info("Fast resolution data is used")
    else:
        raise TypeError("Could not identify if data is fast or burst.")

    theta = vdf.theta.data
    particle_type = vdf.attrs["species"]
    assert particle_type[0].lower() in ["e", "i"], "invalid particle type"

    vdf_data = vdf.data.data * 1e12  # In SI units

    step_table = vdf.attrs["esteptable"]
    energy = vdf.energy.data
    energy0 = vdf.attrs["energy0"]
    energy1 = vdf.attrs["energy1"]
    e_tmp = energy1 - energy0

    flag_same_e = all(e_tmp) == 0

    # resample sc_pot to same resolution as particle distributions
    sc_pot = resample(sc_pot, vdf.time).data

    if "energy_range" in kwargs:
        if (
            isinstance(kwargs["energy_range"], (list, np.ndarray))
            and len(kwargs["energy_range"]) == 2
        ):
            # if not is_brst_data:
            #    energy0 = energy

            # e_min_max = kwargs["energy_range"]
            # start_e = bisect.bisect_left(energy0, e_min_max[0])
            # stop_e = bisect.bisect_left(energy0, e_min_max[1])

            logging.info("Using partial energy range")

    no_sc_pot = kwargs.get("no_sc_pot", False)
    if no_sc_pot:
        sc_pot = np.zeros(sc_pot.shape)
        logging.info("Setting spacecraft potential to zero")

    int_energies = np.arange(
        kwargs.get("en_channels", [0, 32])[0],
        kwargs.get("en_channels", [0, 32])[1],
    )

    if "partial_moments" in kwargs:
        partial_moments = kwargs["partial_moments"]
        if isinstance(partial_moments, xr.DataArray):
            partial_moments = partial_moments.data

        # Check size of partial_moments
        if partial_moments.shape == vdf_data.shape:
            sum_ones = np.sum(
                np.sum(
                    np.sum(np.sum(partial_moments, axis=-1), axis=-1),
                    axis=-1,
                ),
                axis=-1,
            )
            sum_zeros = np.sum(
                np.sum(
                    np.sum(np.sum(-partial_moments + 1, axis=-1), axis=-1),
                    axis=-1,
                ),
                axis=-1,
            )

            if (sum_ones + sum_zeros) == vdf_data.size:
                logging.info(
                    "partial_moments is correct. Partial moments will be calculated"
                )
                vdf_data = vdf_data * partial_moments
            else:
                logging.info(
                    "All values are not ones and zeros in partial_moments. "
                    "Full moments will be calculated"
                )
        else:
            logging.info(
                "Size of partial_moments is wrong. Full moments will be calculated"
            )

    tmp_ = kwargs.get("inner_electron", "")
    flag_inner_electron = tmp_ == "on" and particle_type[0] == "e"

    # Define constants
    q_e = constants.elementary_charge
    k_b = constants.Boltzmann

    if particle_type[0] == "e":
        p_mass = constants.electron_mass
        logging.info("Particles are electrons")
    else:
        p_mass = constants.proton_mass
        sc_pot *= -1.0
        logging.info("Particles are ions")

    # angle between theta and phi points is 360/32 = 11.25 degrees
    phi = vdf.phi.data

    if "delta_phi_minus" in vdf.attrs and "delta_phi_plus" in vdf.attrs:
        delta_phi_minus = vdf.attrs["delta_phi_minus"]
        delta_phi_plus = vdf.attrs["delta_phi_plus"]
        delta_phi = delta_phi_plus + delta_phi_minus
        delta_phi = np.tile(delta_phi[:, :, np.newaxis], (1, 1, vdf_data.shape[3]))
    else:
        delta_phi = np.deg2rad(np.median(np.diff(phi[0, :])))
        delta_phi = delta_phi * np.ones(
            (vdf_data.shape[0], vdf_data.shape[2], vdf_data.shape[3])
        )

    if "delta_theta_minus" in vdf.attrs and "delta_theta_plus" in vdf.attrs:
        delta_theta_minus = vdf.attrs["delta_theta_minus"]
        delta_theta_plus = vdf.attrs["delta_theta_plus"]
        delta_theta = delta_theta_plus + delta_theta_minus
        delta_theta = np.tile(
            delta_theta[np.newaxis, np.newaxis, :],
            (vdf_data.shape[0], vdf_data.shape[2], 1),
        )
    else:
        delta_theta = np.deg2rad(np.median(np.diff(theta)))
        delta_theta = delta_theta * np.ones(
            (vdf_data.shape[0], vdf_data.shape[2], vdf_data.shape[3])
        )

    delta_ang = delta_phi * delta_theta

    phi_mat = np.tile(phi[:, :, np.newaxis], (1, 1, vdf_data.shape[3]))
    theta_mat = np.tile(
        theta[np.newaxis, np.newaxis, :], (vdf_data.shape[0], vdf_data.shape[2], 1)
    )

    energy_minus = vdf.attrs["delta_energy_plus"]
    energy_plus = vdf.attrs["delta_energy_plus"]

    energy_correct = energy - sc_pot[:, np.newaxis]
    velocity = np.sqrt(2 * q_e * energy_correct / p_mass)
    velocity[energy_correct < flag_inner_electron * w_inner_electron] = 0

    # Calculate speed widths associated with each energy channel.
    if is_brst_data:  # Burst mode energy/speed widths
        if flag_same_e:
            energy_upper = energy + energy_plus
            energy_lower = energy - energy_minus
            v_upper = np.sqrt(2 * q_e * energy_upper / p_mass)
            v_lower = np.sqrt(2 * q_e * energy_lower / p_mass)
            delta_v = v_upper - v_lower
            delta_v = np.tile(delta_v, (vdf_data.shape[0], 1))
        else:
            energy_all = np.hstack([energy0, energy1])
            energy_all = np.log10(np.sort(energy_all))

            if np.abs(energy_all[1] - energy_all[0]) > 1e-4:
                temp0 = 2 * energy_all[0] - energy_all[1]
            else:
                temp0 = 2 * energy_all[1] - energy_all[2]

            if np.abs(energy_all[63] - energy_all[62]) > 1e-4:
                temp65 = 2 * energy_all[63] - energy_all[62]
            else:
                temp65 = 2 * energy_all[63] - energy_all[61]

            energy_all = np.hstack([temp0, energy_all, temp65])
            diff_en_all = np.diff(energy_all)
            energy0upper = 10 ** (np.log10(energy0) + diff_en_all[1:64:2] / 2)
            energy0lower = 10 ** (np.log10(energy0) - diff_en_all[0:63:2] / 2)
            energy1upper = 10 ** (np.log10(energy1) + diff_en_all[2:65:2] / 2)
            energy1lower = 10 ** (np.log10(energy1) - diff_en_all[1:64:2] / 2)

            v0upper = np.sqrt(2 * q_e * energy0upper / p_mass)
            v0lower = np.sqrt(2 * q_e * energy0lower / p_mass)
            v1upper = np.sqrt(2 * q_e * energy1upper / p_mass)
            v1lower = np.sqrt(2 * q_e * energy1lower / p_mass)
            delta_v0 = (v0upper - v0lower) * 2.0
            delta_v1 = (v1upper - v1lower) * 2.0

            delta_v = np.tile(delta_v0, (vdf_data.shape[0], 1))
            delta_v[step_table == 1, :] = delta_v1

    else:  # Fast mode energy/speed widths
        energy_all = np.log10(energy[0, :])
        temp0 = 2 * energy_all[0] - energy_all[1]
        temp33 = 2 * energy_all[31] - energy_all[30]
        energy_all = np.hstack([temp0, energy_all, temp33])
        diff_en_all = np.diff(energy_all)
        energy_upper = 10 ** (np.log10(energy[0, :]) + diff_en_all[1:33] / 4)
        energy_lower = 10 ** (np.log10(energy[0, :]) - diff_en_all[0:32] / 4)
        v_upper = np.sqrt(2 * q_e * energy_upper / p_mass)
        v_lower = np.sqrt(2 * q_e * energy_lower / p_mass)
        delta_v = (v_upper - v_lower) * 2.0
        delta_v[0] = delta_v[0] * 2.7
        delta_v = np.tile(delta_v, (vdf_data.shape[0], 1))

    n_psd, v_psd, p_psd, h_psd = _moms(
        energy,
        delta_v,
        q_e,
        sc_pot,
        p_mass,
        flag_inner_electron,
        w_inner_electron,
        phi_mat,
        theta_mat,
        int_energies,
        vdf_data,
        delta_ang,
    )

    # Compute moments in SI units
    p_psd *= p_mass
    v_psd /= n_psd[:, np.newaxis]
    p2_psd = np.zeros_like(p_psd)
    p2_psd[:, 0, 0] = p_psd[:, 0, 0]
    p2_psd[:, 0, 1] = p_psd[:, 0, 1]
    p2_psd[:, 0, 2] = p_psd[:, 0, 2]
    p2_psd[:, 1, 1] = p_psd[:, 1, 1]
    p2_psd[:, 1, 2] = p_psd[:, 1, 2]
    p2_psd[:, 2, 2] = p_psd[:, 2, 2]
    p2_psd[:, 1, 0] = p2_psd[:, 0, 1]
    p2_psd[:, 2, 0] = p2_psd[:, 0, 2]
    p2_psd[:, 2, 1] = p2_psd[:, 1, 2]

    p_psd[:, 0, 0] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 0]
    p_psd[:, 0, 1] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 1]
    p_psd[:, 0, 2] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 2]
    p_psd[:, 1, 1] -= p_mass * n_psd * v_psd[:, 1] * v_psd[:, 1]
    p_psd[:, 1, 2] -= p_mass * n_psd * v_psd[:, 1] * v_psd[:, 2]
    p_psd[:, 2, 2] -= p_mass * n_psd * v_psd[:, 2] * v_psd[:, 2]
    p_psd[:, 1, 0] = p_psd[:, 0, 1]
    p_psd[:, 2, 0] = p_psd[:, 0, 2]
    p_psd[:, 2, 1] = p_psd[:, 1, 2]

    p_trace = np.trace(p_psd, axis1=1, axis2=2)
    t_psd = np.zeros(p_psd.shape)
    t_psd[...] = p_psd[...] / (k_b * n_psd[:, np.newaxis, np.newaxis])
    t_psd[:, 1, 0] = t_psd[:, 1, 0]
    t_psd[:, 2, 0] = t_psd[:, 2, 0]
    t_psd[:, 2, 1] = t_psd[:, 2, 1]

    v_abs2 = np.linalg.norm(v_psd, axis=1) ** 2
    h_psd *= p_mass / 2
    h_psd[:, 0] -= v_psd[:, 0] * p_psd[:, 0, 0]
    h_psd[:, 0] -= v_psd[:, 1] * p_psd[:, 0, 1]
    h_psd[:, 0] -= v_psd[:, 2] * p_psd[:, 0, 2]
    h_psd[:, 0] -= 0.5 * v_psd[:, 0] * (p_trace + p_mass * n_psd * v_abs2)
    h_psd[:, 1] -= v_psd[:, 0] * p_psd[:, 1, 0]
    h_psd[:, 1] -= v_psd[:, 1] * p_psd[:, 1, 1]
    h_psd[:, 1] -= v_psd[:, 2] * p_psd[:, 1, 2]
    h_psd[:, 1] -= 0.5 * v_psd[:, 1] * (p_trace + p_mass * n_psd * v_abs2)
    h_psd[:, 2] -= v_psd[:, 0] * p_psd[:, 2, 0]
    h_psd[:, 2] -= v_psd[:, 1] * p_psd[:, 2, 1]
    h_psd[:, 2] -= v_psd[:, 2] * p_psd[:, 2, 2]
    h_psd[:, 2] -= 0.5 * v_psd[:, 2] * (p_trace + p_mass * n_psd * v_abs2)

    # Convert to typical units (/cc, km/s, nP, eV, and ergs/s/cm^2).
    n_psd /= 1e6
    v_psd /= 1e3
    p_psd *= 1e9
    p2_psd *= 1e9
    t_psd *= k_b / q_e
    h_psd *= 1e3

    # Construct TSeries
    n_psd = ts_scalar(vdf.time.data, n_psd)
    v_psd = ts_vec_xyz(vdf.time.data, v_psd)
    p_psd = ts_tensor_xyz(vdf.time.data, p_psd)
    p2_psd = ts_tensor_xyz(vdf.time.data, p2_psd)
    t_psd = ts_tensor_xyz(vdf.time.data, t_psd)
    h_psd = ts_vec_xyz(vdf.time.data, h_psd)

    return n_psd, v_psd, p_psd, p2_psd, t_psd, h_psd
