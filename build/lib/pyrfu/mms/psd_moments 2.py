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

import bisect
import multiprocessing as mp
import numpy as np
import xarray as xr

from scipy import constants

from ..pyrf import resample, ts_scalar, ts_vec_xyz, ts_tensor_xyz


# noinspection PyUnboundLocalVariable
def calc_moms(time_idx, arguments):
    """

    """
    if len(arguments) > 13:
        [is_brst_data, flag_same_e, flag_de, step_table, energy0, delta_v0, energy1, delta_v1,
         q_e, sc_pot, p_mass, flag_inner_electron, w_inner_electron, phi_tr, theta_k, int_energies,
         vdf, delta_ang] = arguments
    else:
        [is_brst_data, flag_same_e, flag_de, energy, delta_v, q_e, sc_pot, p_mass,
         flag_inner_electron, w_inner_electron, phi_tr, theta_k, int_energies, vdf,
         delta_ang] = arguments

    if is_brst_data:
        if not flag_same_e or not flag_de:
            energy = energy0
            delta_v = delta_v0

            if step_table[time_idx]:
                energy = energy1
                delta_v = delta_v1

    velocity = np.real(np.sqrt(2 * q_e * (energy - sc_pot.data[time_idx]) / p_mass))
    velocity[energy - sc_pot.data[time_idx] - flag_inner_electron * w_inner_electron < 0] = 0

    if is_brst_data:
        phi_j = phi_tr[time_idx, :]
    else:
        phi_j = phi_tr

    phi_j = phi_j[:, np.newaxis]

    n_psd = 0
    v_psd = np.zeros(3)
    p_psd = np.zeros((3, 3))
    h_psd = np.zeros(3)

    psd2_n_mat = np.dot(np.ones(phi_j.shape), np.sin(theta_k * np.pi / 180))
    psd2_v_x_mat = -np.dot(np.cos(phi_j * np.pi / 180),
                           np.sin(theta_k * np.pi / 180) * np.sin(theta_k * np.pi / 180))
    psd2_v_y_mat = -np.dot(np.sin(phi_j * np.pi / 180),
                           np.sin(theta_k * np.pi / 180) * np.sin(theta_k * np.pi / 180))
    psd2_v_z_mat = -np.dot(np.ones(phi_j.shape),
                           np.sin(theta_k * np.pi / 180) * np.cos(theta_k * np.pi / 180))
    psd_mf_xx_mat = np.dot(np.cos(phi_j * np.pi / 180) ** 2, np.sin(theta_k * np.pi / 180) ** 3)
    psd_mf_yy_mat = np.dot(np.sin(phi_j * np.pi / 180) ** 2, np.sin(theta_k * np.pi / 180) ** 3)
    psd_mf_zz_mat = np.dot(np.ones(phi_j.shape),
                           np.sin(theta_k * np.pi / 180) * np.cos(theta_k * np.pi / 180) ** 2)
    psd_mf_xy_mat = np.dot(np.cos(phi_j * np.pi / 180) * np.sin(phi_j * np.pi / 180),
                           np.sin(theta_k * np.pi / 180) ** 3)
    psd_mf_xz_mat = np.dot(np.cos(phi_j * np.pi / 180),
                           np.cos(theta_k * np.pi / 180) * np.sin(theta_k * np.pi / 180) ** 2)
    psd_mf_yz_mat = np.dot(np.sin(phi_j * np.pi / 180),
                           np.cos(theta_k * np.pi / 180) * np.sin(theta_k * np.pi / 180) ** 2)

    for i in int_energies:
        tmp = np.squeeze(vdf[time_idx, i, :, :])
        # n_psd_tmp1 = tmp .* psd2_n_mat * v(ii)^2 * delta_v(ii) * delta_ang;
        # n_psd_e32_phi_theta(nt, ii, :, :) = n_psd_tmp1;
        # n_psd_e32(nt, ii) = n_psd_tmp

        # number density
        n_psd_tmp = np.nansum(np.nansum(tmp * psd2_n_mat, axis=0), axis=0)
        n_psd_tmp *= delta_v[i] * delta_ang * velocity[i] ** 2
        n_psd += n_psd_tmp

        # Bulk velocity
        v_temp_x = np.nansum(np.nansum(tmp * psd2_v_x_mat, axis=0), axis=0)
        v_temp_x *= delta_v[i] * delta_ang * velocity[i] ** 3

        v_temp_y = np.nansum(np.nansum(tmp * psd2_v_y_mat, axis=0), axis=0)
        v_temp_y *= delta_v[i] * delta_ang * velocity[i] ** 3

        v_temp_z = np.nansum(np.nansum(tmp * psd2_v_z_mat, axis=0), axis=0)
        v_temp_z *= delta_v[i] * delta_ang * velocity[i] ** 3

        v_psd[0] += v_temp_x
        v_psd[1] += v_temp_y
        v_psd[2] += v_temp_z

        # Pressure tensor
        p_temp_xx = np.nansum(np.nansum(tmp * psd_mf_xx_mat, axis=0), axis=0)
        p_temp_xx *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_temp_xy = np.nansum(np.nansum(tmp * psd_mf_xy_mat, axis=0), axis=0)
        p_temp_xy *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_temp_xz = np.nansum(np.nansum(tmp * psd_mf_xz_mat, axis=0), axis=0)
        p_temp_xz *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_temp_yy = np.nansum(np.nansum(tmp * psd_mf_yy_mat, axis=0), axis=0)
        p_temp_yy *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_temp_yz = np.nansum(np.nansum(tmp * psd_mf_yz_mat, axis=0), axis=0)
        p_temp_yz *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_temp_zz = np.nansum(np.nansum(tmp * psd_mf_zz_mat, axis=0), axis=0)
        p_temp_zz *= delta_v[i] * delta_ang * velocity[i] ** 4

        p_psd[0, 0] += p_temp_xx
        p_psd[0, 1] += p_temp_xy
        p_psd[0, 2] += p_temp_xz
        p_psd[1, 1] += p_temp_yy
        p_psd[1, 2] += p_temp_yz
        p_psd[2, 2] += p_temp_zz

        h_psd[0] = v_temp_x * velocity[i] ** 2
        h_psd[1] = v_temp_y * velocity[i] ** 2
        h_psd[2] = v_temp_z * velocity[i] ** 2

    return n_psd, v_psd, p_psd, h_psd


def psd_moments(vdf=None, sc_pot=None, **kwargs):
    """Computes moments from the FPI particle phase-space densities.

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution.

    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.

    **kwargs : dict
        Hash table of keyword arguments with :
            * energy_range : list or numpy.ndarray
                Set energy range in eV to integrate over [E_min E_max]. Energy range is applied
                to energy0 and the same elements are used for energy1 to ensure that the same
                number of points are integrated over.

            * no_sc_pot : bool
                Set to 1 to set spacecraft potential to zero. Calculates moments
                without correcting for spacecraft potential.

            * en_channels : list or numpy.ndarray
                Set energy channels to integrate over [min max]; min and max between must be
                between 1 and 32.

            * partial_moments : numpy.ndarray or xarray.DataArray
                Use a binary array (or DataArray) (pmomsarr) to select which psd points are used
                in the moments calculation. pmomsarr must be a binary array (1s and 0s,
                1s correspond to points used). Array (or data of Dataarray) must be the same size
                as vdf.data.

            * inner_electron : {"on", "off"}
                inner_electrontron potential for electron moments.

    Returns
    --------
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

    flag_de, flag_same_e, flag_inner_electron = [False] * 3

    # [eV] sc_pot + w_inner_electron for electron moments calculation; 2018-01-26, wy;
    w_inner_electron = 3.5

    vdf.data.data *= 1e12

    # Check if data is fast or burst resolution
    field_name = vdf.attrs["FIELDNAM"]

    if "brst" in field_name:
        is_brst_data = True
        print("notice : Burst resolution data is used")
    elif "brst" in field_name:
        is_brst_data = False
        print("notice : Fast resolution data is used")
    else:
        raise TypeError("Could not identify if data is fast or burst.")

    phi = vdf.phi.data
    theta_k = vdf.theta
    particle_type = vdf.attrs["species"]

    if is_brst_data:
        step_table = vdf.attrs["esteptable"]
        energy = None
        energy0 = vdf.attrs["energy0"]
        energy1 = vdf.attrs["energy1"]
        e_tmp = energy1 - energy0

        if all(e_tmp) == 0:
            flag_same_e = 1
    else:
        step_table = None
        energy = vdf.energy
        energy0 = None
        energy1 = None
        e_tmp = energy[0, :] - energy[-1, :]

        if all(e_tmp) == 0:
            energy = energy[0, :]
        else:
            raise TypeError("Could not identify if data is fast or burst.")

    # resample sc_pot to same resolution as particle distributions
    sc_pot = resample(sc_pot, vdf.time)

    int_energies = np.arange(32)

    if "energy_range" in kwargs:
        if isinstance(kwargs["energy_range"], (list, np.ndarray)) and \
                len(kwargs["energy_range"]) == 2:
            if not is_brst_data:
                energy0 = energy

            e_min_max = kwargs["energy_range"]
            start_e = bisect.bisect_left(energy0, e_min_max[0])
            stop_e = bisect.bisect_left(energy0, e_min_max[1])

            int_energies = np.arange(start_e, stop_e)
            print("notice : Using partial energy range")

    if "no_sc_pot" in kwargs:
        if isinstance(kwargs["no_sc_pot"], bool) and not kwargs["no_sc_pot"]:
            sc_pot.data = np.zeros(sc_pot.shape)
            print("notice : Setting spacecraft potential to zero")

    if "en_channels" in kwargs:
        if isinstance(kwargs["en_channels"], (list, np.ndarray)):
            int_energies = np.arange(kwargs["en_channels"][0], kwargs["en_channels"][1])

    if "partial_moments" in kwargs:
        partial_moments = kwargs["partial_moments"]
        if isinstance(partial_moments, xr.DataArray):
            partial_moments = partial_moments.data

        # Check size of partial_moments
        if partial_moments.shape == vdf.data.shape:
            sum_ones = np.sum(np.sum(np.sum(np.sum(partial_moments, axis=-1), axis=-1), axis=-1),
                              axis=-1)
            sum_zeros = np.sum(
                np.sum(np.sum(np.sum(-partial_moments + 1, axis=-1), axis=-1), axis=-1), axis=-1)

            if (sum_ones + sum_zeros) == vdf.data.size:
                print("notice : partial_moments is correct. Partial moments will be calculated")
                vdf.data = vdf.data * partial_moments
            else:
                print("notice : All values are not ones and zeros in partial_moments. Full " +
                      "moments will be calculated")
        else:
            print("notice : Size of partial_moments is wrong. Full moments will be calculated")

    if "inner_electron" in kwargs:
        inner_electron_tmp = kwargs["inner_electron"]
        if inner_electron_tmp == "on" and particle_type[0] == "e":
            flag_inner_electron = True

    # Define constants
    q_e = constants.elementary_charge
    k_b = constants.Boltzmann

    if particle_type[0] == "e":
        p_mass = constants.electron_mass
        print("notice : Particles are electrons")
    elif particle_type[0] == "i":
        p_mass = constants.proton_mass
        sc_pot.data = -1. * sc_pot.data
        print("notice : Particles are ions")
    else:
        raise ValueError("Could not identify the particle type")

    p2_psd = np.zeros((len(vdf.time), 3, 3))

    # angle between theta and phi points is 360/32 = 11.25 degrees
    delta_ang = (11.25 * np.pi / 180) ** 2

    if is_brst_data:
        phi_tr = vdf.phi
    else:
        phi_tr = phi
        phi_size = phi_tr.shape

        if phi_size[1] > phi_size[0]:
            phi_tr = phi_tr.T

    if "delta_energy_minus" in vdf.attrs and "delta_energy_plus" in vdf.attrs:
        energy_minus, energy_plus = [vdf.attrs["delta_energy_plus"], vdf.attrs["delta_energy_plus"]]

        flag_de = True
    else:
        energy_minus, energy_plus = [None, None]

    # Calculate speed widths associated with each energy channel.
    if is_brst_data:  # Burst mode energy/speed widths
        if flag_same_e and flag_de:
            energy = energy0
            energy_upper = energy + energy_plus
            energy_lower = energy - energy_minus
            v_upper = np.sqrt(2 * q_e * energy_upper / p_mass)
            v_lower = np.sqrt(2 * q_e * energy_lower / p_mass)
            delta_v = v_upper - v_lower
            delta_v0, delta_v1 = [None, None]
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

            delta_v = None

            # delta_v0(1) = delta_v0(1)*2.7
            # delta_v1(1) = delta_v1(1)*2.7

    else:  # Fast mode energy/speed widths
        energy_all = np.log10(energy)
        temp0 = 2 * energy_all[0] - energy_all[1]
        temp33 = 2 * energy_all[31] - energy_all[30]
        energy_all = np.hstack([temp0, energy_all, temp33])
        diff_en_all = np.diff(energy_all)
        energy_upper = 10 ** (np.log10(energy) + diff_en_all[1:33] / 4)
        energy_lower = 10 ** (np.log10(energy) - diff_en_all[0:32] / 4)
        v_upper = np.sqrt(2 * q_e * energy_upper / p_mass)
        v_lower = np.sqrt(2 * q_e * energy_lower / p_mass)
        delta_v = (v_upper - v_lower) * 2.0
        delta_v[0] = delta_v[0] * 2.7
        delta_v0, delta_v1 = [None, None]

    theta_k = theta_k.data[np.newaxis, :]

    # New version parallel
    # args brst :
    # (is_brst_data, flag_same_e, flag_de, step_table, energy0, delta_v0, energy1, delta_v1, qe,
    # sc_pot.data, p_mass, flag_inner_electron, w_inner_electron, phi_tr.data, theta_k,
    # int_energies, vdf.data.data, delta_ang)
    #
    # args fast :
    # (is_brst_data, flag_same_e, flag_de, energy, delta_v, qe, sc_pot.data, p_mass,
    # flag_inner_electron, w_inner_electron, phi_tr.data, theta_k, int_energies, vdf.data,
    # delta_ang)

    if is_brst_data:
        arguments = (is_brst_data, flag_same_e, flag_de, step_table, energy0, delta_v0, energy1,
                     delta_v1, q_e, sc_pot.data, p_mass, flag_inner_electron, w_inner_electron,
                     phi_tr.data, theta_k, int_energies, vdf.data.data, delta_ang)
    else:
        arguments = (is_brst_data, flag_same_e, flag_de, energy, delta_v, q_e, sc_pot.data,
                     p_mass, flag_inner_electron, w_inner_electron, phi_tr.data, theta_k,
                     int_energies, vdf.data, delta_ang)

    pool = mp.Pool(mp.cpu_count())
    res = pool.starmap(calc_moms, [(nt, arguments) for nt in range(len(vdf.time))])
    out = np.vstack(res)

    n_psd = np.array(out[:, 0], dtype="float")
    v_psd = np.vstack(out[:, 1][:])
    p_psd = np.vstack(out[:, 2][:])
    p_psd = np.reshape(p_psd, (len(n_psd), 3, 3))
    h_psd = np.vstack(out[:, 3][:])

    pool.close()

    # Compute moments in SI units
    p_psd *= p_mass
    v_psd /= n_psd[:, np.newaxis]
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
    h_psd[:, 0] -= 0.5 * v_psd[:, 0] * p_trace + 0.5 * p_mass * n_psd * v_abs2 * v_psd[:, 0]
    h_psd[:, 1] -= v_psd[:, 0] * p_psd[:, 1, 0]
    h_psd[:, 1] -= v_psd[:, 1] * p_psd[:, 1, 1]
    h_psd[:, 1] -= v_psd[:, 2] * p_psd[:, 1, 2]
    h_psd[:, 1] -= 0.5 * v_psd[:, 1] * p_trace + 0.5 * p_mass * n_psd * v_abs2 * v_psd[:, 1]
    h_psd[:, 2] -= v_psd[:, 0] * p_psd[:, 2, 0]
    h_psd[:, 2] -= v_psd[:, 1] * p_psd[:, 2, 1]
    h_psd[:, 2] -= v_psd[:, 2] * p_psd[:, 2, 2]
    h_psd[:, 2] -= 0.5 * v_psd[:, 2] * p_trace + 0.5 * p_mass * n_psd * v_abs2 * v_psd[:, 2]

    # Convert to typical units (/cc, km/s, nP, eV, and ergs/s/cm^2).
    n_psd /= 1e6
    # n_psd_e32              /= 1e6
    # n_psd_e32_phi_theta    /= 1e6
    v_psd /= 1e3
    p_psd *= 1e9
    p2_psd *= 1e9
    t_psd *= k_b / q_e
    h_psd *= 1e3

    # Construct TSeries
    n_psd = ts_scalar(vdf.time.data, n_psd)
    # n_psd_e32 = ts_scalar(vdf.time, n_psd_e32);
    # n_psd_skymap = ts_skymap(vdf.time.data, n_psd_e32_phi_theta,energy, phi.data, theta_k);
    v_psd = ts_vec_xyz(vdf.time.data, v_psd)
    p_psd = ts_tensor_xyz(vdf.time.data, p_psd)
    p2_psd = ts_tensor_xyz(vdf.time.data, p2_psd)
    t_psd = ts_tensor_xyz(vdf.time.data, t_psd)
    h_psd = ts_vec_xyz(vdf.time.data, h_psd)

    return n_psd, v_psd, p_psd, p2_psd, t_psd, h_psd
