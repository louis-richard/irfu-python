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

from ..pyrf import resample, dec_par_perp, norm

from . import rotate_tensor


def make_model_vdf(vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz):
    """
    Make a general bi-Maxwellian distribution function based on particle moment data in the same
    format as PDist.

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

    Returns
    -------
    model_vdf : xarray.Dataset
        Distribution function in the same format as vdf.

    See also
    --------
    pyrfu.mms.calculate_epsilon : Calculates epsilon parameter using model distribution.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Load magnetic field and spacecraft potential

    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint_brst, 1)
    >>> sc_pot = mms.get_data("V_edp_brst_l2", tint_brst, 1)

    Load electron velocity distribution function

    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint_brst, 1)

    Load moments of the electron velocity distribution function

    >>> n_e = mms.get_data("ne_fpi_brst_l2", tint_brst, 1)
    >>> v_xyz_e = mms.get_data("ve_dbcs_fpi_brst_l2", tint_brst, 1)
    >>> t_xyz_e = mms.get_data("te_dbcs_fpi_brst_l2", tint_brst, 1)

    Compute model electron velocity distribution function

    >>> vdf_m_e = mms.make_model_vdf(vdf_e, b_xyz, sc_pot, n_e, v_xyz_e, t_xyz_e)

    """

    assert vdf.attrs["species"].lower() in ["i", "e"], "Invalid specie"

    # Convert to SI units
    vdf /= 1e18

    # Resample b_xyz and sc_pot to particle data resolution
    b_xyz, sc_pot = [resample(b_xyz, n_s), resample(sc_pot, n_s)]

    # Define directions based on b_xyz and v_xyz, calculate relevant temperatures
    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")  # N.B makes final distribution gyrotropic
    t_para, t_ratio = [t_xyzfac[:, 0, 0], t_xyzfac[:, 0, 0] / t_xyzfac[:, 1, 1]]

    v_para, v_perp, alpha = dec_par_perp(v_xyz, b_xyz)

    v_perp_mag, b_xyz_mag = [norm(v_perp), norm(b_xyz)]
    v_perp_dir, b_xyz_dir = [v_perp / v_perp_mag, b_xyz / b_xyz_mag]

    # Define constants
    q_e = constants.elementary_charge

    # Check whether particles are electrons or ions
    if vdf.attrs["species"].lower() == "e":
        p_mass = constants.electron_mass
        print("notice : Particles are electrons")
    else:
        p_mass = constants.proton_mass
        sc_pot.data = -1. * sc_pot.data
        print("notice : Particles are ions")

    # Convert moments to SI units
    vth_para = np.sqrt(2 * t_para * q_e / p_mass)

    v_perp_mag.data *= 1e3
    v_para.data *= 1e3
    n_s.data *= 1e6

    # Defines dimensions of array below
    n_ti, n_ph, n_th = [len(coord) for coord in [vdf.time, vdf.phi[0, :], vdf.theta]]
    n_en = len(vdf.attrs["energy0"])

    # Get energy array
    energy = vdf.energy

    # Define Cartesian coordinates
    time, phi, theta = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    r_mat = np.zeros((n_ti, n_en))

    for i in range(n_ti):
        time[i, ...] = np.outer(-np.cos(vdf.phi.data[i, :] * np.pi / 180),
                                np.sin(vdf.theta.data * np.pi / 180))
        phi[i, ...] = np.outer(-np.sin(vdf.phi.data[i, :] * np.pi / 180),
                               np.sin(vdf.theta.data * np.pi / 180))
        theta[i, ...] = np.outer(-np.ones(n_ph), np.cos(vdf.theta.data * np.pi / 180))
        r_mat[i, ...] = np.real(np.sqrt(2 * (energy[i, :] - sc_pot.data[i]) * q_e / p_mass))

    r_mat[r_mat == 0] = 0.0

    # Define rotation vectors based on B and Ve directions
    r_x, r_y, r_z = v_perp_dir.data, np.cross(b_xyz_dir.data, v_perp_dir.data), b_xyz_dir.data

    # Rotated coordinate system for computing bi-Maxwellian distribution
    x_p, y_p, z_p = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    for i in range(n_ti):
        x_p[i, ...] = time[i, ...] * r_x[i, 0] + phi[i, ...] * r_x[i, 1] + theta[i, ...] * r_x[i, 2]
        y_p[i, ...] = time[i, ...] * r_y[i, 0] + phi[i, ...] * r_y[i, 1] + theta[i, ...] * r_y[i, 2]
        z_p[i, ...] = time[i, ...] * r_z[i, 0] + phi[i, ...] * r_z[i, 1] + theta[i, ...] * r_z[i, 2]

    # Make 4D position matrix
    x_p = np.transpose(np.tile(x_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    y_p = np.transpose(np.tile(y_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    z_p = np.transpose(np.tile(z_p, [n_en, 1, 1, 1]), [1, 0, 2, 3])
    r_mat = np.transpose(np.tile(r_mat, [n_ph, n_th, 1, 1]), [2, 3, 0, 1])

    # Can use 4D matrices. Too much memory is used for my computer; too slow for large files.
    # Make 4D matrices required for distribution calculation
    # nmat = repmat(n.data,1,lengthenergy,lengthphi,lengththeta);
    # Tratmat = repmat(Trat.data,1,lengthenergy,lengthphi,lengththeta);
    # Vpmagmat = repmat(Vpmag.data,1,lengthenergy,lengthphi,lengththeta);
    # Vparmat = repmat(Vpar.data,1,lengthenergy,lengthphi,lengththeta);
    # vthparmat = repmat(vthpar.data,1,lengthenergy,lengthphi,lengththeta);

    # Calculate bi-Maxwellian distribution function
    # bimaxdist = nmat * Tratmat./(sqrt(pi^3)*vthparmat.^3);
    # bimaxdist = bimaxdist.*exp(-(xp.*rmat-Vpmagmat).^2./(vthparmat.^2).*Tratmat);
    # bimaxdist = bimaxdist.*exp(-(yp.*rmat).^2./(vthparmat.^2).*Tratmat);
    # bimaxdist = bimaxdist.*exp(-(zp.*rmat-Vparmat).^2./(vthparmat.^2));

    # Construct bi-Maxwellian distribution function
    bi_max_dist = np.zeros(r_mat.shape)

    for i in range(n_ti):
        coeff = n_s.data[i] * t_ratio.data[i] / (np.sqrt(np.pi ** 3) * vth_para.data[i] ** 3)

        bi_max_temp = coeff * np.exp(
            -(x_p[i, ...] * r_mat[i, ...] - v_perp_mag.data[i]) ** 2 / (vth_para.data[i] ** 2) *
            t_ratio.data[i])
        bi_max_temp = bi_max_temp * np.exp(
            -(y_p[i, ...] * r_mat[i, ...]) ** 2 / (vth_para.data[i] ** 2) * t_ratio.data[i])
        bi_max_temp = bi_max_temp * np.exp(
            -(z_p[i, ...] * r_mat[i, ...] - v_para.data[i]) ** 2 / (vth_para.data[i] ** 2))

        bi_max_dist[i, ...] = bi_max_temp

    # Make modelPDist file for output
    model_vdf = vdf
    model_vdf.data = bi_max_dist
    model_vdf *= 1e18

    return model_vdf
