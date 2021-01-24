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

from astropy import constants

from . import rotate_tensor
from ..pyrf import resample, dec_par_perp, norm


def make_model_vdf(vdf, b_xyz, sc_pot, n, v_xyz, t_xyz):
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

    n : xarray.DataArray
        Time series of the number density.

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

    # Convert to SI units
    vdf /= 1e18

    # Resample b_xyz and sc_pot to particle data resolution
    b_xyz, sc_pot = [resample(b_xyz, n), resample(sc_pot, n)]

    # Define directions based on b_xyz and v_xyz, calculate relevant temperatures
    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")  # N.B makes final distribution gyrotropic
    t_para, t_ratio = [t_xyzfac[:, 0, 0], t_xyzfac[:, 0, 0] / t_xyzfac[:, 1, 1]]

    v_para, v_perp, alpha = dec_par_perp(v_xyz, b_xyz)

    v_perp_mag, b_xyz_mag = [norm(v_perp), norm(b_xyz)]
    v_perp_dir, b_xyz_dir = [v_perp / v_perp_mag, b_xyz / b_xyz_mag]

    # Define constants
    qe = constants.e.value

    # Check whether particles are electrons or ions
    if vdf.attrs["species"].lower() == "e":
        p_mass = constants.m_e.value
        print("notice : Particles are electrons")
    elif vdf.attrs["species"].lower() == "i":
        p_mass = constants.m_p.value
        sc_pot.data = -sc_pot.data
        print("notice : Particles are ions")
    else:
        raise ValueError("Invalid specie")

    # Convert moments to SI units
    vth_para = np.sqrt(2 * t_para * qe / p_mass)

    v_perp_mag.data *= 1e3
    v_para.data *= 1e3
    n.data *= 1e6

    # Defines dimensions of array below
    n_ti = len(vdf.time)
    n_ph = len(vdf.phi[0, :])
    n_th = len(vdf.theta)
    n_en = len(vdf.attrs["energy0"])

    # Get energy array
    energy = vdf.energy

    # Define Cartesian coordinates
    x, y, z = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    r = np.zeros((n_ti, n_en))

    for ii in range(n_ti):
        x[ii, ...] = np.outer(
            -np.cos(vdf.phi.data[ii, :] * np.pi / 180) * np.sin(vdf.theta.data * np.pi / 180))
        y[ii, ...] = np.outer(
            -np.sin(vdf.phi.data[ii, :] * np.pi / 180) * np.sin(vdf.theta.data * np.pi / 180))
        z[ii, ...] = np.outer(-np.ones(n_ph) * np.cos(vdf.theta.data * np.pi / 180))
        r[ii, ...] = np.real(np.sqrt(2 * (energy[ii, :] - sc_pot.data[ii]) * qe / p_mass))

    r[r == 0] = 0.0

    # Define rotation vectors based on B and Ve directions
    r_x, r_y, r_z = v_perp_dir.data, np.cross(b_xyz_dir.data, v_perp_dir.data), b_xyz_dir.data

    # Rotated coordinate system for computing bi-Maxwellian distribution
    x_p, y_p, z_p = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    for ii in range(n_ti):
        x_p[ii, ...] = x[ii, ...] * r_x[ii, 0] + y[ii, ...] * r_x[ii, 1] + z[ii, ...] * r_x[ii, 2]
        y_p[ii, ...] = x[ii, ...] * r_y[ii, 0] + y[ii, ...] * r_y[ii, 1] + z[ii, ...] * r_y[ii, 2]
        z_p[ii, ...] = x[ii, ...] * r_z[ii, 0] + y[ii, ...] * r_z[ii, 1] + z[ii, ...] * r_z[ii, 2]

    # Make 4D position matrix
    x_p = np.tile(x_p, [n_en, 1, 1, 1])
    x_p = np.transpose(x_p, [1, 0, 2, 3])

    y_p = np.tile(x_p, [n_en, 1, 1, 1])
    y_p = np.transpose(y_p, [1, 0, 2, 3])

    z_p = np.tile(x_p, [n_en, 1, 1, 1])
    z_p = np.transpose(z_p, [1, 0, 2, 3])

    r_mat = np.tile(r, [n_ph, n_th, 1, 1])
    r_mat = np.transpose(r_mat, [2, 3, 0, 1])

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

    for ii in range(n_ti):
        coeff = n.data[ii] * t_ratio.data[ii] / (np.sqrt(np.pi ** 3) * vth_para.data[ii] ** 3)

        bi_max_temp = coeff * np.exp(
            -(x_p[ii, ...] * r_mat[ii, ...] - v_perp_mag.data[ii]) ** 2 / (vth_para.data[ii] ** 2) *
            t_ratio.data[ii])
        bi_max_temp = bi_max_temp * np.exp(
            -(y_p[ii, ...] * r_mat[ii, ...]) ** 2 / (vth_para.data[ii] ** 2) * t_ratio.data[ii])
        bi_max_temp = bi_max_temp * np.exp(
            -(z_p[ii, ...] * r_mat[ii, ...] - v_para.data[ii]) ** 2 / (vth_para.data[ii] ** 2))

        bi_max_dist[ii, ...] = bi_max_temp

    # Make modelPDist file for output
    model_vdf = vdf
    model_vdf.data = bi_max_dist
    model_vdf *= 1e18

    return model_vdf
