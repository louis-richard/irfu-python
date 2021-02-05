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


def dispersion_surface_calc(kc_x_max=None, kc_z_max=None, m_i=None, w_pe=None):
    """Calculates the cold plasma dispersion surfaces according to equation
    2.63 in [15]_, and puts them in the matrix W. Additional parameters that
    are needed in dispersion_surface is returned as extra_param.

    Parameters
    ----------
    kc_x_max : float or int
        Max value of k_perpendicular*c/w_c.

    kc_z_max : float or int
        Max value of k_parallel*c/w_c.

    m_i : float
        Ion mass in terms of electron masses.

    w_pe : float
        Electron plasma frequency in terms of electron gyro frequency.

    Returns
    -------
    w_final : numpy.ndarray
        to fill.

    extra_param : numpy.ndarray
        to fill.

    Notes
    -----
    This function is essential for dispersion_surface to work.

    References
    ----------
    .. [15] Swanson, D. G. (2003), Plasma waves, 2nd ed., Institute of Physics
            Pub, doi: https://www.taylorfrancis.com/books/9780367802721

    """

    assert kc_x_max is not None and isinstance(kc_x_max, (float, int))
    assert kc_z_max is not None and isinstance(kc_z_max, (float, int))

    if isinstance(kc_x_max, float):
        kc_x_max = int(kc_x_max)

    if isinstance(kc_z_max, float):
        kc_z_max = int(kc_z_max)

    # First get rid of those annoying "division by zero" - warnings

    # Make vectors of the wave numbers
    kc_x, kc_z = [np.linspace(1e-6, kc_m, 35) for kc_m in [kc_x_max, kc_z_max]]

    # Turn those vectors into matrices
    [kc_x_mat, kc_z_mat] = np.meshgrid(kc_x, kc_z)

    # Find some of the numbers that appear later in the calculations
    k_mag = np.sqrt(kc_x_mat ** 2 + kc_z_mat ** 2)  # Absolute value of k
    theta = np.arctan2(kc_x_mat, kc_z_mat)          # The angle between k and B

    w_ci = 1 / m_i                                  # The ion gyro freq.
    w_pi = w_pe / np.sqrt(m_i)                      # The ion plasma freq.
    w_p = np.sqrt(w_pe ** 2 + w_pi ** 2)            # The total plasma freq.

    # To speed up the program somewhat introduce these
    kc2 = k_mag ** 2
    kc4 = kc2 ** 2
    w_ci2 = w_ci ** 2
    w_p2 = w_p ** 2
    w_extra = w_p2 + w_ci
    w_extra2 = w_extra ** 2
    cos_theta2 = np.cos(theta) ** 2

    # For every k_perp and k_par, turn the dispersion relation into a
    # polynomial eq. and solve it. The polynomial coefficients are calculated
    pol_k_coeff8 = -(2 * kc2 + 1 + w_ci2 + 3 * w_p2)
    pol_k_coeff6 = (kc4 + (2 * kc2 + w_p2) * (1 + w_ci2 + 2 * w_p2) + w_extra2)
    pol_k_coeff4 = -(kc4 * (1 + w_ci2 + w_p2)
                     + 2 * kc2 * w_extra2 + kc2 * w_p2 * (1 + w_ci2 - w_ci)
                     * (1 + cos_theta2) + w_p2 * w_extra2)
    pol_k_coeff2 = (kc4 * (w_p2 * (1 + w_ci2 - w_ci) * cos_theta2
                           + w_ci * w_extra)
                    + kc2 * w_p2 * w_ci * w_extra * (1 + cos_theta2))
    pol_k_coeff0 = -kc4 * w_ci2 * w_p2 * cos_theta2

    # For each k, solve the equation
    w_final = np.zeros((10, 35, 35))
    for k_z in range(len(kc_z)):
        for k_x in range(len(kc_x)):
            dispersion_polynomial = [1, 0, pol_k_coeff8[k_z, k_x], 0,
                                     pol_k_coeff6[k_z, k_x], 0,
                                     pol_k_coeff4[k_z, k_x], 0,
                                     pol_k_coeff2[k_z, k_x], 0,
                                     pol_k_coeff0[k_z, k_x]]
            # theoretically should be real (A. Tjulin)
            w_temp = np.real(np.roots(dispersion_polynomial))
            # We need to sort the answers to get nice surfaces.
            w_final[:, k_z, k_x] = np.sort(w_temp)

    # Now we have solved the dispersion relation. Let us find some other
    # interesting parameters in this context.
    # The elements of the dielectric tensor, using Swanson's notation
    dielectric_s = 1 - w_pe ** 2 / (w_final ** 2 - 1) \
                   - w_pi ** 2 / (w_final ** 2 - w_ci2)
    dielectric_p = 1 - (w_pe ** 2 + w_pi ** 2) / (w_final ** 2)
    dielectric_d = -w_pe ** 2 / (w_final * (w_final ** 2 - 1)) \
                   + w_ci * w_pi ** 2 / (w_final * (w_final ** 2 - w_ci2))

    # The rest of this function is not cleaned yet. The calculations could
    # probably be much shorter ! Use tile instead.
    kc2_mat, k_x_mat, k_z_mat = [np.zeros((10, 35, 35)) for _ in range(3)]
    theta_mat = np.zeros((10, 35, 35))

    for i in range(10):
        kc2_mat[i, :, :] = kc2
        k_x_mat[i, :, :] = kc_x_mat
        k_z_mat[i, :, :] = kc_z_mat
        theta_mat[i, :, :] = theta

    n_2 = kc2_mat / (w_final ** 2)
    v_phase_to_c = np.sqrt(1 / n_2)
    va_to_c = 1 / (w_pe * np.sqrt(m_i))
    v_phase_to_va = v_phase_to_c / va_to_c

    # dielectric_xx = dielectric_s - n2 * np.cos(theta_mat) ** 2
    dielectric_xy = -1j * dielectric_d
    dielectric_xz = n_2 * np.cos(theta_mat) * np.sin(theta_mat)
    dielectric_yy = dielectric_s - n_2
    dielectric_zz = dielectric_p - n_2 * np.sin(theta_mat) ** 2

    e_x = - dielectric_zz / dielectric_xz
    e_y = dielectric_xy / dielectric_yy * e_x
    e_z = 1
    
    e_perp = np.sqrt(e_x * np.conj(e_x) + e_y * np.conj(e_y))
    e_tot = np.sqrt(e_x * np.conj(e_x) + e_y * np.conj(e_y) + 1)
    e_par_k = (k_x_mat * e_x + k_z_mat * e_z) / np.sqrt(kc2_mat)
    e_polar = -2 * np.imag(e_x * np.conj(e_y)) / e_perp ** 2

    b_x = -k_z_mat * e_y / w_final
    b_y = (k_z_mat * e_x - k_x_mat * e_z) / w_final
    b_z = k_x_mat * e_y / w_final
    b_tot = np.sqrt(b_x * np.conj(b_x) + b_y * np.conj(b_y)
                    + b_z * np.conj(b_z))
    b_par = np.sqrt(b_z * np.conj(b_z))
    b_perp = np.sqrt(b_x * np.conj(b_x) + b_y * np.conj(b_y))
    b_polar = -2 * np.imag(b_x * np.conj(b_y)) / (b_perp ** 2)

    # Poynting flux
    s_x = e_y * np.conj(b_z) - e_z * np.conj(b_y)
    s_y = e_z * np.conj(b_x) - e_x * np.conj(b_z)
    s_z = e_x * np.conj(b_y) - e_y * np.conj(b_x)
    s_par = abs(s_z)
    s_tot = np.sqrt(s_x * np.conj(s_x) + s_y * np.conj(s_y)
                    + s_z * np.conj(s_z))

    temp = len(kc_x)
    dk_x = kc_x[1]
    dk_z = kc_z[1]
    dw_x = np.diff(w_final, 1, 3)
    dw_z = np.diff(w_final, 1, 2)
    dw_x[1, temp, temp] = 0
    dw_z[1, temp, temp] = 0
    v_x = dw_x / dk_x
    v_z = dw_z / dk_z

    # Compute ion and electron velocities
    q_e, q_i, m_e, w_ce = [-1, 1, 1, 1]

    v_x_e = 1j * q_e / (m_e * (w_final ** 2 - w_ce ** 2)) \
            * (w_final * e_x - 1j * w_ce * e_y)
    v_y_e = 1j * q_e / (m_e * (w_final ** 2 - w_ce ** 2)) \
            * (w_final * e_y + 1j * w_ce * e_x)
    v_z_e = 1j * q_e * e_z / (m_e * w_final)

    v_x_i = 1j * q_i / (m_i * (w_final ** 2 - w_ci ** 2)) \
            * (w_final * e_x + 1j * w_ci * e_y)
    v_y_i = 1j * q_i / (m_i * (w_final ** 2 - w_ci ** 2)) \
            * (w_final * e_y - 1j * w_ci * e_x)
    v_z_i = 1j * q_i * e_z / (m_i * w_final)

    # Ratio of parallel and perpendicular to B speed
    v_par_o_perp_e = v_z_e * np.conj(v_z_e) \
                     / (v_x_e * np.conj(v_x_e) + v_y_e * np.conj(v_y_e))
    v_par_o_perp_i = v_z_i * np.conj(v_z_i) \
                     / (v_x_i * np.conj(v_x_i) + v_y_i * np.conj(v_y_i))

    # Total particle speeds
    v_e2 = v_x_e * np.conj(v_x_e) + v_y_e * np.conj(v_y_e) \
           + v_z_e * np.conj(v_z_e)
    v_i2 = v_x_i * np.conj(v_x_i) + v_y_i * np.conj(v_y_i) \
           + v_z_i * np.conj(v_z_i)

    # Ion and electron energies
    e_e = 0.5 * m_e * v_e2
    e_i = 0.5 * m_i * v_i2

    # Ratio of particle and field energy densities
    n_e = w_pe ** 2
    e_en = e_e * n_e
    e_in = e_i * n_e
    en_e = 0.5 * e_tot ** 2
    en_b = 0.5 * b_tot ** 2
    ratio_pf = (e_en + e_in) / (en_e + en_b)

    # Continuity equation
    dn_eon = (k_x_mat * v_x_e + k_z_mat * v_z_e) / w_final
    dn_eon = np.sqrt(dn_eon * np.conj(dn_eon))
    dn_ion = (k_x_mat * v_x_i + k_z_mat * v_z_i) / w_final
    dn_ion = np.sqrt(dn_ion * np.conj(dn_ion))
    dn_e_dn_i = dn_eon / dn_ion
    
    dn_eon_o_db_o_b = dn_eon / b_tot
    dn_ion_o_db_o_b = dn_ion / b_tot
    
    dn_eon_o_db_par_o_b = dn_eon / b_par
    dn_ion_o_db_par_o_b = dn_ion / b_par

    dn_e = dn_eon * w_pe ** 2
    k_dot_e = e_x * k_x_mat + e_z * k_z_mat
    k_dot_e = np.sqrt(k_dot_e * np.conj(k_dot_e))

    # Degree of electromagnetism
    extra_param = np.zeros((20, 10, 35, 35))
    extra_param[1, ...] = np.log10(b_tot / e_tot)

    # Degree of longitudinally
    extra_param[2, ...] = abs(e_par_k) / e_tot

    # Degree of parallelism
    extra_param[3, ...] = e_z / e_tot

    # Degree of parallelism
    extra_param[4, ...] = np.sqrt(b_z * np.conj(b_z)) / b_tot

    # Value of the group vel.
    extra_param[5, ...] = np.sqrt(v_x ** 2 + v_z ** 2)

    # Ellipticity
    extra_param[6, ...] = e_polar

    # Degree of electromagnetism
    extra_param[7, ...] = np.log10(e_tot ** 2 / b_tot ** 2)

    # Ratio of electron to ion energy
    extra_param[8, ...] = np.log10(en_e / e_i)

    # Ratio of electron to ion velocity fluctuations
    extra_param[9, ...] = np.log10(np.sqrt(v_e2 / v_i2))

    # Ratio of particle to field energy densities
    extra_param[10, ...] = np.log10(ratio_pf)

    # Ellipticity based on B
    extra_param[11, ...] = b_polar

    # Phase speed divided by Alfven speed
    extra_param[12, ...] = np.log10(v_phase_to_va)

    # Ratio of parallel to perpendicular electron speed
    extra_param[13, ...] = np.log10(v_par_o_perp_e)

    # Ratio of parallel to perpendicular ion speed
    extra_param[14, ...] = np.log10(v_par_o_perp_i)

    #
    extra_param[15, ...] = np.log10(e_en / (en_e + en_b))

    # Ratio of electron to ion density perturbations
    extra_param[16, ...] = np.log10(dn_e_dn_i)

    # (dn_e / n) / (dB / B)
    extra_param[17, ...] = np.log10(dn_eon_o_db_o_b)

    # (dn_i / n) / (dB / B)
    extra_param[18, ...] = np.log10(dn_ion_o_db_o_b)

    # (dn_e / n) / (dB_par / B)
    extra_param[19, ...] = np.log10(dn_eon_o_db_par_o_b)

    # (dn_i / n) / (dB_par / B)
    extra_param[20, ...] = np.log10(dn_ion_o_db_par_o_b)

    # (dn_i / n) / (dB / B)
    extra_param[21, ...] = np.log10(dn_e / k_dot_e)

    # S_par / S_tot
    extra_param[22, ...] = s_par / s_tot

    return w_final, extra_param
