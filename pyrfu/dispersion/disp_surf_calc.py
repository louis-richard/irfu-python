#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"


def _calc_diel(kc_, w_final, theta_, wp_e, wp_i, wc_i):
    # The elements of the dielectric tensor, using Swansons notation
    diel_s = 1 - wp_e ** 2 / (w_final ** 2 - 1) - wp_i ** 2 / (
                w_final ** 2 - wc_i ** 2)
    diel_d = -wp_e ** 2 / (w_final * (w_final ** 2 - 1))
    diel_d += wc_i * wp_i ** 2 / (w_final * (w_final ** 2 - wc_i ** 2))
    diel_p = 1 - (wp_e ** 2 + wp_i ** 2) / w_final ** 2

    n2_ = kc_ ** 2 / w_final ** 2

    diel_xx = diel_s - n2_ * np.cos(theta_) ** 2
    diel_xy = -1j * diel_d
    diel_xz = n2_ * np.cos(theta_) * np.sin(theta_)
    diel_yy = diel_s - n2_
    diel_zz = diel_p - n2_ * np.sin(theta_) ** 2
    return diel_xx, diel_xy, diel_xz, diel_yy, diel_zz


def _calc_e(diel_tensor):
    _, diel_xy, diel_xz, diel_yy, diel_zz = diel_tensor
    e_x = -diel_zz / diel_xz
    e_y = diel_xy / diel_yy * e_x
    e_z = np.ones(e_y.shape)

    e_per = np.sqrt(e_x * np.conj(e_x) + e_y * np.conj(e_y))
    e_tot = np.sqrt(e_x * np.conj(e_x) + e_y * np.conj(e_y) + e_z ** 2)
    e_pol = -2 * np.imag(e_x * np.conj(e_y)) / e_per ** 2

    return e_x, e_y, e_z, e_per, e_tot, e_pol


def _calc_b(kc_x_mat, kc_z_mat, w_final, e_x, e_y, e_z):
    b_x = -kc_z_mat * e_y / w_final
    b_y = (kc_z_mat * e_x - kc_x_mat * e_z) / w_final
    b_z = kc_x_mat * e_y / w_final

    b_par = np.sqrt(b_z * np.conj(b_z))
    b_per = np.sqrt(b_x * np.conj(b_x) + b_y * np.conj(b_y))
    b_pol = -2 * np.imag(b_x * np.conj(b_y)) / b_per ** 2
    b_tot = np.sqrt(
        b_x * np.conj(b_x) + b_y * np.conj(b_y) + b_z * np.conj(b_z))

    return b_x, b_y, b_z, b_par, b_per, b_pol, b_tot


def _calc_s(e_x, e_y, e_z, b_x, b_y, b_z):
    # Poynting flux
    s_x = e_y * np.conj(b_z) - e_z * np.conj(b_y)
    s_y = e_z * np.conj(b_x) - e_x * np.conj(b_z)
    s_z = e_x * np.conj(b_y) - e_y * np.conj(b_x)
    s_par = np.abs(s_z)
    s_tot = np.sqrt(s_x * np.conj(s_x) + s_y * np.conj(s_y)
                    + s_z * np.conj(s_z))

    return s_par, s_tot


def _calc_part2fields(wp_e, en_e, en_i, e_tot, b_tot):
    n_e = wp_e ** 2
    en_e_n = en_e * n_e
    en_i_n = en_i * n_e
    en_efield = 0.5 * e_tot ** 2
    en_bfield = 0.5 * b_tot ** 2
    ratio_part_field = (en_e_n + en_i_n) / (en_efield + en_bfield)
    return ratio_part_field


def _calc_continuity(kc_x_mat, kc_z_mat, w_final, v_ex, v_ez, v_ix, v_iz):
    dn_e_n = (kc_x_mat * v_ex + kc_z_mat * v_ez) / w_final
    dn_e_n = np.sqrt(dn_e_n * np.conj(dn_e_n))
    dn_i_n = (kc_x_mat * v_ix + kc_z_mat * v_iz) / w_final
    dn_i_n = np.sqrt(dn_i_n * np.conj(dn_i_n))
    dne_dni = dn_e_n / dn_i_n

    return dn_e_n, dn_i_n, dne_dni


def _calc_vei(m_i, wc_i, w_final, e_x, e_y, e_z):
    q_e, q_i, m_e, wc_e = [-1, 1, 1, 1]

    v_ex = 1j * q_e * (w_final * e_x - 1j * wc_e * e_y)
    v_ex /= m_e * (w_final ** 2 - wc_e ** 2)

    v_ey = 1j * q_e * (1j * wc_e * e_x + w_final * e_y)
    v_ey /= m_e * (w_final ** 2 - wc_e ** 2)

    v_ez = 1j * q_e * e_z / (m_e * w_final)

    v_ix = 1j * q_i * (w_final * e_x + 1j * wc_i * e_y)
    v_ix /= m_i * (w_final ** 2 - wc_i ** 2)

    v_iy = 1j * q_i * (-1j * wc_i * e_x + w_final * e_y)
    v_iy /= m_i * (w_final ** 2 - wc_i ** 2)

    v_iz = 1j * q_i * e_z / (m_i * w_final)

    return v_ex, v_ey, v_ez, v_ix, v_iy, v_iz


def disp_surf_calc(kc_x_max, kc_z_max, m_i, wp_e):
    r"""Calculate the cold plasma dispersion surfaces according to equation
    2.64 in Plasma Waves by Swanson (2nd ed.)

    Parameters
    ----------
    kc_x_max : float
        Max value of k_perpendicular*c/w_c.
    kc_z_max : float
        Max value of k_parallel*c/w_c.
    m_i : float
        Ion mass in terms of electron masses.
    wp_e : float
        Electron plasma frequency in terms of electron gyro frequency.

    Returns
    -------
    kx_ : numpy.ndarray
        kperpandicular*c/w_c meshgrid
    kz_ : numpy.ndarray
        kparallel*c/w_c meshgrid
    wf_ : numpy.ndarray
        Dispersion surfaces.
    extra_param : dict
        Extra parameters to plot.
    """

    # Make vectors of the wave numbers
    kc_z = np.linspace(1e-6, kc_z_max, 35)
    kc_x = np.linspace(1e-6, kc_x_max, 35)

    # Turn those vectors into matrices
    kc_x_mat, kc_z_mat = np.meshgrid(kc_x, kc_z)

    # Find some of the numbers that appear later in the calculations
    kc_ = np.sqrt(kc_x_mat ** 2 + kc_z_mat ** 2)   # Absolute value of k
    theta_ = np.arctan2(kc_x_mat, kc_z_mat)        # The angle between k and B
    wc_i = 1 / m_i                                 # The ion gyro frequency
    wp_i = wp_e / np.sqrt(m_i)                     # The ion plasma frequency
    wp_ = np.sqrt(wp_e ** 2 + wp_i ** 2)           # The total plasma frequency

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # For every k_perp and k_par, turn the dispersion relation into a
    # polynomial equation and solve it.
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # The polynomial coefficients are calculated
    pol_koeff_8 = -2 * kc_ ** 2
    pol_koeff_8 -= (1 + wc_i ** 2 + 3 * wp_ ** 2) * np.ones(kc_.shape)
    pol_koeff_6 = (2 * kc_ ** 2 + wp_ ** 2) * (1 + wc_i ** 2 + 2 * wp_ ** 2)
    pol_koeff_6 += kc_ ** 4 + (wp_ ** 2 + wc_i) ** 2
    pol_koeff_4 = -kc_ ** 4 * (1 + wc_i ** 2 + wp_ ** 2)
    pol_koeff_4 -= 2 * kc_ ** 2 * (wp_ ** 2 + wc_i) ** 2
    pol_koeff_4 -= (kc_ * wp_) ** 2 * (1 + wc_i ** 2 - wc_i) * (
                1 + np.cos(theta_) ** 2)
    pol_koeff_4 -= wp_ ** 2 * (wp_ ** 2 + wc_i) ** 2
    pol_koeff_2 = kc_ ** 4 * (wp_ ** 2 * (1 + wc_i ** 2 - wc_i) * np.cos(
        theta_) ** 2 + wc_i * (wp_ ** 2 + wc_i))
    pol_koeff_2 += kc_ ** 2 * wp_ ** 2 * wc_i * (wp_ ** 2 + wc_i) * (
                1 + np.cos(theta_) ** 2)
    pol_koeff_0 = -kc_ ** 4 * wc_i ** 2 * wp_ ** 2 * np.cos(theta_) ** 2

    w_final = np.zeros((10, len(kc_z), len(kc_x)))

    # For each k, solve the equation
    for k_z, k_x in itertools.product(range(len(kc_z)), range(len(kc_x))):
        disp_polynomial = [1, 0, pol_koeff_8[k_z, k_x], 0,
                           pol_koeff_6[k_z, k_x], 0, pol_koeff_4[k_z, k_x],
                           0, pol_koeff_2[k_z, k_x], 0, pol_koeff_0[k_z, k_x]]
        # theoretically should be real (A. Tjulin)
        w_temp = np.real(np.roots(disp_polynomial))
        # We need to sort the answers to get nice surfaces.
        w_final[:, k_z, k_x] = np.sort(w_temp)

    n2_ = kc_ ** 2 / w_final ** 2
    v_ph_c = np.sqrt(1. / n2_)
    va_c = 1 / (wp_e * np.sqrt(m_i))
    v_ph_va = v_ph_c / va_c

    diel_tensor = _calc_diel(kc_, w_final, theta_, wp_e, wp_i, wc_i)

    e_x, e_y, e_z, e_per, e_tot, e_pol = _calc_e(diel_tensor)
    e_par = (kc_x_mat * e_x + kc_z_mat * e_z) / kc_

    b_x, b_y, b_z, b_par, b_per, b_pol, b_tot = _calc_b(kc_x_mat, kc_z_mat,
                                                        w_final, e_x, e_y, e_z)

    dk_x, dk_z = [kc_x_mat[1], kc_z_mat[1]]
    dw_x, dw_z = [np.zeros(w_final.shape) for _ in range(2)]
    dw_x[:, :, 1:] = np.diff(w_final, axis=2)
    dw_z[:, 1:, :] = np.diff(w_final, axis=1)
    v_x, v_z = [dw_ / dk for dw_, dk in zip([dw_x, dw_z], [dk_x, dk_z])]

    s_par, s_tot = _calc_s(e_x, e_y, e_z, b_x, b_y, b_z)

    # Compute ion and electron velocities
    v_ex, v_ey, v_ez, v_ix, v_iy, v_iz = _calc_vei(m_i, wc_i, w_final,
                                                   e_x, e_y, e_z)

    # Ratio of parallel and perpendicular to B speed
    vepar_perp = v_ez * np.conj(v_ez)
    vepar_perp /= (v_ex * np.conj(v_ex) + v_ey * np.conj(v_ey))
    vipar_perp = v_iz * np.conj(v_iz)
    vipar_perp /= (v_ix * np.conj(v_ix) + v_iy * np.conj(v_iy))

    # Total particle speeds
    v_e2 = v_ex * np.conj(v_ex) + v_ey * np.conj(v_ey) + v_ez * np.conj(v_ez)
    v_i2 = v_ix * np.conj(v_ix) + v_iy * np.conj(v_iy) + v_iz * np.conj(v_iz)

    # Ion and electron energies
    m_e = -1
    en_e = 0.5 * m_e * v_e2
    en_i = 0.5 * m_i * v_i2

    # Ratio of particle and field energy densities
    ratio_part_field = _calc_part2fields(wp_e, en_e, en_i, e_tot, b_tot)

    # Continuity equation
    dn_e_n, dn_i_n, dne_dni = _calc_continuity(kc_x_mat, kc_z_mat, w_final,
                                               v_ex, v_ez, v_ix, v_iz)

    dn_e_n_db_b = dn_e_n / b_tot
    dn_i_n_db_b = dn_i_n / b_tot

    dn_e_n_dbpar_b = dn_e_n / b_par
    dn_i_n_dbpar_b = dn_i_n / b_par

    dn_e = dn_e_n * wp_e ** 2
    k_dot_e = e_x * kc_x_mat + e_z * kc_z_mat
    k_dot_e = np.sqrt(k_dot_e * np.conj(k_dot_e))

    # Build output dict
    extra_param = {"Degree of electromagnetism": np.log10(b_tot / e_tot),
                   "Degree of longitudinality": np.abs(e_par) / e_tot,
                   "Degree of parallelity E": e_z / e_tot,
                   "Degree of parallelity B": np.sqrt(
                       b_z * np.conj(b_z)) / b_tot,
                   "W_E/W_B": np.log10(e_tot ** 2 / b_tot ** 2),
                   "Ellipticity E": e_pol, "Ellipticity B": b_pol,
                   "E_part/E_field": np.log10(ratio_part_field),
                   "v_g": np.sqrt(v_x ** 2 + v_z ** 2),
                   "v_ph/v_a": np.log10(v_ph_va),
                   "E_e/E_i": np.log10(en_e / en_i),
                   "v_e/v_i": np.log10(np.sqrt(v_e2 / v_i2)),
                   "v_epara/v_eperp": np.log10(vepar_perp),
                   "v_ipara/v_iperp": np.log10(vipar_perp),
                   "dn_e/dn_i": np.log10(dne_dni),
                   "(dn_e/n)/ (dB/B)": np.log10(dn_e_n_db_b),
                   "(dn_i/n)/(dB/B)": np.log10(dn_i_n_db_b),
                   "(dn_i/n)/(dBpar/B)": np.log10(dn_i_n_dbpar_b),
                   "(dn_e/n)/(dB/B)": np.log10(dn_e / k_dot_e),
                   "(dn_e/n)/(dBpar /B)": np.log10(dn_e_n_dbpar_b),
                   " Spar/Stot": s_par / s_tot}

    for k, v in zip(extra_param.keys(), extra_param.values()):
        extra_param[k] = np.transpose(np.real(v), [0, 2, 1])

    kx_ = np.transpose(kc_x_mat)
    kz_ = np.transpose(kc_z_mat)
    wf_ = np.transpose(w_final, [0, 2, 1])

    return kx_, kz_, wf_, extra_param
