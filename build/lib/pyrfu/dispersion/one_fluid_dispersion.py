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

"""one_fluid_dispersion.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from scipy import constants, optimize


def _disprel(w, *args):
    k, theta = args[0:2]
    v_a, c_s = args[2:4]
    wc_e, wc_p = args[4:6]

    theta = np.deg2rad(theta)
    l_00 = 1
    l_01 = - w ** 2 / (k ** 2 * v_a ** 2)
    l_02 = - w ** 2 / (wc_e * wc_p)
    l_03 = k ** 2 * np.sin(theta) ** 2 / ((w / c_s) ** 2 - k ** 2)
    l_0_ = (l_00 + l_01 + l_02 + l_03)

    l_10 = np.cos(theta) ** 2
    l_11 = - w ** 2 / (k ** 2 * v_a ** 2)
    l_12 = - w ** 2 / (wc_e * wc_p)
    l_1_ = l_10 + l_11 + l_12

    r_0_ = w ** 2 * np.cos(theta) ** 2 / wc_p ** 2

    disprel = l_0_ * l_1_ - r_0_

    return disprel


def one_fluid_dispersion(b_0, theta, ions, electrons, n_k: int = 100):
    r"""Solves the one fluid dispersion relation.

    Parameters
    ----------
    b_0 : float
        Magnetic field

    theta : float
        The angle of propagation of the wave with respect to the magnetic
        field, :math:`\cos^{-1}(k_z / k)`

    ions : dict
        Hash table with n : number density, t: temperature, gamma:
        polytropic index.

    electrons : dict
        Hash table with n : number density, t: temperature, gamma:
        polytropic index.

    n_k : int, optional
        Number of wavenumbers.

    Returns
    -------
    wc_1 : xarray.DataArray
        1st root

    wc_2 : xarray.DataArray
        2nd root

    wc_3 : xarray.DataArray
        3rd root

    """

    keys = ["n", "t", "gamma"]
    n_p, t_p, gamma_p = [ions[k] for k in keys]
    n_e, t_e, gamma_e = [electrons[k] for k in keys]

    q_e = constants.elementary_charge
    m_e = constants.electron_mass
    m_p = constants.proton_mass
    ep_0 = constants.epsilon_0
    mu_0 = constants.mu_0

    wc_e = q_e * b_0 / m_e
    wc_p = q_e * b_0 / m_p

    wp_e = np.sqrt(q_e ** 2 * n_e / (ep_0 * m_e))
    wp_p = np.sqrt(q_e ** 2 * n_p / (ep_0 * m_p))

    v_p = np.sqrt(q_e * t_p / m_p)
    v_e = np.sqrt(q_e * t_e / m_e)

    v_a = b_0 / np.sqrt(mu_0 * n_p * m_p)
    c_s = np.sqrt((gamma_e * q_e * t_e + gamma_p * q_e * t_p) / (m_e + m_p))

    k_vec = np.linspace(2e-7, 1.0e-4, n_k)

    wc_1, wc_2, wc_3 = [np.zeros(len(k_vec)) for _ in range(3)]

    for i, k in enumerate(k_vec):
        if i < 10:
            guess_w1 = v_a * k * 1.50
            guess_w2 = v_a * k * 0.70
            guess_w3 = c_s * k * 0.99
        else:
            guess_w1 = wc_1[i - 1] + (wc_1[i - 1] - wc_1[i - 2])
            guess_w2 = wc_2[i - 1] + (wc_2[i - 1] - wc_2[i - 2])
            guess_w3 = wc_3[i - 1] + (wc_3[i - 1] - wc_3[i - 2])

        arguments = (k, theta, v_a, c_s, wc_e, wc_p)
        wc_1[i] = optimize.fsolve(_disprel, guess_w1, args=arguments)[0]
        wc_2[i] = optimize.fsolve(_disprel, guess_w2, args=arguments)[0]
        wc_3[i] = optimize.fsolve(_disprel, guess_w3, args=arguments)[0]

    attrs = {"wc_e": wc_e, "wc_p": wc_p, "wp_e": wp_e, "wp_p": wp_p,
             "v_p": v_p, "v_e": v_e, "v_a": v_a, "c_s": c_s}

    wc_1 = xr.DataArray(wc_1, coords=[k_vec], dims=["k"], attrs=attrs)
    wc_2 = xr.DataArray(wc_2, coords=[k_vec], dims=["k"], attrs=attrs)
    wc_3 = xr.DataArray(wc_3, coords=[k_vec], dims=["k"], attrs=attrs)

    return wc_1, wc_2, wc_3
