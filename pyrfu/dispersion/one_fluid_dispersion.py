#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.polynomial import polynomial
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _disprel(w, *args):
    assert len(args) == 6, "not enougth arguments"
    k, theta, v_a, c_s, wc_e, wc_p = args

    theta = np.deg2rad(theta)
    l_00 = 1
    l_01 = -(w**2) / (k**2 * v_a**2)
    l_02 = -(w**2) / (wc_e * wc_p)
    l_03 = k**2 * np.sin(theta) ** 2 / ((w / c_s) ** 2 - k**2)
    l_0_ = l_00 + l_01 + l_02 + l_03

    l_10 = np.cos(theta) ** 2
    l_11 = -(w**2) / (k**2 * v_a**2)
    l_12 = -(w**2) / (wc_e * wc_p)
    l_1_ = l_10 + l_11 + l_12

    r_0_ = w**2 * np.cos(theta) ** 2 / wc_p**2

    disprel = l_0_ * l_1_ - r_0_

    return disprel


def _disprel_roots(k, theta, v_a, c_s, wc_e, wc_p):
    # Multiplied by D = w^2 / c_s^2 - k^2, the dispersion relation is a cubic
    # in x = w^2, solved here in x / wc_p^2 for a better conditioning:
    # [(1 - a x) D + k^2 sin^2(theta)] (cos^2(theta) - a x) - x cos^2(theta) D
    # / wc_p^2 = 0, with a = 1 / (k^2 v_a^2) + 1 / (wc_e wc_p)
    cos2, sin2 = [np.cos(np.deg2rad(theta)) ** 2, np.sin(np.deg2rad(theta)) ** 2]
    a_ = (1 / (k**2 * v_a**2) + 1 / (wc_e * wc_p)) * wc_p**2
    d_ = [-(k**2), wc_p**2 / c_s**2]

    l_0 = polynomial.polyadd(polynomial.polymul([1, -a_], d_), [k**2 * sin2])
    l_1 = [cos2, -a_]
    r_0 = polynomial.polymul([0, cos2], d_)
    coeffs = polynomial.polysub(polynomial.polymul(l_0, l_1), r_0)

    # Real roots, up to round-off imaginary parts
    x_roots = np.real(polynomial.polyroots(coeffs)) * wc_p**2
    w_roots = np.sort(np.sqrt(x_roots))[::-1]

    return w_roots


def one_fluid_dispersion(b_0, theta, ions, electrons, n_k: int = 100):
    r"""Solves the one fluid dispersion relation.

    Parameters
    ----------
    b_0 : float
        Magnetic field

    theta : float
        The angle of propagation of the wave with respect to the magnetic
        field, :math:`\cos^{-1}(k_z / k)`, in degrees.

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
        Largest root (fast/whistler branch).

    wc_2 : xarray.DataArray
        Intermediate root (Alfven/ion cyclotron branch).

    wc_3 : xarray.DataArray
        Smallest root (slow branch).

    Notes
    -----
    The three branches are the roots of a cubic polynomial in
    :math:`\omega^2`, sorted so that wc_1 >= wc_2 >= wc_3 at every k. At
    small angles, the sound wave crosses the other branches, so it changes
    from one output to another at the crossings.

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

    wp_e = np.sqrt(q_e**2 * n_e / (ep_0 * m_e))
    wp_p = np.sqrt(q_e**2 * n_p / (ep_0 * m_p))

    v_p = np.sqrt(q_e * t_p / m_p)
    v_e = np.sqrt(q_e * t_e / m_e)

    v_a = b_0 / np.sqrt(mu_0 * n_p * m_p)
    c_s = np.sqrt((gamma_e * q_e * t_e + gamma_p * q_e * t_p) / (m_e + m_p))

    k_vec = np.linspace(2e-7, 1.0e-4, n_k)

    w_roots = np.stack(
        [_disprel_roots(k, theta, v_a, c_s, wc_e, wc_p) for k in k_vec],
    )
    wc_1, wc_2, wc_3 = w_roots.T

    attrs = {
        "wc_e": wc_e,
        "wc_p": wc_p,
        "wp_e": wp_e,
        "wp_p": wp_p,
        "v_p": v_p,
        "v_e": v_e,
        "v_a": v_a,
        "c_s": c_s,
    }

    wc_1 = xr.DataArray(wc_1, coords=[k_vec], dims=["k"], attrs=attrs)
    wc_2 = xr.DataArray(wc_2, coords=[k_vec], dims=["k"], attrs=attrs)
    wc_3 = xr.DataArray(wc_3, coords=[k_vec], dims=["k"], attrs=attrs)

    return wc_1, wc_2, wc_3
