#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plasma_calc.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from astropy import constants

from .resample import resample


def plasma_calc(b=None, t_i=None, t_e=None, n_i=None, n_e=None):
    """
    Computes plasma parameters including characteristic length and time scales

    Parameters :
        b : DataArray
            Time series of the magnetic field [nT]

        t_i : DataArray
            Time series of the ions temperature [eV]

        t_e : DataArray
            Time series of the electrons temperature [eV]

        n_i : DataArray
            Time series of the ions number density [cm^{-3}]

        n_e : DataArray
            Time series of the electrons number density [cm^{-3}]

    Returns :
        out : Dataset:
            Dataset of the plasma parameters :
                * time : DataArray
                    Time

                * Wpe : DataArray
                    Time series of the electron plasma frequency [rad.s^{-1}]

                * Fpe : DataArray
                    Time series of the electron plasma frequency [Hz]

                * Wce : DataArray
                    Time series of the electron cyclotron frequency [rad.s^{-1}]

                * Fce : DataArray
                    Time series of the electron cyclotron frequency [Hz]

                * Wpp : DataArray
                    Time series of the ion plasma frequency [rad.s^{-1}]

                * Fpp : DataArray
                    Time series of the ion plasma frequency [Hz]

                * Fcp : DataArray
                    Time series of the ion cyclotron frequency [Hz]

                * Fuh : DataArray
                    Time series of the upper hybrid frequency [Hz]

                * Flh : DataArray
                    Time series of the lower hybrid frequency [Hz]

                * Va : DataArray
                    Time series of the Alfvèn velocity (ions) [m.s^{-1}]

                * Vae : DataArray
                    Time series of the Alfvèn velocity (electrons) [m.s^{-1}]

                * Vte : DataArray
                    Time series of the electron thermal velocity [m.s^{-1}]

                * Vtp : DataArray
                    Time series of the electron thermal velocity [m.s^{-1}]

                * Vts : DataArray
                    Time series of the sound speed [m.s^{-1}]

                * gamma_e : DataArray
                    Time series of the electron Lorentz factor

                * gamma_p : DataArray
                    Time series of the electron Lorentz factor

                * Le : DataArray
                    Time series of the electron inertial length [m]

                * Li : DataArray
                    Time series of the electron inertial length [m]

                * Ld : DataArray
                    Time series of the Debye length [m]

                * Nd : DataArray
                    Time series of the number of electrons in the Debye sphere

                * Roe : DataArray
                    Time series of the electron Larmor radius [m]

                * Rop : DataArray
                    Time series of the ion Larmor radius [m]

                * Ros : DataArray
                    Time series of the length associated to the sound speed [m]

    Example :
        >>> from pyrfu import mms, pyrf
        >>> # Time interval
        >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
        >>> # Spacecraft index
        >>> mms_id = 1
        >>> # Load magnetic field, ion/electron temperature and number density
        >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
        >>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)
        >>> t_xyz_e = mms.get_data("Te_gse_fpi_fast_l2", tint, mms_id)
        >>> n_i = mms.get_data("Ni_fpi_fast_l2", tint, mms_id)
        >>> n_e = mms.get_data("Ne_fpi_fast_l2", tint, mms_id)
        >>> # Compute scalar temperature
        >>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")
        >>> t_xyzfac_e = mms.rotate_tensor(t_xyz_e, "fac", b_xyz, "pp")
        >>> t_i = pyrf.trace(t_xyzfac_i)
        >>> t_e = pyrf.trace(t_xyzfac_e)
        >>> # Compute plasma parameters
        >>> pparam = pyrf.plasma_calc(b_xyz, t_i, t_e, n_i, n_e)

    """

    if b is None or t_i is None or t_e is None or n_i is None or n_e is None:
        raise ValueError("plasma_calc requires at least 5 arguments")

    if not isinstance(b, xr.DataArray):
        raise TypeError("Inputs must be DataArrays")

    if not isinstance(t_i, xr.DataArray):
        raise TypeError("Inputs must be DataArrays")

    if not isinstance(t_e, xr.DataArray):
        raise TypeError("Inputs must be DataArrays")

    if not isinstance(n_i, xr.DataArray):
        raise TypeError("Inputs must be DataArrays")

    if not isinstance(n_e, xr.DataArray):
        raise TypeError("Inputs must be DataArrays")

    # Get constants
    e, mu0, c, eps0 = [constants.e.value, constants.mu0.value, constants.c.value, constants.eps0.value]
    m_p, m_e, mp_me = [constants.m_p.value, constants.m_e.value, constants.m_p.value / constants.m_e.value]

    # Resample all variables with respect to the magnetic field
    n_t = len(b)

    if len(t_i) != n_t:
        t_i = resample(t_i, b).data

    if len(t_e) != n_t:
        t_e = resample(t_e, b).data

    if len(n_i) != n_t:
        n_i = resample(n_i, b).data

    if len(n_e) != n_t:
        n_e = resample(n_e, b).data

    # Transform number density and magnetic field to SI units
    n_i, n_e = [1e6 * n_i, 1e6 * n_e]

    if b.ndim == 2:
        b_si = 1e-9 * np.linalg.norm(b, axis=1)
    else:
        b_si = 1e-9 * np.linalg.norm(b, axis=1)

    w_pe = np.sqrt(n_e * e ** 2 / (m_e * eps0)) 	# rad/s
    w_ce = e * b_si / m_e   						# rad/s
    w_pp = np.sqrt(n_i * e ** 2 / (m_p * eps0))

    v_a = b_si / np.sqrt(mu0 * n_i * m_p)

    v_ae = b_si / np.sqrt(mu0 * n_e * m_e)
    v_te = c * np.sqrt(1 - 1 / (t_e * e / (m_e * c ** 2) + 1) ** 2) 	# m/s (relativ. correct)
    v_tp = c * np.sqrt(1 - 1 / (t_i * e / (m_p * c ** 2) + 1) ** 2)     # m/s
    v_ts = np.sqrt((t_e * e + 3 * t_i * e) / m_p)              			# Sound speed formula (F. Chen, Springer 1984).

    gamma_e = 1 / np.sqrt(1 - (v_te / c) ** 2)
    gamma_p = 1 / np.sqrt(1 - (v_tp / c) ** 2)

    l_e = c / w_pe
    l_i = c / w_pp
    l_d = v_te / (w_pe * np.sqrt(2)) 				# Debye length scale, sqrt(2) needed because of Vte definition
    n_d = l_d * eps0 * m_e * v_te ** 2 / e ** 2 	# number of e- in Debye sphere

    f_pe = w_pe / (2 * np.pi) 				# Hz
    f_ce = w_ce / (2 * np.pi)
    f_uh = np.sqrt(f_ce ** 2 + f_pe ** 2)
    f_pp = w_pp / (2 * np.pi)
    f_cp = f_ce / mp_me
    f_lh = np.sqrt(f_cp * f_ce / (1 + f_ce ** 2 / f_pe ** 2) + f_cp ** 2)

    rho_e = m_e * c / (e * b_si) * np.sqrt(gamma_e ** 2 - 1) 	# m, relativistically correct
    rho_p = m_p * c / (e * b_si) * np.sqrt(gamma_p ** 2 - 1) 	# m, relativistically correct
    rho_s = v_ts / (f_cp * 2 * np.pi) 							# m

    out = xr.Dataset({"time": b.time.data, "w_pe": (["time"], w_pe), "w_ce": (["time"], w_ce), "w_pp": (["time"], w_pp),
                      "v_a": (["time"], v_a), "v_ae": (["time"], v_ae), "v_te": (["time"], v_te),
                      "v_tp": (["time"], v_tp), "v_ts": (["time"], v_ts), "gamma_e": (["time"], gamma_e),
                      "gamma_p": (["time"], gamma_p), "l_e": (["time"], l_e), "l_i": (["time"], l_i),
                      "l_d": (["time"], l_d), "n_d": (["time"], n_d), "f_pe": (["time"], f_pe),
                      "f_ce": (["time"], f_ce), "f_uh": (["time"], f_uh), "f_pp": (["time"], f_pp),
                      "f_cp": (["time"], f_cp), "f_lh": (["time"], f_lh), "rho_e": (["time"], rho_e),
                      "rho_p": (["time"], rho_p), "rho_s": (["time"], rho_s)})

    return out
