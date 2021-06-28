#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _print_header():
    print("=" * 70)
    print("IRFU plasma calculator, relativistic effects not fully included")
    print("velocities, gyroradia are relativistically correct")
    print("can somebody fix relativstically correct frequencies Fpe, Fce,.. ?")
    print("=" * 70)


def _print_frequencies(f_pe, f_ce, f_uh, f_lh, f_pp, f_cp, f_col):
    print("\nFrequencies: ")
    print("*" * 12)
    print(f"{'F_pe':>5} = {f_pe:>6.2E} Hz")
    print(f"{'F_ce':>5} = {f_ce:>6.2E} Hz")
    print(f"{'F_uh':>5} = {f_uh:>6.2E} Hz")
    print(f"{'F_lh':>5} = {f_lh:>6.2E} Hz")
    print(f"{'F_pp':>5} = {f_pp:>6.2E} Hz")
    print(f"{'F_cp':>5} = {f_cp:>6.2E} Hz")
    print(f"{'F_col':>5} = {f_col:>6.2E} Hz")


def _print_lengths(l_d, l_e, l_i, rho_e, rho_p, r_col):
    print("\nLengths: ")
    print("*" * 11)
    print(f"{'l_d':>5} = {l_d:>6.2E} m")
    print(f"{'d_e':>5} = {l_e:>6.2E} m")
    print(f"{'d_i':>5} = {l_i:>6.2E} m")
    print(f"{'r_e':>5} = {rho_e:>6.2E} m")
    print(f"{'r_p':>5} = {rho_p:>6.2E} m")
    print(f"{'r_col':>5} = {r_col:6.2E} m")


def _print_velocities(v_a, v_ae, v_te, v_tp, v_ts):
    print("\nVelocities: ")
    print("*" * 11)
    print(f"{'V_a':>5} = {v_a:>6.2E} m/s")
    print(f"{'V_ae':>5} = {v_ae:>6.2E} m/s")
    print(f"{'V_te':>5} = {v_te:>6.2E} m/s")
    print(f"{'V_tp':>5} = {v_tp:>6.2E} m/s")
    print(f"{'C_s':>5} = {v_ts:>6.2E} m/s")


def _print_other(n_d, eta, p_mag):
    print("\nOther parameters: ")
    print("*" * 17)
    print(f"{'N_deb':>5} = {n_d:>6.2E} {'':<6} "
          f"# number of particle in Debye sphere")
    print(f"{'eta':>5} = {eta:>6.2E} {'Ohm m':<6} # Spitzer resistivity")
    print(f"{'P_B':>5} = {p_mag:>6.2E} {'Pa':<6} # Magnetic pressure")


def _print_dimensionless(beta, gamma_e):
    m_p = constants.proton_mass
    m_e = constants.electron_mass
    mp_me = m_p / m_e

    print("\nDimensionless parameters: ")
    print("*" * 25)
    print(f"{'beta':>20} = {beta:>6.2E} #  H+ beta")
    print(f"{'beta*sqrt(m_p/m_e)':>20} = {beta * np.sqrt(mp_me):>6.2E}")
    print(f"{'beta*(m_p/m_e)':>20} = {beta * mp_me:>6.2E}")
    print(f"{'gamma_e':>20} = {gamma_e:>6.2E} # 1/sqrt(1-(V_te/c)^2)")


def iplasma_calc(output: bool = False, verbose: bool = True):
    r"""Interactive function to calcute plasma paramters.

    Parameters
    ----------
    output : bool, Optional
        Flag to return dict with output. Default is False.
    verbose : bool, Optional
        Flag to print the results function. Default is True.

    Returns
    -------
    out : dict
        Hashtable with everything.

    """

    b_0 = float(input(f"{'Magnetic field in nT [10] ':<34}: ") or "10") * 1e-9
    n_hplus = float(input(f"{'H+ desity in cc [1] ':<34}: ") or "1") * 1e6
    t_e = float(input(f"{'Electron  temperature in eV [100] ':<34}: ") or "10")
    t_i = float(input(f"{'Ion  temperature in eV [1000] ':<34}: ") or "1000")

    n_i, n_e = [n_hplus] * 2

    # Get constants
    q_e = constants.elementary_charge
    cel = constants.speed_of_light
    mu0 = constants.mu_0
    ep0 = constants.epsilon_0
    m_p = constants.proton_mass
    m_e = constants.electron_mass
    mp_me = m_p / m_e

    w_pe = np.sqrt(n_e * q_e ** 2 / (m_e * ep0)) 	# rad/s
    w_ce = q_e * b_0 / m_e   						# rad/s
    w_pp = np.sqrt(n_i * q_e ** 2 / (m_p * ep0))

    p_mag = b_0 ** 2 / (2 * mu0)

    v_a = b_0 / np.sqrt(mu0 * n_i * m_p)

    v_ae = b_0 / np.sqrt(mu0 * n_e * m_e)
    v_te = cel * np.sqrt(1 - 1 / (t_e * q_e / (m_e * cel ** 2) + 1) ** 2)
    v_tp = cel * np.sqrt(1 - 1 / (t_i * q_e / (m_p * cel ** 2) + 1) ** 2)
    # Sound speed formula (F. Chen, Springer 1984).
    v_ts = np.sqrt((t_e * q_e + 3 * t_i * q_e) / m_p)

    gamma_e = 1 / np.sqrt(1 - (v_te / cel) ** 2)
    gamma_p = 1 / np.sqrt(1 - (v_tp / cel) ** 2)

    l_e = cel / w_pe
    l_i = cel / w_pp
    # Debye length scale, sqrt(2) needed because of Vte definition
    l_d = v_te / (w_pe * np.sqrt(2))
    # number of e- in Debye sphere
    n_d = l_d * ep0 * m_e * v_te ** 2 / q_e ** 2

    f_pe = w_pe / (2 * np.pi) 				# Hz
    f_ce = w_ce / (2 * np.pi)
    f_uh = np.sqrt(f_ce ** 2 + f_pe ** 2)
    f_pp = w_pp / (2 * np.pi)
    f_cp = f_ce / mp_me
    f_lh = np.sqrt(f_cp * f_ce / (1 + f_ce ** 2 / f_pe ** 2) + f_cp ** 2)

    rho_e = m_e * cel / (q_e * b_0) * np.sqrt(gamma_e ** 2 - 1)
    rho_p = m_p * cel / (q_e * b_0) * np.sqrt(gamma_p ** 2 - 1)
    rho_s = v_ts / (f_cp * 2 * np.pi)

    # Collision stuff
    # collision frequency e-/ions
    f_col = (n_e * q_e ** 4) / 16 * np.pi * ep0 ** 2 * m_e ** 2 * v_te ** 3
    # Spitzer resistivity
    eta = (np.pi * q_e ** 2 * np.sqrt(m_e))
    eta /= ((4 * np.pi * ep0) ** 2 * (q_e * t_e) ** (3 / 2))
    eta *= np.log(4 * np.pi * n_d)
    # resistive scale
    r_col = eta / (mu0 * v_a)

    beta = v_tp ** 2 / v_a ** 2

    if verbose:
        _print_header()
        _print_frequencies(f_pe, f_ce, f_uh, f_lh, f_pp, f_cp, f_col)
        _print_lengths(l_d, l_e, l_i, rho_e, rho_p, r_col)
        _print_velocities(v_a, v_ae, v_te, v_tp, v_ts)
        _print_other(n_d, eta, p_mag)
        _print_dimensionless(beta, gamma_e)

    if output:
        out = {"w_pe": w_pe, "w_ce": w_ce, "w_pp": w_pp, "v_a": v_a,
               "v_ae": v_ae, "v_te": v_te, "v_tp": v_tp, "v_ts": v_ts,
               "gamma_e": gamma_e, "gamma_p": gamma_p, "l_e": l_e, "l_i": l_i,
               "l_d": l_d, "n_d": n_d, "f_pe": f_pe, "f_ce": f_ce,
               "f_uh": f_uh, "f_pp": f_pp, "f_cp": f_cp, "f_lh": f_lh,
               "rho_e": rho_e, "rho_p": rho_p, "rho_s": rho_s}
    else:
        out = None

    return out
