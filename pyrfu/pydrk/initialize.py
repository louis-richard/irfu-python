#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import pdb
import warnings

# 3rd partu imports
import yaml
import numpy as np

from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.4.8"
__status__ = "Prototype"


def _read_input(file_path):
    keys = ["q", "m", "n", "tpara", "tperp", "alpha", "delta", "vd"]

    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

        # Get particles inputs
        particles_ = config.pop("particles")

        inp = {}
        rsab = np.zeros((2, len(particles_.keys())))

        for i, k in enumerate(keys):
            inp[k] = np.array([particles_[specie][k] for specie in particles_])
            inp[k] = inp[k].astype(float)

        for i, (alpha, delta) in enumerate(zip(inp["alpha"], inp["delta"])):
            if alpha == 1.:
                rsab[0, i] = 1
                rsab[1, i] = 0
            else:
                rsab[0, i] = (1 - alpha * delta) / (1 - alpha)
                rsab[1, i] = alpha * (delta - 1) / (1 - alpha)

    return config, inp, rsab


def _poles(n_poles):
    bz_j, cz_j = [np.zeros(n_poles, dtype="complex128") for _ in range(2)]

    if n_poles == 8:
        # Ronnmark1982, 8-pole for Z function, and Z'
        bz_j[0] = -1.734012457471826e-2 - 4.630639291680322e-2 * 1j
        bz_j[1] = -7.399169923225014e-1 + 8.395179978099844e-1 * 1j
        bz_j[2] = +5.8406286421840730e0 + 9.536009057643667e-1 * 1j
        bz_j[3] = -5.5833715252868530e0 - 1.1208543191265990e1 * 1j
        bz_j[4:] = np.conj(bz_j[:4])

        cz_j[0] = 2.2376877892019000000 - 1.625940856173727000 * 1j
        cz_j[1] = 1.4652341261060040000 - 1.789620129162444000 * 1j
        cz_j[2] = .83925398172326380000 - 1.891995045765206000 * 1j
        cz_j[3] = .27393622262855640000 - 1.941786875844713000 * 1j
        cz_j[4:] = -np.conj(cz_j[:4])

    elif n_poles == 12:
        # from Cal_J_pole_bjcj.m, Xie2016
        bz_j[0] = -0.004547861216545870 - 0.000621096230229454 * 1j
        bz_j[1] = +0.215155729087593000 + 0.201505401672306000 * 1j
        bz_j[2] = +0.439545042119629000 + 4.161084683482920000 * 1j
        bz_j[3] = -20.21696733235520000 - 12.88550354824400000 * 1j
        bz_j[4] = +67.08148824503560000 + 20.84634584995040000 * 1j
        bz_j[5] = -48.01467382500760000 + 107.2756140925700000 * 1j
        bz_j[6:] = np.conj(bz_j[:6])

        cz_j[0] = -2.978429162451640000 - 2.049696666440500000 * 1j
        cz_j[1] = +2.256783783966820000 - 2.208618411895420000 * 1j
        cz_j[2] = -1.673799856171610000 - 2.324085194163360000 * 1j
        cz_j[3] = -1.159032033804220000 - 2.406739409547180000 * 1j
        cz_j[4] = +0.682287636603418000 - 2.460365014610040000 * 1j
        cz_j[5] = -0.225365375071350000 - 2.486779417047530000 * 1j
        cz_j[6:] = -np.conj(cz_j[:6])

    elif n_poles == 4:
        # Martin1980
        bz_j[0] = 0.5467968598340320000 + 0.037196505239277000 * 1j
        bz_j[1] = -1.046796859834027000 + 2.101852568038518000 * 1j
        bz_j[2:] = np.conj(bz_j[:2])

        cz_j[0] = 1.2358876534359200000 - 1.214982132557310000 * 1j
        cz_j[1] = -0.378611612386277000 - 1.350943585432730000 * 1j
        cz_j[2:] = -np.conj(cz_j[:2])
    else:
        raise ValueError("Invalid number of poles")

    return bz_j, cz_j


def _check_case(a_, b_):
    """
    Typical cases of (ip_a, ip_b):
        1. (1,1) scan k, fixed theta
        2. (2,2) scan theta, fixed k
        3. (1,2) scan 2D (k, theta)
        4. (3,3) scan kz, fixed kx
        5. (4,4) scan kx, fixed kz
        6. (3,4) scan 2D (kz,kx)
        7. (..,..) others, please modify 'pdrk_si_kernel.py'
    """

    cases = {1: [1, 1], 2: [2, 2], 3: [1, 2], 4: [3, 3],
             5: [4, 4], 6: [3, 4]}

    case_ = filter(lambda k: cases[k] == [a_["i_p"], b_["i_p"]], cases.keys())

    path_ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pdrky_cases.yml")

    with open(path_) as file:
        cases_cfg = yaml.load(file, Loader=yaml.FullLoader)
        pdb.set_trace()
        case = cases_cfg[list(case_)[0]]

    a_["strp"] = case.get("strp_a", None)
    b_["strp"] = case.get("strp_b", None)
    str_sc = case.get("str_sc", None)
    ip_b_tmp = case.get("ip_b_tmp", None)

    return a_, b_, str_sc, ip_b_tmp


def _disp(config):
    a_ = config["a"]
    b_ = config["b"]

    # print some basic info
    print("-" * 100)

    if config["i_em"]:
        print(f"iem = {config['i_em']}, this is an * electromagnetic * run.")
    else:
        print(f"iem = {config['i_em']}, this is an * electrostatic * run.")

    print(f"n = {config['n']}, [Number harmonics. ! Try larger N to make "
          f"sure the results are convergent")

    print(f"J = {config['j_poles']}, [J-pole, usually J=8 is sufficient; "
          f"other choice: 4 (faster), 12 (more accurate)]")

    print(f"sparse = {config['sparse']}, [sp=1, sparse eigs() to obtain nw0 "
          f"solutions; sp=0, eig() to obtain all solutions]")

    print(f"nw0 = {config['nw_0']}, [solve nw0 solutions one time]")

    print(f"i_out = {config['i_out']}, [=1, only (omega,gamma); =2, also "
          f"calculate (E,B)]")

    print(f"(npa, npb) = ({a_['n_p']}, {b_['n_p']} [number scan for the (1st, "
          f"2nd) variables]")

    print(f"(npa, npb) = ({a_['i_p']}, {b_['i_p']} , [see 'setup.m', 1: k; 2: "
          f"theta; 3: kz; 4: kx]")

    print(f"(iloga, iloga) = ({a_['i_log']}, {b_['i_log']}, [=0, linear scan; "
          f"=1, 10^(pa or pb) scan]")

    print(f"pa = {np.min(a_['p'])}:"
          f" {(a_['p'][-1] - a_['p'][0]) / (a_['n_p'] - 1 + 1e-10):4.3f}:"
          f"{np.max(a_['p'])}, [1st parameter range]")

    print(f"pa = {np.min(b_['p'])}:"
          f" {(b_['p'][-1] - b_['p'][0]) / (b_['n_p'] - 1 + 1e-10):4.3f}:"
          f"{np.max(b_['p'])}, [2nd parameter range]")

    print(f"This run: pa = {a_['strp']}, pb = {b_['strp']}, "
          f"{config['str_sc']}")

    print("-" * 100)

    print(f"S [number of species] = {config['n_species']:d}")
    print(f"B0 [backgroud magnetic field, Tesla] = {config['b0']:3.2e}")

    print("-" * 100)

    print(f"qs0 [charge, q/e] = {config['q_s0']}")
    print(f"ms0 [mass, m/mp] = {config['m_s0']}")
    print(f"ns0 [density, m^-3] = {config['n_s']}")
    print(f"Tzs0 [parallel temperature, eV] = {config['tz_s0']}")
    print(f"Tps0 [perp temperature, eV] = {config['tp_s0']}")

    print("-" * 100)

    print(f"lambdaDs [Debye length, m] = {config['lambdad_s']}")
    print(f"wps [plasma frequency, Hz] = {config['wp_s']}")
    print(f"wcs [cyclotron frequency, Hz] = {config['wc_s']}")
    print(f"rhocs [cyclotron radius, m] = {config['rhoc_s']}")
    print(f"lmdT [Tpara/Tperp] = {config['lmd_t']}")
    print(f"betasz [parallel beta] = {config['betaz_s']}")
    print(f"betapz [perp beta] = {config['betap_s']}")

    print("".join(["=" * 23,
                   " In PDRK plot/output, k -> k*cwp, omega -> omega/wcs1 ",
                   "=" * 23]))
    print("---- ! Set the 1st species to be ion in ''pdrk.in'', if")
    print("---- ! you hope wcs1=omega_ci and cwp=c/omega_pi.")

    print(f"wcs1 [1st species cyclotron frequency, Hz] = {config['wc_s0']}")
    print(f"wps1 [1st species plasma frequency, Hz] = {config['wp_s0']}")
    print(f"cwp [c/wps1, m] = {config['c_wp']}")

    return


def initialize(file_path: str) -> dict:
    r"""Initialize the root search based on the config (.yml) file.

    Parameters
    ----------
    file_path : str
        Path to the config (.yml) file.

    Returns
    -------
    config : dict
        Hash table with all properties.

    """

    eps_0 = constants.epsilon_0
    mu_0 = constants.mu_0
    k_b = constants.Boltzmann
    q_e = constants.elementary_charge
    m_p = constants.proton_mass
    ev_ = constants.electron_volt

    config, inp, config["rsab"] = _read_input(file_path)

    par_ = np.zeros(6)
    # k, = sqrt(kx ^ 2 + kz ^ 2), normalized by * c / omega_{p1}
    par_[0] = config["k"]
    # theta, the angle between k and B0, normalized by * pi / 180
    par_[1] = config["theta"]
    # kz, i.e., kpara * c / omega_{p1}
    par_[2] = np.cos(np.deg2rad(par_[1])) * par_[0]
    # kx, i.e., kperp * c / omega_{p1}
    par_[3] = np.sin(np.deg2rad(par_[1])) * par_[0]
    config["par"] = par_

    n_species = len(inp["q"])
    config["n_species"] = n_species

    q_total = np.sum(inp["q"] * inp["n"])
    j_total = np.sum(inp["q"] * inp["n"] * inp["vd"])

    if q_total != 0 or j_total != 0:
        warnings.warn("Total charge or current not zero !!!", UserWarning)

    # * electron charge, e -> C (coulomb)
    config["q_s0"] = inp["q"]
    config["q_s"] = inp["q"] * q_e

    # * proton mass, m_p -> kg, 18-10-13 09:57
    config["m_s0"] = inp["m"]
    config["m_s"] = inp["m"] * m_p

    config["n_s"] = inp["n"]

    # T//, eV -> K (eV -> J * J -> K)
    config["tz_s0"] = inp["tpara"]
    config["tp_s0"] = inp["tperp"]
    config["tz_s"] = inp["tpara"] * ev_ / k_b
    config["tp_s"] = inp["tperp"] * ev_ / k_b

    # vds, speed of light c -> m/s
    config["vd_s"] = inp["vd"] * constants.speed_of_light

    # para thermal velocity, note the sqrt(2)
    config["vtz_s"] = np.sqrt(2 * k_b * config["tz_s"] / config["m_s"])

    # Debye length, Tzs
    config["lambdad_s"] = np.sqrt(eps_0 * k_b * config["tz_s"]
                                  / (config["n_s"] * config["q_s"] ** 2))
    config["kd_s"] = 1. / config["lambdad_s"]

    # plasma and cyclotron frequency
    config["wp_s"] = np.sqrt(config["n_s"] * config["q_s"] ** 2
                             / (config["m_s"] * eps_0))
    config["wc_s"] = config["b0"] * config["q_s"] / config["m_s"]

    # cyclotron radius
    config["rhoc_s"] = np.sqrt(k_b * config["tp_s"]
                               / config["m_s"]) / config["wc_s"]

    config["wp_s2"] = config["wp_s"] ** 2
    config["lmd_t"] = config["tz_s"] / config["tp_s"]

    # for sigma=a,b due to the two perp temperatures for core and loss cone fv
    tp_s_ab = config["tp_s"] * np.vstack([np.ones(n_species), inp["alpha"]])
    config["lmd_t_ab"] = config["tz_s"] / tp_s_ab
    config["rhoc_s_ab"] = np.sqrt(k_b * tp_s_ab / config["m_s"]) \
                          / config["wc_s"]

    # beta_para,  beta_perp
    config["betaz_s"] = 2. * mu_0 * k_b * config["n_s"] * config["tz_s"]
    config["betaz_s"] /= config["b0"] ** 2
    config["betap_s"] = 2. * mu_0 * k_b * config["n_s"] * config["tp_s"]
    config["betap_s"] /= config["b0"] ** 2

    # Alfven speed
    config["v_a"] = config["b0"] \
                    / np.sqrt(mu_0 * np.sum(config["m_s"] * config["n_s"]))

    # normalized by omega_c and omega_p of the first species, 18-10-18 19:25
    # sound speed
    cs_s0 = np.sqrt(2 * k_b * config["tz_s"][0] / config["m_s"][0])
    # omega_{ci}
    config["wc_s0"] = np.absolute(config["wc_s"][0])
    config["wp_s0"] = np.sqrt(config["n_s"][0] * config["q_s"][0] ** 2
                              / (config["m_s"][0] * eps_0))

    # c/omega_{p1}
    config["c_wp"] = np.sqrt(constants.speed_of_light ** 2) / config["wp_s0"]
    config["v_a_wp"] = config["v_a"] / config["wc_s0"]  # v_A/omega_{c1}

    config["bz_j"], config["cz_j"] = _poles(config["j_poles"])

    """
    % J=length(bzj);
    % sum(bzj) % should be -1
    % sum(bzj.*czj) % should be 0
    % sum(bzj.*czj.^2) % should be -1/2
    """

    config["snj_0"] = n_species * (2 * config["n"] + 1) * config["j_poles"]
    config["snj_1"] = config["snj_0"] + 1

    if config["i_em"]:
        # electromagnetic case
        config["snj_3"] = 3 * config["snj_1"]
        config["nn_"] = config["snj_3"] + 6
    else:
        # electrostatic case
        config["nn_"] = config["snj_1"]

    if not config["sparse"]:
        config["nw_"] = config["nn_"]  # number of roots to obtain
    else:
        # using sparse matrix
        # !! only nw solutions around the initial guesses are given
        config["nw_"] = 1
        config["wg_"] = complex(config["wg_0"][0], config["wg_0"][1]) \
                        * config["wc_s0"]

    config["nw_0"] = config["nw_"]

    a_, b_ = [config[k] for k in ["a", "b"]]

    a_["n_p"] = int(np.round((a_["p2"] - a_["p1"]) / a_["dp"]) + 1)
    a_["p"] = a_["p1"] + np.arange(0, a_["n_p"]) * a_["dp"]

    if a_["i_p"] == b_["i_p"]:
        # if ipa==ipb, do only 1D scan of pa
        b_["n_p"] = 1
        b_["p"] = np.array([a_["p"][0]])  # 18-10-06 01:00
    else:
        # do 2D scan (pa,pb)
        b_["n_p"] = int(np.round((b_["p2"] - b_["p1"]) / b_["dp"]) + 1)
        b_["p"] = b_["p1"] + np.arange(0, b_["n_p"]) * b_["dp"]

    a_, b_, config["str_sc"], config["ip_b_tmp"] = _check_case(a_, b_)

    config["a"] = a_
    config["b"] = b_

    _disp(config)

    return config
