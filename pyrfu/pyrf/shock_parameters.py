#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# Third party imports
import numpy as np
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["shock_parameters"]

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def shock_parameters(spec):
    r"""Calculate shock related plasma parameters.

    Parameters
    ----------
    spec : dict
        Hash table with parameters input with fixed names. After the parameter name,
        a name of the region can be given, e.g. "u" and "d". All parameters except b
        are optional.

    Returns
    -------
    out : dict
        Hash table with derived plasma parameters from hash table spec with measured
        plasma parameters.

    """

    id_b = list(filter(lambda x: x[0].lower() == "b", spec))
    regions = [id_[1:] for id_ in id_b if len(id_) > 1]
    regions = regions if len(regions) > 1 else [""]

    spec["ref_sys"] = spec.get("ref_sys", "sc")
    assert spec["ref_sys"].lower() in ["sc", "nif"], "Invalid reference frame"

    if spec["ref_sys"].lower() == "nif":
        if "nvec" not in spec:
            logging.warning("Setting shock speed, nvec to [1, 0, 0]")
            spec["nvec"] = np.array([1, 0, 0])

        if "v_sh" not in spec:
            logging.warning("Setting shock speed, Vsh, to 0.")
            spec["v_sh"] = 0.0

    dspec = {}

    # Alfven speed
    if f"n{regions[0]}" in spec:
        for region in regions:
            dspec[f"v_a{region}"] = _v_alfv(spec[f"b{region}"], spec[f"n{region}"])

    # Sound speed
    if f"t_i{regions[0]}" in spec and f"t_e{regions[0]}" in spec:
        for region in regions:
            dspec[f"v_ts{region}"] = _v_sound(
                spec[f"t_i{region}"], spec[f"t_e{region}"]
            )

    # Fast speed
    if (
        f"n{regions[0]}" in spec
        and f"v{regions[0]}" in spec
        and f"t_i{regions[0]}" in spec
        and f"t_e{regions[0]}" in spec
    ):
        for region in regions:
            dspec[f"v_f{region}"] = _v_fast(
                spec[f"b{region}"],
                spec[f"n{region}"],
                spec[f"v{region}"],
                spec[f"t_i{region}"],
                spec[f"t_e{region}"],
            )

    # Normal incidence frame velocity
    if f"v{regions[0]}" in spec and "nvec" in spec and "v_sh" in spec:
        for region in regions:
            dspec[f"v_nif{region}"] = _nif_speed(
                spec[f"v{region}"], spec["v_sh"], spec["nvec"]
            )

    # de Hoffman-Teller frame velocity
    if f"v{regions[0]}" in spec and "nvec" in spec and "v_sh" in spec:
        for region in regions:
            dspec[f"v_htf{region}"] = _htf_speed(
                spec[f"b{region}"], spec[f"v{region}"], spec["v_sh"], spec["nvec"]
            )

    # Ion gyrofrequency
    for region in regions:
        dspec[f"f_cp{region}"] = _ion_gyro_freq(spec[f"b{region}"])

    # Ion inertial length
    if f"n{regions[0]}" in spec:
        for region in regions:
            dspec[f"l_i{region}"] = _ion_in_len(spec[f"n{region}"])

    if f"n{regions[0]}" in spec:
        for region in regions:
            dspec[f"r_cp{region}"] = _ion_gyro_rad(
                spec[f"b{region}"], spec[f"v{region}"]
            )

    # Alfven Mach number
    if f"n{regions[0]}" in spec and f"v{regions[0]}" in spec:
        for region in regions:
            dspec[f"m_a{region}"] = _alfv_mach(
                spec[f"b{region}"],
                spec[f"n{region}"],
                spec[f"v{region}"],
                spec["ref_sys"],
                spec["v_sh"],
                spec["nvec"],
            )

    # Sonic Mach number
    if (
        f"v{regions[0]}" in spec
        and f"t_i{regions[0]}" in spec
        and f"t_e{regions[0]}" in spec
    ):
        for region in regions:
            dspec[f"m_s{region}"] = _sonic_mach(
                spec[f"v{region}"],
                spec[f"t_i{region}"],
                spec[f"t_e{region}"],
                spec["ref_sys"],
                spec["v_sh"],
                spec["nvec"],
            )

    if (
        f"n{regions[0]}" in spec
        and f"v{regions[0]}" in spec
        and f"t_i{regions[0]}" in spec
        and f"t_e{regions[0]}" in spec
    ):
        for region in regions:
            dspec[f"m_f{region}"] = _fast_mach(
                spec[f"b{region}"],
                spec[f"n{region}"],
                spec[f"v{region}"],
                spec[f"t_i{region}"],
                spec[f"t_e{region}"],
                spec["ref_sys"],
                spec["v_sh"],
                spec["nvec"],
            )

    if f"n{regions[0]}" in spec and f"t_i{regions[0]}" in spec:
        for region in regions:
            dspec[f"beta_i{region}"] = _beta(
                spec[f"b{region}"], spec[f"n{region}"], spec[f"t_i{region}"]
            )

    if f"n{regions[0]}" in spec and f"t_e{regions[0]}" in spec:
        for region in regions:
            dspec[f"beta_e{region}"] = _beta(
                spec[f"b{region}"], spec[f"n{region}"], spec[f"t_e{region}"]
            )

    return dspec


def _ion_gyro_freq(b):
    b_si = 1e-9 * np.linalg.norm(b)
    w_cp = constants.elementary_charge * b_si / constants.proton_mass
    return w_cp / (2 * np.pi)


def _ion_in_len(n):
    n_si = 1e6 * n
    w_pp = np.sqrt(
        n_si
        * constants.elementary_charge**2
        / (constants.proton_mass * constants.epsilon_0)
    )
    l_i = constants.speed_of_light / w_pp
    return l_i


def _ion_gyro_rad(b, v):
    b_si = 1e-9 * np.linalg.norm(b)
    e_i = 0.5 * constants.proton_mass * np.linalg.norm(v) ** 2 / constants.electron_volt
    v_tp = constants.speed_of_light * np.sqrt(
        1
        - 1
        / (
            e_i
            * constants.elementary_charge
            / (constants.proton_mass * constants.speed_of_light**2)
            + 1
        )
        ** 2
    )
    gamma_p = 1 / np.sqrt(1 - (v_tp / constants.speed_of_light) ** 2)
    rho_p = (
        constants.proton_mass
        * constants.speed_of_light
        / (constants.elementary_charge * b_si)
        * np.sqrt(gamma_p**2 - 1)
    )
    return rho_p


def _v_alfv(b, n):
    b_si = 1e-9 * np.linalg.norm(b)
    n_si = 1e6 * n
    return b_si / np.sqrt(constants.mu_0 * n_si * constants.proton_mass)


def _v_sound(t_i, t_e):
    t_i_si = t_i * constants.electron_volt
    t_e_si = t_e * constants.electron_volt
    return np.sqrt((t_e_si + 3 * t_i_si) / constants.proton_mass)


def _v_fast(b, n, v, t_i, t_e, theta=None):
    if theta is None:
        theta = np.arccos(np.sum(b * v) / (np.linalg.norm(b) * np.linalg.norm(v)))
    else:
        theta = np.deg2rad(theta)

    v_a = _v_alfv(b, n)
    c_s = _v_sound(t_i, t_e)
    c_ms0 = np.sqrt(v_a**2 + c_s**2)
    v_f = np.sqrt(
        c_ms0**2 / 2 + np.sqrt(c_ms0**4 / 4 - v_a**2.0 * c_s**2 * np.cos(theta) ** 2)
    )
    return v_f


def _nif_speed(v, v_sh, nvec):
    v_si = 1e3 * v
    v_sh_si = 1e3 * v_sh
    v_nif = v_si - (np.sum(v_si * nvec) - v_sh_si) * nvec
    return v_nif


def _htf_speed(b, v, v_sh, nvec):
    v_si = 1e3 * v
    v_sh_si = 1e3 * v_sh

    # first get the velocity in a shock rest frame
    v_in_shock_rest_frame = v_si - v_sh_si * nvec

    # then get the dHT frame speed in the shock rest frame
    v_htf_srf = np.cross(nvec, np.cross(v_in_shock_rest_frame, b)) / np.sum(b * nvec)

    # then the dHT frame speed in the sc frame is the shock speed plus  the dHT frame
    # speed in the shock rest frame (I think)
    v_htf = v_sh_si * nvec + v_htf_srf
    return v_htf


def _alfv_mach(b, n, v, ref_sys, v_sh, nvec):
    v_si = 1e3 * v
    v_sh_si = 1e3 * v_sh

    if ref_sys.lower() == "nif" and nvec is not None:
        m_a = np.abs(np.sum(v_si * nvec) - v_sh_si) / _v_alfv(b, n)
    else:
        m_a = np.linalg.norm(v_si) / _v_alfv(b, n)

    return m_a


def _sonic_mach(v, t_i, t_e, ref_sys, v_sh, nvec):
    v_si = 1e3 * v
    v_sh_si = 1e3 * v_sh

    if ref_sys.lower() == "nif" and nvec is not None:
        m_s = np.abs(np.sum(v_si * nvec) - v_sh_si) / _v_sound(t_i, t_e)
    else:
        m_s = np.linalg.norm(v_si) / _v_sound(t_i, t_e)

    return m_s


def _fast_mach(b, n, v, t_i, t_e, ref_sys, v_sh, nvec):
    v_si = 1e3 * v
    v_sh_si = 1e3 * v_sh

    if ref_sys.lower() == "nif" and nvec is not None:
        theta_bn = np.rad2deg(np.arccos(np.sum(b * nvec) / np.linalg.norm(b)))
        m_f = np.abs(np.sum(v_si * nvec) - v_sh_si) / _v_fast(
            b, n, v, t_i, t_e, theta_bn
        )
    else:
        m_f = np.linalg.norm(v_si) / _v_fast(b, n, v, t_i, t_e)

    return m_f


def _beta(b, n, t_s):
    b_si = 1e-9 * np.linalg.norm(b)
    n_si = 1e6 * n
    t_s_si = t_s * constants.electron_volt
    beta_s = (n_si * t_s_si) / (b_si**2 / (2 * constants.mu_0))
    return beta_s
