#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# Third party imports
import numpy as np
from scipy import constants

__all__ = ["shock_parameters"]

logger = logging.getLogger(__name__)


def shock_parameters(spec):
    r"""Calculate shock related plasma parameters.

    Parameters
    ----------
    spec : dict
        Hash table with parameters input with fixed names. After the parameter name,
        a name of the region can be given, e.g. "u" and "d". All parameters except b
        are optional. The keys (vectors as arrays or lists of 3 components):

        * b : magnetic field [nT].
        * n : density [cm^-3].
        * v : bulk velocity [km/s].
        * t_i, t_e : ion and electron temperatures [eV].
        * ref_sys : frame of the Mach numbers, "sc" (spacecraft, default) or
          "nif" (normal incidence frame, along nvec).
        * nvec : shock normal (unit vector), default [1, 0, 0] with "nif".
        * v_sh : shock speed along nvec [km/s], default 0 with "nif".

    Returns
    -------
    out : dict
        Hash table with derived plasma parameters from hash table spec with measured
        plasma parameters, for each region (when their inputs are given):

        * v_a, v_ts, v_f : Alfven, sound and fast speeds [m/s].
        * v_nif, v_htf : normal incidence and de Hoffmann-Teller frame
          velocities [m/s] (with nvec and v_sh).
        * f_cp : proton gyrofrequency [Hz].
        * l_i : ion inertial length [m].
        * r_cp : proton gyroradius [m].
        * m_a, m_s, m_f : Alfven, sonic and fast Mach numbers.
        * beta_i, beta_e : ion and electron plasma betas.

    Raises
    ------
    ValueError
        If ref_sys is not "sc" or "nif".

    """

    # Copy so that the defaults set below don't change the caller's dict;
    # lists as arrays (b_u or v_u given as lists raised TypeError at 1e3 * v)
    spec = {
        k: np.asarray(v) if isinstance(v, (list, tuple)) else v for k, v in spec.items()
    }

    id_b = list(filter(lambda x: x[0].lower() == "b", spec))
    regions = [id_[1:] for id_ in id_b if len(id_) > 1]
    regions = regions or [""]

    spec["ref_sys"] = spec.get("ref_sys", "sc")
    if spec["ref_sys"].lower() not in ["sc", "nif"]:
        raise ValueError(f'ref_sys must be "sc" or "nif", not {spec["ref_sys"]!r}')

    if spec["ref_sys"].lower() == "nif":
        if "nvec" not in spec:
            logger.warning("Setting shock speed, nvec to [1, 0, 0]")
            spec["nvec"] = np.array([1, 0, 0])

        if "v_sh" not in spec:
            logger.warning("Setting shock speed, Vsh, to 0.")
            spec["v_sh"] = 0.0

    # The parameters of each region whose inputs are given for the first
    # region, in the order of _PARAMETERS
    dspec = {}

    for name, inputs, frame, func in _PARAMETERS:
        given = all(f"{q}{regions[0]}" in spec for q in inputs)

        # The frame velocities need the shock normal and speed
        if given and (not frame or ("nvec" in spec and "v_sh" in spec)):
            for region in regions:
                dspec[f"{name}{region}"] = func(spec, region)

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
    # v in km/s
    e_i = (
        0.5
        * constants.proton_mass
        * (1e3 * np.linalg.norm(v)) ** 2
        / constants.electron_volt
    )
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


def _frame(spec):
    # Frame of the Mach numbers: the reference system, the shock speed (0 if
    # not given) and normal (None if not given)
    return spec["ref_sys"], spec.get("v_sh", 0.0), spec.get("nvec")


# Derived parameters: name, inputs of each region (all given for the first
# region), whether nvec and v_sh are needed, and the function of spec and the
# region (one guarded loop each before)
_PARAMETERS = [
    ("v_a", ["n"], False, lambda s, r: _v_alfv(s[f"b{r}"], s[f"n{r}"])),
    ("v_ts", ["t_i", "t_e"], False, lambda s, r: _v_sound(s[f"t_i{r}"], s[f"t_e{r}"])),
    (
        "v_f",
        ["n", "v", "t_i", "t_e"],
        False,
        lambda s, r: _v_fast(
            s[f"b{r}"], s[f"n{r}"], s[f"v{r}"], s[f"t_i{r}"], s[f"t_e{r}"]
        ),
    ),
    ("v_nif", ["v"], True, lambda s, r: _nif_speed(s[f"v{r}"], s["v_sh"], s["nvec"])),
    (
        "v_htf",
        ["v"],
        True,
        lambda s, r: _htf_speed(s[f"b{r}"], s[f"v{r}"], s["v_sh"], s["nvec"]),
    ),
    ("f_cp", [], False, lambda s, r: _ion_gyro_freq(s[f"b{r}"])),
    ("l_i", ["n"], False, lambda s, r: _ion_in_len(s[f"n{r}"])),
    ("r_cp", ["v"], False, lambda s, r: _ion_gyro_rad(s[f"b{r}"], s[f"v{r}"])),
    (
        "m_a",
        ["n", "v"],
        False,
        lambda s, r: _alfv_mach(s[f"b{r}"], s[f"n{r}"], s[f"v{r}"], *_frame(s)),
    ),
    (
        "m_s",
        ["v", "t_i", "t_e"],
        False,
        lambda s, r: _sonic_mach(s[f"v{r}"], s[f"t_i{r}"], s[f"t_e{r}"], *_frame(s)),
    ),
    (
        "m_f",
        ["n", "v", "t_i", "t_e"],
        False,
        lambda s, r: _fast_mach(
            s[f"b{r}"], s[f"n{r}"], s[f"v{r}"], s[f"t_i{r}"], s[f"t_e{r}"], *_frame(s)
        ),
    ),
    (
        "beta_i",
        ["n", "t_i"],
        False,
        lambda s, r: _beta(s[f"b{r}"], s[f"n{r}"], s[f"t_i{r}"]),
    ),
    (
        "beta_e",
        ["n", "t_e"],
        False,
        lambda s, r: _beta(s[f"b{r}"], s[f"n{r}"], s[f"t_e{r}"]),
    ),
]
