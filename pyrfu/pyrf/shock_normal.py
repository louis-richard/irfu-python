#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# Built-in imports
import os

# Thrid party imports
import numpy as np
import xarray as xr
from scipy import constants, interpolate, optimize
from scipy.spatial.transform import Rotation as R

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["shock_normal"]


def shock_normal(spec, leq90: bool = True):
    r"""Calculates shock normals with different methods. Normal vectors are calculated
    by methods described in [1]_ and references therein.

    The data can be averaged values or values from the time series in matrix format.
    If series is from time series all parameters are calculated from a random
    upstream and a random downstream point. This can help set errorbars on shock
    angle etc. The time series input must have the same size (up- and downstream can
    be different), so generally the user needs to resample the data first.

    Parameters
    ----------
    spec : dict
        Hash table with:
            * b_u : Upstream magnetic field (nT).
            * b_d : Downstream magnetic field.
            * v_u : Upstream plasma bulk velocity (km/s).
            * v_d : Downstream plasma bulk velocity.
            * n_u : Upstream number density (cm^-3).
            * n_d : Downstream number density.
            * r_xyz : Spacecraft position in time series format of 1x3 vector. Optional.
            * d2u : Down-to-up, is 1 or -1. Optional.
            * dt_f : Time duration of shock foot (s). Optional.
            * f_cp : Reflected ion gyrofrequency (Hz). Optional.
            * n : Number of Monte Carlo particles. Optional, default is 100.

    leq90 : bool, Optional
        Force angles to be less than 90 (default). For leq90 = 0, angles can be between
        0 and 180 deg. For time series input and quasi-perp shocks,leq90 = 0 is
        recommended.

    Returns
    -------
    out : dict
        Hash table with:
            * n : Hash table containing normal vectors (n always points toward the
            upstream region).
            From data:
                * mc : Magnetic coplanarity (10.14)
                * vc : Velocity coplanarity (10.18)
                * mx_1 : Mixed method 1 (10.15), [2]_
                * mx_2 : Mixed method 2 (10.16), [2]_
                * mx_3 : Mixed method 3 (10.17), [2]_
            From models (only if r_xyz is included in spec):
                * farris : [3]_
                * slho : [4]_
                * per : [5]_, (z = 0)
                * fa4o : [6]_
                * fan4o : [6]_
                * foun : [7]_

            * theta_bn : Angle between normal vector and b_u, same fields as n.
            * theta_vn : Angle between normal vector and v_u, same fields as n.
            * v_sh : Hash table containing shock velocities:
                * gt : Using shock foot thickness (10.32). [8]_
                * mf : Mass flux conservation (10.29).
                * sb : Using jump conditions (10.33). [9]_
                * mo : Using shock foot thickness
            * info : Hash table containing some more info:
                * msh : Magnetic shear angle.
                * vsh : Velocity shear angle.
                * cmat : Constraints matrix with normalized errors.
                * sig : Scaling factor to fit shock models to sc position. Calculated
                from (10.9-10.13) in [1]_


    References
    ----------
    .. [1]  Schwartz, S. J. (1998), Shock and Discontinuity Normals, Mach Numbers, and
            Related Parameters, ISSI Scientific Reports Series, vol. 1, pp. 249–270.
    .. [2]  Abraham-Shrauner, B. (1972), “Determination of magnetohydrodynamic shock
            normals”, Journal of Geophysical Research, vol. 77, no. 4, p. 736.
            doi:10.1029/JA077i004p00736.
    .. [3]  Farris, M. H., Petrinec, S. M., and Russell, C. T. (1991), The thickness of
            the magnetosheath: Constraints on the polytropic index”, Geophysical
            Research Letters, vol. 18, no. 10, pp. 1821–1824. doi:10.1029/91GL02090.
    .. [4]  Slavin, J. A. and Holzer, R. E. (1981), Solar wind flow about the
            terrestrial planets, 1. Modeling bow shock position and shape, Journal of
            Geophysical Research, vol. 86, no. A13, pp. 11401–11418.
            doi:10.1029/JA086iA13p11401.
    .. [5]  Peredo, M., Slavin, J. A., Mazur, E., and Curtis, S. A. (1995),
            Three-dimensional position and shape of the bow shock and their variation
            with Alfvénic, sonic, and magnetosonic Mach numbers and interplanetary
            magnetic field orientation, Journal of Geophysical Research, vol. 100,
            no. A5, pp. 7907–7916. doi:10.1029/94JA02545.
    .. [6]  Fairfield, D. H. (1971), Average and unusual locations of the Earth's
            magnetopause and bow shock, Journal of Geophysical Research, vol. 76,
            no. 28, p. 6700, 1971. doi:10.1029/JA076i028p06700.
    .. [7]  Formisano, V. (1979), Orientation and Shape of the Earth's Bow Shock in
            Three Dimensions, Planetary and Space Science, vol. 27, no. 9,
            pp. 1151–1161. doi:10.1016/0032-0633(79)90135-1.
    .. [8]  Gosling, J. T. and Thomsen, M. F. (1985), Specularly reflected ions, shock
            foot thicknesses, and shock velocity determination in space, Journal of
            Geophysical Research, vol. 90, no. A10, pp. 9893–9896.
            doi:10.1029/JA090iA10p09893.
    .. [9]  Smith, E. J. and Burton, M. E. (1988), Shock analysis: Three useful new
            relations, Journal of Geophysical Research, vol. 93, no. A4, pp. 2730–2734.
            doi:10.1029/JA093iA04p02730.



    """

    # Check input
    assert isinstance(spec, dict), "spec must be a dictionary"

    if spec["b_u"].ndim > 1 or spec["b_d"].ndim > 1:
        n_bu = len(spec["b_u"])
        n_bd = len(spec["b_d"])

        # randomize points upstream and downstream
        n = int(np.floor(spec.get("n", 10.0)))
        idt_u, idt_d = [np.random.rand(n) for _ in range(2)]

        tmp_spec = {}
        for i in range(n):
            f_bu = interpolate.interp1d(np.linspace(0, 1, n_bu), spec["b_u"], axis=0)
            tmp_spec["b_u"] = f_bu(idt_u[i])
            f_vu = interpolate.interp1d(np.linspace(0, 1, n_bu), spec["v_u"], axis=0)
            tmp_spec["v_u"] = f_vu(idt_u[i])
            f_nu = interpolate.interp1d(np.linspace(0, 1, n_bu), spec["n_u"], axis=0)
            tmp_spec["n_u"] = f_nu(idt_u[i])

            f_bd = interpolate.interp1d(np.linspace(0, 1, n_bd), spec["b_d"], axis=0)
            tmp_spec["b_d"] = f_bd(idt_d[i])
            f_vd = interpolate.interp1d(np.linspace(0, 1, n_bd), spec["v_d"], axis=0)
            tmp_spec["v_d"] = f_vd(idt_d[i])
            f_nd = interpolate.interp1d(np.linspace(0, 1, n_bd), spec["n_d"], axis=0)
            tmp_spec["n_d"] = f_nd(idt_d[i])

    # normal vector, according to different models
    normal = {}

    b_u, b_d = np.array(spec["b_u"]), np.array(spec["b_d"])
    v_u, v_d = np.array(spec["v_u"]), np.array(spec["v_d"])

    delta_b = b_d - b_u
    delta_v = v_d - v_u
    spec["delta_b"] = delta_b
    spec["delta_v"] = delta_v

    # magenetic coplanarity
    mc = np.cross(np.cross(b_d, b_u), delta_b)
    mc /= np.linalg.norm(mc, keepdims=True)
    normal["mc"] = mc

    # velocity coplanarity
    vc = delta_v / np.linalg.norm(delta_v, keepdims=True)
    normal["vc"] = vc

    # mixed methods
    mx_1 = np.cross(np.cross(b_u, delta_v), delta_b)
    mx_1 /= np.linalg.norm(mx_1)
    normal["mx_1"] = mx_1
    mx_2 = np.cross(np.cross(b_d, delta_v), delta_b)
    mx_2 /= np.linalg.norm(mx_2)
    normal["mx_2"] = mx_2
    mx_3 = np.cross(np.cross(delta_b, delta_v), delta_b)
    mx_3 /= np.linalg.norm(mx_3)
    normal["mx_3"] = mx_3

    # Load shock
    # pkg_path = os.path.join(os.getcwd(), "sandbox", "sbox")
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(pkg_path, "shock_models_parameters.json"), "r", encoding="utf-8"
    ) as fs:
        shock_models_params = json.load(fs)

    if "r_xyz" in spec:
        # info
        info = {k: {} for k in shock_models_params["farris"]}
        sig = {}

        # overwrite alpha to azimuthal angle of the bulk velocity for
        # Slavin-Holzer model
        alpha_sh = -np.rad2deg(np.arctan(spec["v_u"][1] / spec["v_u"][0]))
        shock_models_params["slavin_holzer"]["alpha"] = alpha_sh

        for m in shock_models_params:
            for k in info:
                info[k][m] = shock_models_params[m][k]

            normal[m], sig[m] = _shock_model(spec, *shock_models_params[m].values())

        info["sig"] = sig
    else:
        info = {}

    # make sure all normal vectors are pointing upstream based on delta_sv,
    # should work for IP shocks also
    for n_ in normal.values():
        if np.sum(delta_v * n_) < 0:
            n_ *= -1

    # Shock normal to magnetic field and velocity angles
    theta_bn = _shock_angle(spec, normal, "b", leq90)
    theta_vn = _shock_angle(spec, normal, "v", leq90)

    # Magnetic and velocity shear angles
    info["msh"] = _shear_angle(b_u, b_d)
    info["vsh"] = _shear_angle(v_u, v_d)

    # Constraint matrix
    info["cmat"] = _constraint_matrix(spec, normal)

    # Shock speed
    v_sh = {}
    for m in ["gt", "mf", "sb", "mo"]:
        v_sh[m] = _shock_speed(spec, normal, theta_bn, m)

    out = {
        "info": info,
        "n": normal,
        "theta_bn": theta_bn,
        "theta_vn": theta_vn,
        "v_sh": v_sh,
    }

    return out


def _shear_angle(au, ad):
    theta = np.arccos(np.sum(au * ad) / (np.linalg.norm(au) * np.linalg.norm(ad)))
    return np.rad2deg(theta)


def _shock_angle(spec, n, field, leq90):
    if field.lower() == "b":
        a = spec["b_u"]
    else:
        a = spec["v_u"]

    theta = {}

    for fname in n:
        tmp = np.rad2deg(np.arccos(np.sum(a * n[fname]) / np.linalg.norm(a)))

        if tmp > 90.0 and leq90:
            theta[fname] = 90.0 - tmp
        else:
            theta[fname] = tmp

    return theta


def _shock_model(spec, *args):
    eps, l_, x_0, y_0, alpha = args

    # Rotation matrix
    rot_mat = R.from_euler("z", alpha, degrees=True).as_matrix()

    # offset from GSE
    r_0 = np.array([x_0, y_0, 0])

    # sc position in GSE (or GSM or whatever) in Earth radii
    if isinstance(spec["r_xyz"], xr.DataArray):
        # Time series
        r_sc = np.mean(spec["r_xyz"].data, axis=0) / 6371.0
    elif isinstance(spec["r_xyz"], (np.ndarray, list)) and len(spec["r_xyz"]) == 3:
        # Array like
        r_sc = spec["r_xyz"] / 6371.0
    else:
        raise TypeError("r_xyz must be a time series or a vector!!")

    def fval(sig, *args):
        rot_mat, r_sc, r_0, l_ = args

        # sc position in the natural system (cartesian)
        r_p = np.matmul(rot_mat, r_sc) - sig * r_0

        # sc polar angle in the natural system
        theta_p = np.arctan2(np.sqrt(r_p[1] ** 2 + r_p[2] ** 2), r_p[0])

        # minimize |LH-RH| in eq 10.22
        res = np.abs(sig * l_ / np.linalg.norm(r_p) - 1 - eps * np.cos(theta_p))
        return res

    # find the best fit for sigma
    sig_0 = optimize.minimize(fval, 1, args=(rot_mat, r_sc, r_0, l_))
    # to make sure it finds the largest sigma
    sig_0 = optimize.minimize(fval, 2 * sig_0.x[0], args=(rot_mat, r_sc, r_0, l_))

    # calculate normal
    r_p0 = np.matmul(rot_mat, r_sc) - sig_0.x[0] * r_0

    # gradient to model surface
    grad_s_x = (r_p0[0] * (1 - eps**2) + eps * sig_0.x[0] * l_) * np.cos(
        np.deg2rad(alpha)
    ) + r_p0[1] * np.sin(np.deg2rad(alpha))
    grad_s_y = -(r_p0[0] * (1 - eps**2) + eps * sig_0.x[0] * l_) * np.sin(
        np.deg2rad(alpha)
    ) + r_p0[1] * np.cos(np.deg2rad(alpha))
    grad_s_z = r_p0[2]

    grad_s = np.array([grad_s_x, grad_s_y, grad_s_z], dtype=np.float32)
    grad_s /= np.linalg.norm(r_p0, keepdims=True) * 2 * sig_0.x[0] * l_

    # normal vector
    n = grad_s / np.linalg.norm(grad_s, keepdims=True)

    return n, sig_0.x[0]


def _shock_speed(spec, n, theta_bn, method):
    if method.lower() == "gt" and "f_cp" in spec and "dt_f" in spec and "d2u" in spec:
        v_sh = _speed_gosling_thomsen(spec, n, theta_bn)
    elif method.lower() == "mf":
        v_sh = _speed_mass_flux(spec, n)
    elif method.lower() == "sb":
        v_sh = _speed_smith_burton(spec, n)
    elif method.lower() == "mo" and "f_cp" in spec and "dt_f" in spec and "d2u" in spec:
        v_sh = _speed_moses(spec, n, theta_bn)
    else:
        v_sh = {k: 0.0 for k in n}

    return v_sh


def _speed_gosling_thomsen(spec, n, theta_bn):
    v_sh = {}

    for k, nvec in n.items():
        theta = np.deg2rad(theta_bn[k])
        nvec = n[k]

        # Notation as in (Gosling and Thomsen 1985)
        w = 2 * np.pi * spec["f_cp"]
        t1 = np.arccos((1 - 2 * np.cos(theta) ** 2) / (2 * np.sin(theta) ** 2)) / w
        x_0 = w * t1 * (2 * np.cos(theta) ** 2 - 1) + 2 * np.sin(theta) ** 2 * np.sin(
            w * t1
        )
        x_0 /= w * spec["dt_f"]

        # the sign of Vsh in this method is ambiguous, assume n points upstream
        v_sh[k] = (
            spec["d2u"] * np.sum(spec["v_u"] * nvec) * (x_0 / (1 + spec["d2u"] * x_0))
        )

    return v_sh


def _speed_mass_flux(spec, n):
    rho_u = spec["n_u"] * constants.proton_mass
    rho_d = spec["n_d"] * constants.proton_mass

    v_sh = {}

    for k, nvec in n.items():
        v_sh[k] = (
            rho_d * np.sum(spec["v_d"] * nvec) - rho_u * np.sum(spec["v_u"] * nvec)
        ) / (rho_d - rho_u)

    return v_sh


def _speed_smith_burton(spec, n):
    v_sh = {}

    for k, nvec in n.items():
        v_sh[k] = np.sum(spec["v_u"] * nvec) + np.linalg.norm(
            np.cross(spec["delta_v"], spec["b_d"])
        ) / np.linalg.norm(spec["delta_b"])

    return v_sh


def _speed_moses(spec, n, theta_bn):
    v_sh = {}

    for k, nvec in n.items():
        theta = np.deg2rad(theta_bn[k])
        theta_vn = np.arccos(np.sum(nvec * spec["v_u"]) / np.linalg.norm(spec["v_u"]))

        # Notation as in (Moses et al., 1985)
        w = 2 * np.pi * spec["f_cp"]
        x = 0.68 * np.sin(theta) ** 2 * np.cos(theta_vn) / (w * spec["dt_f"])

        v_sh[k] = np.sum(spec["v_u"] * nvec) * (x / (1 + spec["d2u"] * x))

    return v_sh


def _constraint_matrix(spec, n):
    u_vecs = [
        spec["delta_b"],
        np.cross(spec["b_d"], spec["b_u"]),
        np.cross(spec["b_u"], spec["delta_v"]),
        np.cross(spec["b_d"], spec["delta_v"]),
        np.cross(spec["delta_b"], spec["delta_v"]),
    ]

    c_mat = np.zeros((len(u_vecs), len(n)))

    for i, u in enumerate(u_vecs):
        for j, nvec in enumerate(n.values()):
            c_mat[i, j] = np.sum(u * nvec) / np.linalg.norm(u)

    # fields that are by definition zero are set to 0
    c_mat[[0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 4], [0, 2, 3, 4, 0, 1, 2, 1, 3, 1, 4]] = 0.0

    return c_mat
