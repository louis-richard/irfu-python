#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np

from scipy.special import erf
from scipy.constants import elementary_charge, Boltzmann

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _spherical_body(z, u, x_, i_p):
    j_thermal = np.zeros(u.shape)

    if z > 0:
        j_thermal[u >= 0] = i_p * np.exp(-x_[u >= 0])
        j_thermal[u < 0] = i_p * (1 - x_[u < 0])
    elif z < 0:
        j_thermal[u >= 0] = i_p * (1 + x_[u >= 0])
        j_thermal[u < 0] = i_p * np.exp(x_[u < 0])

    return j_thermal


def _cylindrical_body(z, u, x_, i_p):
    sq = np.zeros(u.shape)
    j_thermal = np.zeros(u.shape)

    sq[u < 0] = np.sqrt(abs(-x_[u < 0]))
    sq[u >= 0] = np.sqrt(abs(+x_[u >= 0]))
    erfv = erf(sq)

    if z > 0:
        j_thermal[u >= 0] = i_p * np.exp(-x_[u >= 0])
        c_0 = (2.0 / np.sqrt(np.pi)) * sq[u < 0]
        c_1 = np.exp(-x_[u < 0]) * (1.0 - erfv[u < 0])
        j_thermal[u < 0] = i_p * (c_0 + c_1)
    elif z < 0:
        j_thermal[u < 0] = i_p * np.exp(x_[u < 0])

        c_0 = (2.0 / np.sqrt(np.pi)) * sq[u >= 0]
        c_1 = np.exp(x_[u >= 0]) * (1.0 - erfv[u >= 0])
        j_thermal[u >= 0] = i_p * (c_0 + c_1)

    return j_thermal


def thermal_current(n: float, t: float, m: float, v: float, z: float,
                    u: Union[float, np.ndarray], a: float,
                    p_type: str) -> Union[float, np.ndarray]:
    r"""Calculates the thermal probe current to/from a cylindrical or
    spherical body, e.g. a Langmuir probe or the a spherical (cylindrical) S/C.

    Parameters
    ----------
    n : float
        Number density [m^3].
    t : float
        Temperature [K]
    m : float
        Mass [kg].
    z : {-1, 1}
        Charge
    v : float
        Velocity of the body with respect to the plasma [m/s].
    u : float or numpy.ndarray
        Body potential [V]
    a : float
        Area of body [m^2].
    p_type : {"sphere", "cylinder"}
        Probe type.

    Returns
    -------
    j_thermal : float or numpy.ndarray

    """

    assert p_type.lower() in ["sphere", "cylinder"]

    u = np.atleast_1d(u)

    # If zero density return zero current
    if n == 0 or t == 0:
        return

    # Is the body moving with a velocity, V, with respect to the plasma ?
    # Criteria set such that it is considered important if V > 0.1 * V_th.
    if v < 0.1 * np.sqrt(Boltzmann * t / m):
        # Ratio of potential to thermal energy.
        x_ = elementary_charge * u / (Boltzmann * t)

        # Total current to/from body.
        i_p = np.sqrt(t * Boltzmann / (2.0 * np.pi * m))
        i_p *= a * n * elementary_charge

    else:
        x_ = (elementary_charge / (m * v ** 2 / 2 + Boltzmann * t)) * u
        i_p = np.sqrt(v ** 2 / 16 + t * Boltzmann / (2.0 * np.pi * m))
        i_p *= a * n * elementary_charge

    if p_type == "sphere":
        # Spherical body case.
        j_thermal = _spherical_body(z, u, x_, i_p)

    else:
        # Cylindrical body case.
        j_thermal = _cylindrical_body(z, u, x_, i_p)

    return j_thermal
