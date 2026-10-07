#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _estimate_capa_disk(radius):
    return 8 * constants.epsilon_0 * radius


def _estimate_capa_sphe(radius):
    return 4 * np.pi * constants.epsilon_0 * radius


def _estimate_capa_wire(radius, length):
    if length and radius != 0 and length >= 10 * radius:
        l_ = np.log(length / radius)
        out = length / l_ * (1 + 1 / l_ * (1 - np.log(2)))
        out *= 2 * np.pi * constants.epsilon_0
    else:
        raise ValueError(
            "capacitance_wire requires length at least 10 times the radius!",
        )
    return out


def _estimate_capa_cyli(a, h):
    # Verolino (1995), Eqs. 21-22, as in irf_estimate.m with h the half length,
    # corrected against a boundary element solution: irf_estimate.m has
    # pi * h / (2 * a) instead of pi * h / a and (4 - pi**2) instead of
    # (4 - pi**2 / 3), which are 51 % and 10-46 % too low.
    coef = 4 * np.pi**2 * a * constants.epsilon_0

    if 0.5 < h / a < 4:
        c_1 = np.pi * h / (a * (np.log(16 * h / a) ** 2 + np.pi**2 / 12))
        out = coef * c_1
    elif h / a >= 4:
        o_m = 2 * (np.log(4 * h / a) - 1)
        c_1 = 2 * h / (np.pi * a) * (1.0 / o_m + (4 - np.pi**2 / 3) / o_m**3)
        out = coef * c_1
    else:
        raise ValueError("half length must be larger than radius / 2")

    return out


def estimate(what_to_estimate: str, radius: float, length: float = None):
    r"""Estimate values for some everyday stuff.

    Parameters
    ----------
    what_to_estimate : str
        Value to estimate:

        * "capacitance_disk" estimates the capacitance of a disk
          (requires radius of the disk).
        * "capacitance_sphere" estimates of a sphere
          (requires radius of the sphere).
        * "capacitance_wire" estimates the capacitance of a wire
          (requires radius and length of the wire).
        * "capacitance_cylinder" estimates the capacitance of a solid
          cylinder (requires radius and half length of the cylinder, with
          half length larger than radius / 2), from [1]_. Compared with a
          boundary element solution, the error is less than 3 % for
          half length / radius < 4 or > 6, and up to +8 % in between, just
          above the switch between the two formulas at 4, which is kept from
          irf_estimate.m. Unlike irf_estimate.m, length is the half length.

    radius :  float
        Radius of the disk, sphere, wire or cylinder
    length : float, Optional
        Length of the wire or half length of the cylinder.

    Returns
    -------
    out : float
        Estimated value.

    Raises
    ------
    NotImplementedError
        If what_to_estimate is not one of the above.

    References
    ----------
    .. [1]  Verolino, L. (1995), Electrical Engineering, 78, 201-207,
            Eqs. 21-22.

    Examples
    --------
    >>> from pyrfu import pyrf

    Define radius of the sphere in SI units

    >>> r_sphere = 20e-2

    Computes the capacitance of the sphere

    >>> c_sphere = pyrf.estimate("capacitance_sphere", r_sphere)


    """
    if what_to_estimate.lower() == "capacitance_disk":
        out = _estimate_capa_disk(radius)
    elif what_to_estimate.lower() == "capacitance_sphere":
        out = _estimate_capa_sphe(radius)
    elif what_to_estimate.lower() == "capacitance_wire":
        out = _estimate_capa_wire(radius, length)
    elif what_to_estimate.lower() == "capacitance_cylinder":
        out = _estimate_capa_cyli(radius, length)
    else:
        raise NotImplementedError(f"unknown estimate {what_to_estimate!r}")

    return out
