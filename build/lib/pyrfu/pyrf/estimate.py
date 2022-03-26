#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _estimate_capa_disk(radius):
    return 8 * constants.epsilon_0 * radius


def _estimate_capa_sphe(radius):
    return 4 * pi * constants.epsilon_0 * radius


def _estimate_capa_wire(radius, length):
    if not radius or radius == 0 or not lenght:
        return None
    elif length and radius and length >= 10 * radius:
        l_ = np.log(length / radius)
        out = length / l_ * (1 + 1 / l_ * (1 - np.log(2)))
        return 2 * pi * constants.epsilon_0 * out
    else:
        raise ValueError("capacitance_wire requires length at least 10 times "
                         "the radius!")


def _estimate_capa_cyli(a, h):
    coef = 4 * np.pi ** 2 * a * constants.epsilon_0

    if .5 < h / a < 4:
        c_1 = h / (2. * a * (np.log(16 * h / a) ** 2 + np.pi ** 2 / 12))
        out = coef * np.pi * c_1
    elif h / a >= 4:
        o_m = 2 * (np.log(4 * h / a) - 1)
        c_1 = 2 * h / (np.pi * a) * (1. / o_m + (4 - np.pi ** 2) / o_m ** 3)
        out = coef * c_1
    else:
        raise ValueError("length less than diameter, do not have formula yet")

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
            * "capacitance_cylinder" estimates the capacitance of a cylinder
            (requires radius and half length of the cylinder).
    radius :  float
        Radius of the disk, sphere, wire or cylinder
    length : float, Optional
        Length of the wire or half lenght of the cylinder.

    Returns
    -------
    out : float
        Estimated value.

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
        raise NotImplementedError

    return out
