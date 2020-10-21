#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
solid_angle.py

@author : Louis RICHARD
"""

import numpy as np


def solid_angle(x=None, y=None, z=None):
    """Calculates the solid angle of three vectors making up a triangle in a unit sphere with the
    sign taken into account.

    Parameters
    ----------
    x : numpy.ndarray
        First vector.

    y : numpy.ndarray
        Second vector.

    z : numpy.ndarray
        Third vector.

    Returns
    -------
    angle : float
        Solid angle.

    """

    assert x is not None and isinstance(x, np.ndarray) and x.ndim == 1 and len(x) == 3
    assert y is not None and isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == 3
    assert z is not None and isinstance(z, np.ndarray) and z.ndim == 1 and len(z) == 3

    # Calculate the smaller angles between the vectors around origin
    a = np.arccos(np.sum(z * y))
    b = np.arccos(np.sum(x * z))
    c = np.arccos(np.sum(y * x))

    # Calculate the angles in the spherical triangle (Law of Cosines)
    u = np.arccos((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)))
    v = np.arccos((np.cos(b) - np.cos(a) * np.cos(c)) / (np.sin(a) * np.sin(c)))
    w = np.arccos((np.cos(c) - np.cos(b) * np.cos(a)) / (np.sin(b) * np.sin(a)))

    # Calculates the Surface area on the unit sphere (solid angle)
    angle = (u + v + w - np.pi)
    # Calculate the sign of the area
    var = np.cross(z, y)
    div = np.sum(var * x)
    sgn = np.sign(div)

    # Solid angle with sign taken into account
    angle = sgn * angle

    return angle
