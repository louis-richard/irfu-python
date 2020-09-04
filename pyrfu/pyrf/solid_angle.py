#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
solid_angle.py

@author : Louis RICHARD
"""

import numpy as np


def solid_angle(x=None, y=None, z=None):
    """
    Calculates the solid angle of three vectors making up a triangle in a unit sphere with the sign taken into account

    Parameters :
        x : array
            First vector
        y : array
            Second vector
        z : array
            Third vector

    Return :
        angle : float
            Solid angle

    """

    if x is None or y is None or z is None:
        raise ValueError("solidangle requires at least 3 arguments")

    # Check x is a vector
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a np.ndarray")
    elif x.ndim != 1:
        raise TypeError("x must be a one dimension np.ndarray")
    elif np.size(x) != 3:
        raise TypeError("x must have only 3 components")

    # Check y is a vector
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a np.ndarray")
    elif y.ndim != 1:
        raise TypeError("y must be a one dimension np.ndarray")
    elif np.size(y) != 3:
        raise TypeError("y must have only 3 components")

    # Check z is a vector
    if not isinstance(z, np.ndarray):
        raise TypeError("z must be a np.ndarray")
    elif z.ndim != 1:
        raise TypeError("z must be a one dimension np.ndarray")
    elif np.size(z) != 3:
        raise TypeError("z must have only 3 components")

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
    signarea = np.sign(div)

    # Solid angle with sign taken into account
    angle = signarea * angle

    return angle
