#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import numpy as np


def solid_angle(x, y, z):
    """Calculates the solid angle of three vectors making up a triangle in a unit sphere with the
    sign taken into account.

    Parameters
    ----------
    x : ndarray
        First vector.

    y : ndarray
        Second vector.

    z : ndarray
        Third vector.

    Returns
    -------
    angle : float
        Solid angle.

    """

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
