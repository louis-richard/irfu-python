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

"""solid_angle.py
@author: Louis Richard
"""

import numpy as np


def solid_angle(inp0, inp1, inp2):
    """Calculates the solid angle of three vectors making up a triangle
    in a unit sphere with the sign taken into account.

    Parameters
    ----------
    inp0 : ndarray
        First vector.

    inp1 : ndarray
        Second vector.

    inp2 : ndarray
        Third vector.

    Returns
    -------
    angle : float
        Solid angle.

    """

    # Calculate the smaller angles between the vectors around origin
    acos_12 = np.arccos(np.sum(inp2 * inp1))
    acos_02 = np.arccos(np.sum(inp0 * inp2))
    acos_01 = np.arccos(np.sum(inp1 * inp0))

    # Calculate the angles in the spherical triangle (Law of Cosines)
    alpha = np.arccos((np.cos(acos_12) - np.cos(acos_02) * np.cos(acos_01))
                  / (np.sin(acos_02) * np.sin(acos_01)))
    beta = np.arccos((np.cos(acos_02) - np.cos(acos_12) * np.cos(acos_01))
                  / (np.sin(acos_12) * np.sin(acos_01)))
    gamma = np.arccos((np.cos(acos_01) - np.cos(acos_02) * np.cos(acos_12))
                  / (np.sin(acos_02) * np.sin(acos_12)))

    # Calculates the Surface area on the unit sphere (solid angle)
    angle = (alpha + beta + gamma - np.pi)
    # Calculate the sign of the area
    var = np.cross(inp2, inp1)
    div = np.sum(var * inp0)
    sgn = np.sign(div)

    # Solid angle with sign taken into account
    angle = sgn * angle

    return angle
