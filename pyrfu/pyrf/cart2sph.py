#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 - 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""cart2sph.py
@author: Louis Richard
"""

import numpy as np


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`


    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates

    y : float or array_like
        y-component of Cartesian coordinates

    z : float or array_like
        z-component of Cartesian coordinates


    Returns
    -------
    alpha : float or array_like
        Azimuth angle in radiants

    beta : float or array_like
        Elevation angle in radiants (with 0 denoting North pole)

    r : float or array_like
        Radius
    """

    # Radius
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuthal angle
    alpha = np.arctan2(y, x)

    # Elevation angle
    beta = np.arccos(z / r)

    return alpha, beta, r
