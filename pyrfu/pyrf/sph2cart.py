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

"""sph2cart.py
@author: Louis Richard
"""

import numpy as np


def sph2cart(azimuth, elevation, r):
    """Transform spherical to cartesian coordinates

    Parameters
    ----------
    azimuth : float or ndarray
        Azimuthal angle (phi)

    elevation : float or ndarray
        Elevation angle (theta)

    r : float or ndarray
        Radius

    Returns
    -------
    x : float or ndarray
        Cartesian x-axis coordinates.

    y : float or ndarray
        Cartesian y-axis coordinates.

    z : float or ndarray
        Cartesian z-axis coordinates

    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return x, y, z