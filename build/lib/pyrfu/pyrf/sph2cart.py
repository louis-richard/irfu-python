#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def sph2cart(azimuth, elevation, r):
    r"""Transform spherical to cartesian coordinates

    Parameters
    ----------
    azimuth : float or ndarray
        Azimuthal angle (phi).
    elevation : float or ndarray
        Elevation angle (theta)
    r : float or ndarray
        Radius.

    Returns
    -------
    x : float or ndarray
        Cartesian x-axis coordinates.
    y : float or ndarray
        Cartesian y-axis coordinates.
    z : float or ndarray
        Cartesian z-axis coordinates

    """

    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)

    return x, y, z
