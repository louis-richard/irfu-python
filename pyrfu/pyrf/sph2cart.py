#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def sph2cart(azimuth, elevation, r):
    r"""Transform spherical to cartesian coordinates, the inverse of
    cart2sph.

    .. math::

        x = r \sin\theta \cos\phi, \quad y = r \sin\theta \sin\phi,
        \quad z = r \cos\theta

    Parameters
    ----------
    azimuth : float or ndarray
        Azimuthal angle phi in radians, from the x axis towards the y axis.
    elevation : float or ndarray
        Polar angle theta in radians, from the z axis (the colatitude: 0 on
        the z axis, pi / 2 in the xy plane), despite its name. MATLAB's
        sph2cart takes the elevation from the xy plane, pi / 2 - theta.
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

    See Also
    --------
    pyrfu.pyrf.cart2sph

    Examples
    --------
    >>> from pyrfu import pyrf

    The z axis is at theta = 0

    >>> x, y, z = pyrf.sph2cart(0.0, 0.0, 1.0)

    """

    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)

    return x, y, z
