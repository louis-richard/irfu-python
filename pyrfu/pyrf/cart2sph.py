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


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \mathrm{atan2}(y, x) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\alpha \in (-\pi, \pi], \beta \in [0, \pi], r \geq 0`

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
        Azimuth angle in radians, from the x axis towards the y axis, between
        -pi and pi.
    beta : float or array_like
        Polar angle in radians, from the z axis (the colatitude: 0 on the z
        axis). MATLAB's cart2sph returns the elevation from the xy plane,
        pi / 2 - beta.
    r : float or array_like
        Radius

    See Also
    --------
    pyrfu.pyrf.sph2cart

    """

    # Radius
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuthal angle
    alpha = np.arctan2(y, x)

    # Elevation angle
    beta = np.arccos(z / r)

    return alpha, beta, r
