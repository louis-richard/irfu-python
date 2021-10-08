#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np

from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

surface_materials = ['cluster', 'themis', 'cassini', 'aluminium', 'aquadag',
                     'gold', 'graphite', 'solar cells', '1eV', 'TiN',
                     'elgiloy']

j_zeros = {"cassini": 25e-6, "tin": 25e-6, "cluster": 25e-6,
           "aluminium": 30e-6, "aquadag": 18e-6, "gold": 29e-6,
           "graphite": 7.2e-6, "solar cells": 20e-6, "solar cell": 20e-6,
           "elgiloy": 30e-6}


def photo_current(iluminated_area: float = None,
                  u: Union[float, np.ndarray] = None,
                  distance_sun: float = None,
                  flag: Union[str, float] = "cluster") -> Union[float,
                                                                np.ndarray]:
    r"""Calculates the photo-current emitted by an arbitrary body.

    Parameters
    ----------
    iluminated_area : float
        Cross section area [m^2].
    u : float or numpy.ndarray
        Potential [V].
    distance_sun : float
        Distance form the Sun [AU].
    flag : str or float, Optional
        Surface materials or surface photoemission in [A/m^2].
        Default is "cluster".

    Returns
    -------
    j_photo : float or numpy.ndarray
        Photo-current emitted.

    Notes
    -----
    Estimates are done for the solar minimum conditions.

    """

    assert isinstance(flag, (str, float))

    if not iluminated_area and not u and not distance_sun:
        for surf in surface_materials:
            j0 = photo_current(1, 0, 1, surf)
            print(f"{surf}: Io= {j0 * 1e6:3.2f} uA/m2")

        return

    # Assert than u is an array
    u = np.atleast_1d(u)

    if isinstance(flag, (float, int)):
        photoemisson = flag

        # Initialize
        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= photoemisson * iluminated_area / distance_sun ** 2

        a_ = 5.0e-5 / 5.6e-5 * np.exp(- u[u >= 0.] / 2.74)
        b_ = 1.2e-5 / 5.6e-5 * np.exp(- (u[u >= 0] + 10.0) / 14.427)
        j_photo[u >= 0] *= a_ + b_

    elif flag.lower() == "1ev":

        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= 5.0e-5 * iluminated_area / distance_sun ** 2

        j_photo[u >= 0] *= np.exp(- u[u >= 0])

    elif flag.lower() == "themis":
        ref_u = np.array([.1, 1, 5, 10, 50])
        ref_j_photo = np.array([50, 27, 10, 5, .5]) * 1e-6
        log_u = np.log(ref_u)
        log_j = np.log(ref_j_photo)

        # Initialize
        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= iluminated_area / distance_sun ** 2

        f_ = interpolate.PchipInterpolator(log_u, log_j, extrapolate=None)
        j_photo[u >= ref_u[0]] *= np.exp(f_(np.log(u[u >= ref_u[0]])))
        j_photo[u < ref_u[0]] *= ref_j_photo[0]

    elif flag.lower() in j_zeros:
        j_photo = photo_current(iluminated_area, u, distance_sun, "themis")
        j_photo *= j_zeros[flag] / photo_current(1., 0., 1., "themis")

    else:
        raise ValueError("Unknown surface material.")

    return j_photo
