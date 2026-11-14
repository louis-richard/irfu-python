#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
from typing import cast

# 3rd party imports
import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


surface_materials = [
    "cluster",
    "themis",
    "cassini",
    "aluminium",
    "aquadag",
    "gold",
    "graphite",
    "solar cells",
    "1eV",
    "TiN",
    "elgiloy",
]

j_zeros = {
    "cassini": 25e-6,
    "tin": 25e-6,
    "cluster": 25e-6,
    "aluminium": 30e-6,
    "aquadag": 18e-6,
    "gold": 29e-6,
    "graphite": 7.2e-6,
    "solar cells": 20e-6,
    "solar cell": 20e-6,
    "elgiloy": 30e-6,
}


def photo_current(
    iluminated_area: float | None = None,
    u: float | np.ndarray | None = None,
    distance_sun: float | None = None,
    flag: str | float = "cluster",
) -> float | np.ndarray | None:
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

    Raises
    ------
    TypeError
        If flag is not a surface material or a photoemission.
    ValueError
        If some but not all of iluminated_area, u and distance_sun are given.

    Notes
    -----
    Estimates are done for the solar minimum conditions.

    """

    if not isinstance(flag, (str, float, int)):
        raise TypeError("flag must be a surface material or a photoemission")

    # Without arguments, the photoemission of the surface materials (an array
    # u with a zero area raised "truth value of an array is ambiguous", and
    # zeros for all three printed the table)
    args = (iluminated_area, u, distance_sun)

    if all(arg is None for arg in args):
        for surf in surface_materials:
            j0 = cast(np.ndarray, photo_current(1, 0, 1, surf))
            logger.info(
                "%(surf)s: Io= %(i0)3.2f uA/m2",
                {"surf": surf, "i0": float(j0[0]) * 1e6},
            )

        return None

    if any(arg is None for arg in args):
        raise ValueError("give iluminated_area, u and distance_sun, or none of them")

    # Given (checked above)
    iluminated_area = cast(float, iluminated_area)
    distance_sun = cast(float, distance_sun)

    # Assert than u is an array
    u = np.atleast_1d(u)

    if isinstance(flag, (float, int)):
        photoemisson = flag

        # Initialize
        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= photoemisson * iluminated_area / distance_sun**2

        a_ = 5.0e-5 / 5.6e-5 * np.exp(-u[u >= 0.0] / 2.74)
        b_ = 1.2e-5 / 5.6e-5 * np.exp(-(u[u >= 0] + 10.0) / 14.427)
        j_photo[u >= 0] *= a_ + b_

    elif flag.lower() == "1ev":
        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= 5.0e-5 * iluminated_area / distance_sun**2

        j_photo[u >= 0] *= np.exp(-u[u >= 0])

    elif flag.lower() == "themis":
        ref_u = np.array([0.1, 1, 5, 10, 50])
        ref_j_photo = np.array([50, 27, 10, 5, 0.5]) * 1e-6
        log_u = np.log(ref_u)
        log_j = np.log(ref_j_photo)

        # Initialize
        j_photo = np.ones(u.shape)

        # initialize to current valid for negative potentials
        j_photo *= iluminated_area / distance_sun**2

        f_ = interpolate.PchipInterpolator(log_u, log_j, extrapolate=None)
        j_photo[u >= ref_u[0]] *= np.exp(f_(np.log(u[u >= ref_u[0]])))
        j_photo[u < ref_u[0]] *= ref_j_photo[0]

    elif flag.lower() in j_zeros:
        j_photo = cast(
            np.ndarray, photo_current(iluminated_area, u, distance_sun, "themis")
        )
        j_photo *= j_zeros[flag.lower()] / cast(
            np.ndarray, photo_current(1.0, 0.0, 1.0, "themis")
        )

    else:
        raise ValueError("Unknown surface material.")

    return j_photo
