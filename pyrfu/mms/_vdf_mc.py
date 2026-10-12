#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from ._vdf_bins import (
    _energy_edges,
    _kinetic_energy,
    _mass,
    _phi_edges,
    _relativistic,
    _sc_pot,
    _speed,
    _theta_edges,
)

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Monte-Carlo particles in the bins of a skymap, shared by vdf_macroparticles
# and vdf_shift (irfu-matlab PDist.macroparticles and PDist.shift)


def _rng(random_state):
    r"""Random generator from None, a seed or a numpy.random.Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state

    return np.random.default_rng(random_state)


def _source_bins(vdf, sc_pot=None, relativistic=None):
    r"""Edges of the bins of a skymap: speeds (m/s, time x energy), azimuths
    (degrees, time x phi) and cosines of the polar angles (theta,)."""
    sc_pot = _sc_pot(vdf, sc_pot)
    mass = _mass(vdf)
    relativistic = _relativistic(vdf, relativistic)
    v_lower, v_upper = (
        _speed(_kinetic_energy(vdf, edge, sc_pot), mass, relativistic)
        for edge in _energy_edges(vdf)
    )
    phi_lower, phi_upper = _phi_edges(vdf)
    theta_lower, theta_upper = _theta_edges(vdf)
    return {
        "v_lower": v_lower,
        "v_upper": v_upper,
        "phi_lower": np.asarray(phi_lower),
        "phi_upper": np.asarray(phi_upper),
        "cos_upper": np.cos(np.deg2rad(theta_lower)),
        "cos_lower": np.cos(np.deg2rad(theta_upper)),
    }


def _sample(rng, bins, i_t, i_e, i_p, i_q):
    r"""Particle velocities (m/s, n x 3) drawn uniformly in the volume of the
    bins (i_t, i_e, i_p, i_q) of `_source_bins`: v^3, cos(theta) and phi
    uniform. The particle moves opposite to the look direction."""
    v_3 = bins["v_lower"][i_t, i_e] ** 3
    d_v3 = bins["v_upper"][i_t, i_e] ** 3 - v_3
    speed = np.cbrt(v_3 + rng.random(len(i_t)) * d_v3)

    cos_lower = bins["cos_lower"][i_q]
    cos_theta = cos_lower + rng.random(len(i_t)) * (bins["cos_upper"][i_q] - cos_lower)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, None))

    phi_lower = bins["phi_lower"][i_t, i_p]
    phi = phi_lower + rng.random(len(i_t)) * (bins["phi_upper"][i_t, i_p] - phi_lower)
    phi = np.deg2rad(phi)

    return -speed[:, np.newaxis] * np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1
    )


def _speed_to_energy(speed, mass, relativistic):
    r"""Kinetic energy (eV) of particles of speed (m/s)."""
    if relativistic:
        # gamma - 1 = beta^2 / (s (1 + s)), s = sqrt(1 - beta^2), without
        # cancellation
        beta2 = (speed / constants.c) ** 2
        root = np.sqrt(1.0 - beta2)
        gamma_1 = beta2 / (root * (1.0 + root))
        return gamma_1 * mass * constants.c**2 / constants.elementary_charge

    return 0.5 * mass * speed**2 / constants.elementary_charge
