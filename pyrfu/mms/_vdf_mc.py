#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Monte-Carlo particles in the bins of a skymap, shared by vdf_macroparticles
# and vdf_shift (irfu-matlab PDist.macroparticles and PDist.shift)


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


def _speed_to_energy(speed, mass):
    r"""Kinetic energy (eV) of particles of relativistic speed (m/s); it was
    non-relativistic without the ``relativistic`` attribute."""
    # gamma - 1 = beta^2 / (s (1 + s)), s = sqrt(1 - beta^2), without
    # cancellation
    beta2 = (np.asarray(speed) / constants.c) ** 2
    root = np.sqrt(1.0 - beta2)
    gamma_1 = beta2 / (root * (1.0 + root))
    return gamma_1 * mass * constants.c**2 / constants.elementary_charge
