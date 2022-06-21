#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np

from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def match_phibe_v(b_0, b_z, int_e_dt, n, v):
    r"""Get propagation velocity by matching dBpar and phi. Used together with
    irf_match_phibe_dir.m.Finds best match in amplitude given, B0, dB_par,
    phi, propagation direction implied, for specified n and v given as
    vectors.Returns a matrix of correlations and the two potentials that were
    correlated.

    Parameters
    ----------
    b_0 : array_like
        Average background magnetic field.
    b_z : array_like
        Parallel wave magnetic field.
    int_e_dt : array_like
        Potential.
    n : array_like
        Vector of densities
    v : array_like
        Vector of velocities.

    Returns
    -------
    corr_mat : numpy.ndarray
        Correlation matrix(nn x nv).
    phi_b : numpy.ndarray
        B0 * dB_par / n_e * e * mu0
    phi_e : numpy.ndarray
        int(E) dt * v(dl=-vdt = > -dl = vdt)

    """

    # Define constants
    mu0 = constants.mu_0
    q_e = constants.elementary_charge

    # density in #/m^3
    n.data *= 1e6

    # Allocate correlations matrix rows: n, cols: v
    nn_, nv_ = [len(n), len(v)]
    corr_mat = np.zeros((nn_, nv_))

    # Setup potentials
    phi_e = int_e_dt * v  # depends on v
    phi_b = np.transpose(b_z[:, 0] * b_0 * 1e-18 / (mu0 * q_e * n[:, None]))

    # Get correlation
    for k, p in itertools.product(range(nn_), range(nv_)):
        corr_mat[k, p] = np.sum((np.log10(abs(phi_e[:, p]) / phi_b[:, k])))

    return corr_mat, phi_b, phi_e
