#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def match_phibe_v(b_0, b_z, int_e_dt, n, v):
    r"""Get propagation velocity by matching dBpar and phi. Used together with
    match_phibe_dir. Finds best match in amplitude given, B0, dB_par, phi,
    propagation direction implied, for specified n and v given as vectors.
    Returns a matrix of correlations and the two potentials that were
    correlated.

    As irf_match_phibe_v.m, the "correlation" is the sum over time of
    :math:`\log_{10} |\phi_E / \phi_B|`, i.e., the number of samples times
    the mean logarithmic ratio of the potentials: the best match in amplitude
    is where it is closest to zero. Since :math:`\phi_E / \phi_B` is
    proportional to :math:`n v`, only the product of the density and the
    velocity is determined.

    Parameters
    ----------
    b_0 : float
        Average background magnetic field [nT].
    b_z : array_like
        Parallel wave magnetic field [nT].
    int_e_dt : array_like
        Potential, time integral of the electric field in the propagation
        direction [mV/m s] (e.g., the best direction from match_phibe_dir).
    n : array_like
        Vector of densities [cm^{-3}].
    v : array_like
        Vector of velocities [km/s].

    Returns
    -------
    corr_mat : numpy.ndarray
        Correlation matrix(nn x nv).
    phi_b : numpy.ndarray
        B0 * dB_par / n_e * e * mu0 [V] (size: n_times x nn).
    phi_e : numpy.ndarray
        int(E) dt * v(dl=-vdt = > -dl = vdt) [V] (size: n_times x nv).

    """

    # Define constants
    mu0 = constants.mu_0
    q_e = constants.elementary_charge

    # density in #/m^3
    n_si = 1e6 * np.atleast_1d(np.asarray(n, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    b_z = np.ravel(np.asarray(b_z, dtype=np.float64))
    int_e_dt = np.ravel(np.asarray(int_e_dt, dtype=np.float64))

    # Setup potentials
    phi_e = np.outer(int_e_dt, v)  # depends on v
    phi_b = np.outer(b_z * b_0 * 1e-18 / (mu0 * q_e), 1 / n_si)  # depends on n

    # Get correlation, rows: n, cols: v
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(np.abs(phi_e[:, None, :] / phi_b[:, :, None]))

    corr_mat = np.sum(log_ratio, axis=0)

    return corr_mat, phi_b, phi_e
