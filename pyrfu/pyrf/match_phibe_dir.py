#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from scipy import signal

# Local imports
from .resample import resample
from .filt import filt
from .norm import norm
from .integrate import integrate
from .ts_scalar import ts_scalar
from .calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def match_phibe_dir(b_xyz, e_xyz, angles: np.ndarray = None, f: float = None):
    r"""Get propagation direction by matching dBpar and "phi". Tries different
    propagation directions and finds the direction perpendicular to the
    magnetic field that gives the best correlation between the electrostatic
    potential and the parallel wave magnetic field according to

    .. math::

            \int E \textrm{d}t = \frac{B_0}{ne \mu_0} B_{wave}


    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field (to be filtered if f is given).
    e_xyz : xarray.DataArray
        Time series of the electric field (to be filtered if f is given).
    angles : array_like, Optional
        The angles in degrees to try (1-180 default)
    f : float, Optional
        Filter frequency.

    Returns
    -------
    x : ndarray
        Normal direction (size: n_triesx3).
    y : ndarray
        Propagation direction.
    z : ndarray
        Magnetic field direction.
    corr_vec : ndarray
        Correlation vector.
    int_e_dt : ndarray
        Potential.
    b_z : ndarray
        Wave magnetic field in parallel direction.
    b_0 : ndarray
        Mean magnetic field.
    de_k : ndarray
        Wave electric field in propagation direction.
    de_n : ndarray
        Wave electric field in propagation normal direction.
    e_k : ndarray
        Electric field in propagation direction.
    e_n : ndarray
        Electric field in propagation normal direction.

    """

    # Resample B to E if they have different size
    b_xyz = resample(b_xyz, e_xyz)

    # Filter if f is given, otherwise assume it is filtered
    if f is not None:
        b_ac = filt(b_xyz, f, 0, 5)
        e_ac = filt(e_xyz, f, 0, 5)
    else:
        b_ac = b_xyz
        e_ac = e_xyz

    # Get background magnetic field, for irf_match_phibe_v
    b_0 = np.mean(norm(b_xyz))

    # If no angles are specified, set 1,4,7,...,158 as default
    if angles is None:
        angles = np.linspace(0, 360, 121)

    # number of angles
    na_ = len(angles)

    # Set up coordinate systems
    b_hat = np.mean(b_xyz.data, axis=0)
    b_hat /= np.linalg.norm(b_hat)
    y_ = np.cross(np.cross(b_hat, np.array([1, 0, 0])), b_hat)
    y_ /= np.linalg.norm(y_)
    x_ = np.cross(y_, b_hat)
    x_ /= np.linalg.norm(x_)
    x_ = np.tile(x_, (na_, 1))      # perp1
    y_ = np.tile(y_, (na_, 1))      # perp2
    z_ = np.tile(b_hat, (na_, 1))   # B / z direction, tries * 3

    theta = np.linspace(0, 2 * np.pi - np.pi / na_, na_)  # angles

    x_n = x_ * np.transpose(np.tile(np.cos(theta), (3, 1)))
    x_n += y_ * np.transpose(np.tile(np.sin(theta), (3, 1)))

    y_ = np.cross(z_, x_n)
    x_ = x_n

    # Field aligned B
    b_z = np.sum(b_ac * z_[0, :], axis=1)

    # Allocate correlations
    corr_vec = np.zeros(na_)

    # Allocate vectors, 4 first used for illustration
    int_e_dt = np.zeros((len(e_xyz), na_))                          # potential
    e_k, e_n = [np.zeros((len(e_xyz), na_)) for _ in range(2)]      #
    de_k, de_n = [np.zeros((len(e_xyz), na_)) for _ in range(2)]    #

    # Integrate E in all x - directions
    for k in range(na_):
        de_k[:, k] = np.sum(e_ac.data * x_[k, :], axis=1)
        de_n[:, k] = np.sum(e_ac.data * y_[k, :], axis=1)

        e_k[:, k] = np.sum(e_xyz.data * x_[k, :], axis=1)
        e_n[:, k] = np.sum(e_xyz.data * y_[k, :], axis=1)

        # Get Phi_E = int(Ek), there's no minus since the field is integrated
        # in the opposite direction of the wave propagation direction.
        prel_ = integrate(ts_scalar(e_xyz.time.data, de_k[:, k]),
                          calc_dt(e_xyz))
        int_e_dt[:, k] = prel_.data - np.mean(prel_.data)

        # Get correlation
        corr_ = signal.correlate(int_e_dt[:, k], b_z[:, 1], mode="valid")
        scale = np.sqrt(np.dot(int_e_dt[:, k], int_e_dt[:, k])
                        * np.dot(b_z[:, 1], b_z[:, 1]))
        corr_vec[k] = corr_ / scale

    return x_, y_, z_, corr_vec, int_e_dt, b_z, b_0, de_k, de_n, e_k, e_n
