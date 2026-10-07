#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .calc_dt import calc_dt
from .filt import filt
from .integrate import integrate
from .norm import norm
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def match_phibe_dir(b_xyz, e_xyz, angles: np.ndarray = None, f: float = None):
    r"""Get propagation direction by matching dBpar and "phi". Tries different
    propagation directions and finds the direction perpendicular to the
    magnetic field that gives the best correlation between the electrostatic
    potential and the parallel wave magnetic field according to

    .. math::

            \int E \textrm{d}t = \frac{B_0}{ne \mu_0} B_{wave}

    Port of irf_match_phibe_dir.m, except that the angles are used (irfu-matlab
    only uses their number, spread over 360 degrees).

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field (to be filtered if f is given).
    e_xyz : xarray.DataArray
        Time series of the electric field (to be filtered if f is given).
    angles : array_like, Optional
        The angles in degrees to try, in the plane perpendicular to the mean
        magnetic field. Default is 1, 4, 7, ..., 358.
    f : float, Optional
        Filter frequency.

    Returns
    -------
    x : ndarray
        Propagation directions (size: n_tries x 3).
    y : ndarray
        Normal to the propagation directions and B (size: n_tries x 3).
    z : ndarray
        Magnetic field direction (size: n_tries x 3).
    corr_vec : ndarray
        Normalised zero-lag correlation of the potential and the parallel
        wave magnetic field for each direction.
    int_e_dt : ndarray
        Potential (time integral of the wave electric field) for each
        direction (size: n_times x n_tries).
    b_z : ndarray
        Wave magnetic field in parallel direction.
    b_0 : float
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

    # Get background magnetic field, for match_phibe_v
    b_0 = float(np.mean(norm(b_xyz).data))

    # If no angles are specified, set 1,4,7,...,358 as default
    if angles is None:
        angles = np.arange(1, 360, 3)

    theta = np.deg2rad(np.atleast_1d(angles).astype(np.float64))

    # Set up coordinate systems
    b_hat = np.mean(b_xyz.data, axis=0)
    b_hat /= np.linalg.norm(b_hat)
    perp_1 = np.cross(np.cross(b_hat, np.array([1, 0, 0])), b_hat)
    perp_1 /= np.linalg.norm(perp_1)
    perp_2 = np.cross(perp_1, b_hat)

    x_ = np.outer(np.cos(theta), perp_2) + np.outer(np.sin(theta), perp_1)
    z_ = np.tile(b_hat, (len(theta), 1))  # B / z direction, tries * 3
    y_ = np.cross(z_, x_)

    # Field aligned B
    b_z = b_ac.data @ b_hat

    # Fields in all directions
    de_k, de_n = [e_ac.data @ vec.T for vec in [x_, y_]]
    e_k, e_n = [e_xyz.data @ vec.T for vec in [x_, y_]]

    # Get Phi_E = int(Ek), there's no minus since the field is integrated
    # in the opposite direction of the wave propagation direction.
    int_e = integrate(ts_vec_xyz(e_xyz.time.data, e_ac.data), calc_dt(e_xyz))
    int_e_dt = int_e.data @ x_.T
    int_e_dt -= np.mean(int_e_dt, axis=0)

    # Normalised zero-lag correlation (xcorr(x, y, 0, "coeff"))
    corr_vec = b_z @ int_e_dt
    corr_vec /= np.sqrt(np.sum(int_e_dt**2, axis=0) * np.sum(b_z**2))

    return x_, y_, z_, corr_vec, int_e_dt, b_z, b_0, de_k, de_n, e_k, e_n
