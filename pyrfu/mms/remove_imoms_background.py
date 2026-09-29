#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def remove_imoms_background(n_i, v_gse_i, p_gse_i, n_bg_i, p_bg_i):
    r"""Remove the mode background population due to penetrating radiation
    from the the moments (density `n_i`, bulk velocity `v_gse_i` and
    pressure tensor `p_gse_i`) of the ion velocity distribution function
    using the method from [1]_.

    Parameters
    ----------
    n_i : xarray.DataArray
        Time series of the ion density in cm^-3.
    v_gse_i : xarray.DataArray
        Time series of the ion bulk velocity in km/s.
    p_gse_i : xarray.DataArray
        Time series of the ion pressure tensor in nPa.
    n_bg_i : xarray.DataArray
        Time series of the background ion number density in cm^-3. Resampled to
        the time line of `n_i` if needed.
    p_bg_i : xarray.DataArray
        Time series of the background ion pressure scalar in nPa. Resampled to
        the time line of `n_i` if needed.

    Returns
    -------
    n_i_new : xarray.DataArray
        Time series of the corrected ion number density.
    v_gse_i_new : xarray.DataArray
        Time series of the corrected ion bulk velocity.
    p_gse_i : xarray.DataArray
        Time series of the corrected ion pressure tensor.

    References
    ----------
    .. [1]  Gershman, D. J., Dorelli, J. C., Avanov,L. A., Gliese, U., Barrie,
            A., Schiff, C.,et al. (2019). Systematic uncertainties in plasma
            parameters reported by the fast plasma investigation on NASA's
            magnetospheric multiscale mission. Journal of Geophysical
            Research: Space Physics, 124, https://doi.org/10.1029/2019JA026980

    """

    m_p = constants.proton_mass

    # m_p * n * v_i * v_j with n in cm^-3 and v in km/s, in nPa
    # (1e6 cm^-3 -> m^-3, 1e6 (km/s)^2 -> (m/s)^2, 1e9 Pa -> nPa)
    dyn_fact = m_p * 1e21

    # Background moments on the time line of the moments
    if not np.array_equal(n_bg_i.time.data, n_i.time.data):
        n_bg_i = resample(n_bg_i, n_i)

    if not np.array_equal(p_bg_i.time.data, n_i.time.data):
        p_bg_i = resample(p_bg_i, n_i)

    # Correct the ion number density
    n_i_new = n_i - n_bg_i.data

    # Mask the non-positive values of the corrected ion number density
    mask = n_i_new.data <= 0.0
    n_i_new.data[mask] = np.nan

    # Correct the ion bulk velocity
    v_gse_i_new = v_gse_i.copy()
    v_gse_i_new.data *= n_i.data[:, None] / n_i_new.data[:, None]

    # Mask the ion bulk velocity where the corrected ion number density is negative
    v_gse_i_new.data[mask, :] = np.nan

    # Correct the ion pressure tensor
    p_gse_i_new = np.zeros(p_gse_i.shape)
    n_old, v_old = [n_i.data, v_gse_i.data]
    n_new, v_new = [n_i_new.data, v_gse_i_new.data]

    for i, j in zip([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]):
        p_gse_i_new[:, i, j] += p_gse_i.data[:, i, j]
        p_gse_i_new[:, i, j] += dyn_fact * n_old * v_old[:, i] * v_old[:, j]
        p_gse_i_new[:, i, j] -= dyn_fact * n_new * v_new[:, i] * v_new[:, j]

    # Remove isotropic background pressure
    p_bkg_mat = np.tile(np.eye(3, 3), (len(p_bg_i.data), 1, 1))
    p_bkg_mat *= p_bg_i.data[:, None, None]
    p_gse_i_new -= p_bkg_mat

    # Fill the lower left off diagonal terms using symetry of the
    # pressure tensor
    p_gse_i_new[:, 1, 0] = p_gse_i_new[:, 0, 1]
    p_gse_i_new[:, 2, 0] = p_gse_i_new[:, 0, 2]
    p_gse_i_new[:, 2, 1] = p_gse_i_new[:, 1, 2]

    # Mask the pressure tensor where the density is negative
    p_gse_i_new[mask, :, :] = np.nan

    # Create time series of the ion pressure tensor
    p_gse_i_new = ts_tensor_xyz(
        p_gse_i.time.data, p_gse_i_new, attrs=dict(p_gse_i.attrs)
    )

    return n_i_new, v_gse_i_new, p_gse_i_new
