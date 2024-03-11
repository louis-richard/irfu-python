#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
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
        Time series of the ion density.
    v_gse_i : xarray.DataArray
        Time series of the ion bulk velocity.
    p_gse_i : xarray.DataArray
        Time series of the ion pressure tensor.
    n_bg_i : xarray.DataArray
        Time series of the background ion number density.
    p_bg_i : xarray.DataArray
        Time series of the background ion pressure scalar.

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

    # Correct the ion number density
    n_i_new = n_i - n_bg_i.data

    # Correct the ion bulk velocity
    v_gse_i_new = v_gse_i.copy()
    v_gse_i_new.data *= n_i.data[:, None] / n_i_new.data[:, None]

    # Correct the ion pressure tensor
    p_gse_i_new = np.zeros(p_gse_i.shape)
    n_old, v_old = [n_i.data, v_gse_i.data]
    n_new, v_new = [n_i_new.data, v_gse_i_new.data]

    for i, j in zip([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]):
        p_gse_i_new[:, i, j] += p_gse_i.data[:, i, j]

        # TODO : use * instead of np.multiply??
        p_gse_i_new[:, i, j] += m_p * n_old * np.multiply(v_old[:, i], v_old[:, j])
        p_gse_i_new[:, i, j] -= m_p * n_new * np.multiply(v_new[:, i], v_new[:, j])

    # Remove isotropic background pressure
    p_bkg_mat = np.tile(np.eye(3, 3), (len(p_bg_i.data), 1, 1))
    p_bkg_mat *= p_bg_i.data[:, None, None]
    p_gse_i_new -= p_bkg_mat

    # Fill the lower left off diagonal terms using symetry of the
    # pressure tensor
    p_gse_i_new[:, 1, 0] = p_gse_i_new[:, 0, 1]
    p_gse_i_new[:, 2, 0] = p_gse_i_new[:, 0, 2]
    p_gse_i_new[:, 2, 1] = p_gse_i_new[:, 1, 2]

    # Create time series of the ion pressure tensor
    p_gse_i_new = ts_tensor_xyz(p_gse_i.time.data, p_gse_i_new)

    return n_i_new, v_gse_i_new, p_gse_i_new
