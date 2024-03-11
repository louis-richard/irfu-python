#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def remove_idist_background(vdf, def_bg):
    r"""Remove the mode background population due to penetrating radiation
    `def_bg` from the ion velocity distribution function `vdf` using the
    method from [1]_.

    Parameters
    ----------
    vdf : xarray.Dataset
        Ion velocity distribution function.
    def_bg : xarray.DataArray
        Omni-directional ion differential energy flux.

    Returns
    -------
    vdf_new : xarray.Dataset
        Ion velocity distribution function cleaned.

    References
    ----------
    .. [1]  Gershman, D. J., Dorelli, J. C., Avanov,L. A., Gliese, U.,
            Barrie, A., Schiff, C.,et al. (2019). Systematic uncertainties
            in plasma parameters reported by the fast plasma investigation on
            NASA's magnetospheric multiscale mission. Journal of Geophysical
            Research: Space Physics, 124, https://doi.org/10.1029/2019JA026980

    """

    # Tile background flux to number of energy channels of the
    # FPI-DIS instrument
    def_bg_tmp = np.tile(def_bg.data[:, np.newaxis], (1, vdf.energy.shape[1]))

    # Convert differential energy flux (cm^2 s sr)^{-1} of the background
    # population (penetrating radiations) to phase-space density (s^3 m^{-6})
    coeff = constants.proton_mass / (constants.elementary_charge * vdf.energy.data)
    vdf_bg = def_bg_tmp.copy() * 1e4 / 2
    vdf_bg *= coeff**2
    vdf_bg /= 1e12

    # Tile the background phase-space density to number of azimuthal
    # and elevation angles channels of the FPI-DIS instrument
    vdf_bg = np.tile(
        vdf_bg[:, :, np.newaxis, np.newaxis],
        (1, 1, vdf.phi.shape[1], vdf.theta.shape[0]),
    )

    vdf_new = vdf.copy()
    vdf_new.data.data -= vdf_bg.data

    return vdf_new
