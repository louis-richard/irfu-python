#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np

from scipy import constants

# Local imports
from ..pyrf import resample, ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def calculate_epsilon(vdf, model_vdf, n_s, sc_pot, **kwargs):
    r"""Calculates epsilon parameter using model distribution.

    Parameters
    ----------
    vdf : xarray.Dataset
        Observed particle distribution (skymap).
    model_vdf : xarray.Dataset
        Model particle distribution (skymap).
    n_s : xarray.DataArray
        Time series of the number density.
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    epsilon : xarray.DataArray
        Time series of the epsilon parameter.

    Other Parameters
    ----------------
    en_channels : array_like
        Set energy channels to integrate over [min max]; min and max between
        must be between 1 and 32.

    Examples
    --------
    >>> from pyrfu import mms
    >>> options = {"en_channel": [4, 32]}
    >>> eps = mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot, **options)

    """

    # Default energy channels used to compute epsilon, lowest energy channel
    # should not be used.
    energy_range = kwargs.get("en_channels", [2, 32])
    int_energies = np.arange(energy_range[0], energy_range[1] + 1)

    # Resample sc_pot
    sc_pot = resample(sc_pot, n_s)

    # Remove zero count points from final calculation
    # model_vdf.data.data[vdf.data.data <= 0] = 0

    model_vdf /= 1e18
    vdf /= 1e18

    vdf_diff = np.abs(vdf.data.data - model_vdf.data.data)

    # Define constants
    q_e = constants.elementary_charge

    # Check whether particles are electrons or ions
    if vdf.attrs["specie"] == "e":
        m_s = constants.electron_mass
        print("notice : Particles are electrons")
    elif vdf.attrs["specie"] == "i":
        sc_pot.data *= -1
        m_s = constants.proton_mass
        print("notice : Particles are electrons")
    else:
        raise ValueError("Invalid specie")

    # Define lengths of variables
    n_ph = len(vdf.phi.data[0, :])

    # Define corrected energy levels using spacecraft potential
    energy_arr = vdf.energy.data
    v_s, delta_v = [np.zeros(energy_arr.shape) for _ in range(2)]

    for i in range(len(vdf.time)):
        energy_vec = energy_arr[i, :]
        energy_log = np.log10(energy_arr[i, :])

        v_s[i, :] = np.real(
            np.sqrt(2 * (energy_vec - sc_pot.data[i]) * q_e / m_s))

        temp0 = 2 * energy_log[0] - energy_log[1]
        temp33 = 2 * energy_log[-1] - energy_log[-2]

        energy_all = np.hstack([temp0, energy_log, temp33])

        diff_en_all = np.diff(energy_all)

        energy_upper = 10 ** (energy_log + diff_en_all[1:34] / 2)
        energy_lower = 10 ** (energy_log - diff_en_all[0:33] / 2)

        v_upper = np.sqrt(2 * q_e * energy_upper / m_s)
        v_lower = np.sqrt(2 * q_e * energy_lower / m_s)

        v_lower[np.isnan(v_lower)] = 0
        v_upper[np.isnan(v_upper)] = 0

        delta_v[i, :] = (v_upper - v_lower)

    v_s[v_s < 0] = 0

    # Calculate density of vdf_diff
    delta_ang = (11.25 * np.pi / 180) ** 2
    theta_k = vdf.theta.data

    epsilon = np.zeros(len(vdf.time))
    m_psd2_n = np.ones(n_ph) * np.sin(theta_k * np.pi / 180)

    for i, j in itertools.product(range(len(vdf.time)), int_energies):
        tmp = np.squeeze(vdf_diff[i, j, ...])
        fct = v_s[i, j] ** 2 * delta_v[i, j] * delta_ang

        epsilon[i] += np.nansum(np.nansum(tmp * m_psd2_n, axis=0),
                                axis=0) * fct

    epsilon /= 1e6 * n_s.data * 2
    epsilon = ts_scalar(vdf.time.data, epsilon)

    return epsilon
