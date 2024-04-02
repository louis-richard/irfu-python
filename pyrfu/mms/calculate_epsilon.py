#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
from scipy import constants
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

q_e = constants.elementary_charge


def calculate_epsilon(
    vdf: Dataset,
    model_vdf: Dataset,
    n_s: DataArray,
    sc_pot: DataArray,
    en_channels: Optional[list[int]] = None,
) -> DataArray:
    r"""Calculate epsilon parameter using model distribution.

    Parameters
    ----------
    vdf : Dataset
        Observed particle distribution (skymap).
    model_vdf : Dataset
        Model particle distribution (skymap).
    n_s : DataArray
        Time series of the number density.
    sc_pot : DataArray
        Time series of the spacecraft potential.
    en_channels : list, Optional
        Set energy channels to integrate over [min max]; min and max between
        must be between 1 and 32.

    Returns
    -------
    DataArray
        Time series of the epsilon parameter.

    Raises
    ------
    ValueError
        If VDF and n_s have different times.
    TypeError
        If en_channels is not a list.


    Examples
    --------
    >>> from pyrfu import mms
    >>> options = {"en_channel": [4, 32]}
    >>> eps = mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot, **options)

    """
    # Resample sc_pot
    sc_pot = resample(sc_pot, n_s)

    vdf_data = vdf.data.data.copy() * 1e12
    model_vdf_data = model_vdf.data.data.copy() * 1e-18

    energy = vdf.energy.data.copy()
    phi = vdf.phi.data.copy()
    theta = vdf.theta.data.copy()

    vdf_diff = np.abs(vdf_data - model_vdf_data)

    if vdf.attrs["species"][0].lower() == "e":
        m_s = constants.electron_mass
    elif vdf.attrs["species"][0].lower() == "i":
        sc_pot.data *= -1
        m_s = constants.proton_mass
    else:
        raise ValueError("Invalid specie")

    if np.abs(np.median(np.diff(vdf.time.data - n_s.time.data))) > 0:
        raise ValueError("vdf and moments have different times.")

    # Default energy channels used to compute epsilon.
    if en_channels is None:
        energy_range = [0, vdf.energy.shape[1]]
    elif isinstance(en_channels, list):
        energy_range = en_channels
    else:
        raise TypeError("en_channels must be a list.")

    int_energies = np.arange(energy_range[0], energy_range[1])

    flag_same_e = np.sum(np.abs(vdf.attrs["energy0"] - vdf.attrs["energy1"])) < 1e-4

    # Calculate angle differences
    delta_phi = np.deg2rad(np.median(np.diff(phi[0, :])))
    delta_theta = np.deg2rad(np.median(np.diff(theta)))

    delta_ang = delta_phi * delta_theta

    phi_tr = phi.copy()
    theta_tr = np.tile(theta, (len(vdf.time.data), 1))

    energy_minus = vdf.attrs["delta_energy_minus"]
    energy_plus = vdf.attrs["delta_energy_plus"]

    # Calculate speed widths associated with each energy channel.
    energy_scpot = np.transpose(np.tile(sc_pot.data, (energy.shape[1], 1)))
    energy_corr = energy - np.transpose(
        np.tile(sc_pot.data, (energy.shape[1], 1)),
    )
    velocity = np.real(np.sqrt(2 * q_e * energy_corr / m_s))

    if flag_same_e:
        energy_upper = energy + energy_plus
        energy_lower = energy - energy_minus
        v_upper = np.sqrt(2 * q_e * (energy_upper - energy_scpot) / m_s)
        v_lower = np.sqrt(2 * q_e * (energy_lower - energy_scpot) / m_s)

    else:
        energy_upper = energy + energy_plus
        energy_lower = energy - energy_minus
        v_upper = np.sqrt(2 * q_e * (energy_upper - energy_scpot) / m_s)
        v_lower = np.sqrt(2 * q_e * (energy_lower - energy_scpot) / m_s)

    v_upper[v_upper < 0] = 0
    v_lower[v_lower < 0] = 0
    v_upper = np.real(v_upper)
    v_lower = np.real(v_lower)

    delta_v = v_upper - v_lower
    v_mat = np.tile(velocity, (phi_tr.shape[1], theta_tr.shape[1], 1, 1))
    v_mat = np.transpose(v_mat, [2, 3, 0, 1])

    delta_v_mat = np.tile(delta_v, (phi_tr.shape[1], theta_tr.shape[1], 1, 1))
    delta_v_mat = np.transpose(delta_v_mat, [2, 3, 0, 1])

    v_mat = v_mat[:, int_energies, ...]
    delta_v_mat = delta_v_mat[:, int_energies, ...]

    theta_mat = np.tile(theta_tr, (len(int_energies), phi_tr.shape[1], 1, 1))
    theta_mat = np.transpose(theta_mat, [2, 0, 1, 3])

    m_mat = np.sin(np.deg2rad(theta_mat)) * delta_ang

    epsilon = np.nansum(
        np.nansum(
            np.nansum(
                m_mat * vdf_diff[:, int_energies, ...] * v_mat**2 * delta_v_mat,
                axis=-1,
            ),
            axis=-1,
        ),
        axis=-1,
    )

    epsilon /= 1e6 * (n_s.data * 2)

    return ts_scalar(vdf.time.data, epsilon)
