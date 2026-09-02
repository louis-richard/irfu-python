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


def _get_si_vdf(vdf: Dataset) -> np.ndarray:
    r"""Convert vdf to SI units (s^3 m^-6).

    Parameters
    ----------
    vdf : Dataset
        Particle distribution (skymap).

    Returns
    -------
    np.ndarray
        Particle distribution in SI units (s^3 m^-6).
    """

    if vdf.data.attrs["UNITS"] == "s^3/km^6":
        out = vdf.data.data.copy() * 1e-18
    elif vdf.data.attrs["UNITS"] == "s^3/m^6":
        out = vdf.data.data.copy()
    elif vdf.data.attrs["UNITS"] == "s^3/cm^6":
        out = vdf.data.data.copy() * 1e12
    else:
        raise ValueError("Invalid units for vdf.")

    return out


def _energy_bin_edges(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Upper/lower energy-bin edges for a single 1-D energy table."""
    temp0 = 2 * energy[0] - energy[1]
    tempend = 2 * energy[-1] - energy[-2]
    energy_all = np.concatenate(([temp0], energy, [tempend]))
    diff_en_all = np.diff(energy_all)
    energy_upper = 10 ** (np.log10(energy + diff_en_all[1:] / 2))
    energy_lower = 10 ** (np.log10(energy - diff_en_all[:-1] / 2))
    return energy_upper, energy_lower


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
        Observed particle distribution (skymap). Must be in s^3 cm^-6.
    model_vdf : Dataset
        Model particle distribution (skymap). Must be in s^3 km^-6.
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

    # Get vdf and model_vdf in SI units (s^3 m^-6)
    vdf_data = _get_si_vdf(vdf)
    model_vdf_data = _get_si_vdf(model_vdf)

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

    if not np.array_equal(vdf.time.data, n_s.time.data):
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

    if "delta_energy_minus" in vdf.attrs and "delta_energy_plus" in vdf.attrs:
        flag_delta_e = True
        energy_minus = vdf.attrs["delta_energy_minus"]
        energy_plus = vdf.attrs["delta_energy_plus"]
    else:
        energy_minus = np.zeros_like(np.unique(energy, axis=0))
        energy_plus = np.zeros_like(np.unique(energy, axis=0))
        flag_delta_e = False

    # Calculate speed widths associated with each energy channel.
    energy_scpot = np.transpose(np.tile(sc_pot.data, (energy.shape[1], 1)))
    energy_corr = energy - np.transpose(
        np.tile(sc_pot.data, (energy.shape[1], 1)),
    )
    velocity = np.real(np.sqrt(2 * q_e * energy_corr / m_s))

    if flag_delta_e:
        energy_upper = energy + energy_plus
        energy_lower = energy - energy_minus
        v_upper = np.sqrt(2 * q_e * (energy_upper - energy_scpot) / m_s)
        v_lower = np.sqrt(2 * q_e * (energy_lower - energy_scpot) / m_s)
    elif flag_same_e and not flag_delta_e:
        # extrapolate one bin before the first and after the last energy column
        temp0 = 2 * energy[:, 0] - energy[:, 1]
        tempend = 2 * energy[:, -1] - energy[:, -2]

        # [temp0 energy tempend] horzcat -> column_stack
        energyall = np.column_stack([temp0, energy, tempend])

        # diff(energyall, 1, 2) -> np.diff along columns (axis=1)
        diffenall = np.diff(energyall, n=1, axis=1)

        # diffenall(:,2:end) -> [:, 1:] ; diffenall(:,1:end-1) -> [:, :-1]
        energyupper = 10 ** (np.log10(energy + diffenall[:, 1:] / 2))
        energylower = 10 ** (np.log10(energy - diffenall[:, :-1] / 2))

        # SCpot.data*ones(size(energy(1,:))) is just broadcasting SCpot per row
        # across all energy columns — numpy does this for free with [:, None]
        v_upper = np.sqrt(2 * q_e * (energyupper - sc_pot[:, None]) / m_s)
        v_lower = np.sqrt(2 * q_e * (energylower - sc_pot[:, None]) / m_s)
    elif not flag_same_e and not flag_delta_e:
        energy0 = np.ravel(vdf.attrs["energy0"])
        energy1 = np.ravel(vdf.attrs["energy1"])
        esteptable = np.ravel(vdf.attrs["esteptable"])

        energyupper0, energylower0 = _energy_bin_edges(energy0)
        energyupper1, energylower1 = _energy_bin_edges(energy1)

        # esteptable flags which table (0 or 1) applies at each time step;
        # broadcast it across energy channels to select per row.
        esteptablemat = esteptable[:, None].astype(float) * np.ones_like(energy0)

        energyupper = (
            esteptablemat * energyupper1 + np.abs(esteptablemat - 1) * energyupper0
        )
        energylower = (
            esteptablemat * energylower1 + np.abs(esteptablemat - 1) * energylower0
        )

        v_upper = np.sqrt(2 * q_e * (energyupper - energy_scpot) / m_s)
        v_lower = np.sqrt(2 * q_e * (energylower - energy_scpot) / m_s)
    else:
        raise NotImplementedError(
            "Unsupported combination: energy0 != energy1 with no "
            "delta_energy_minus/delta_energy_plus available."
        )

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
