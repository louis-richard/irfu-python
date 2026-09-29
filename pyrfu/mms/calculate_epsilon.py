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
        Observed particle distribution (skymap), in s^3/cm^6, s^3/m^6 or s^3/km^6.
    model_vdf : Dataset
        Model particle distribution (skymap), in s^3/cm^6, s^3/m^6 or s^3/km^6.
    n_s : DataArray
        Time series of the number density in cm^-3 (same times as vdf).
    sc_pot : DataArray
        Time series of the spacecraft potential in V.
    en_channels : list, Optional
        Energy channels to integrate over, as 0-based indices [start, stop)
        (stop excluded), e.g., [3, 32] for all but the three lowest of 32
        channels. Default is all channels.

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
    >>> options = {"en_channels": [3, 32]}
    >>> eps = mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot, **options)

    """
    # Resample sc_pot
    sc_pot = resample(sc_pot, n_s)

    # Get vdf and model_vdf in SI units (s^3 m^-6)
    vdf_data = _get_si_vdf(vdf)
    model_vdf_data = _get_si_vdf(model_vdf)

    energy = vdf.energy.data.astype(np.float64)
    phi = vdf.phi.data
    theta = vdf.theta.data

    # NaNs (e.g., fill values) don't contribute to the integral
    vdf_diff = np.nan_to_num(np.abs(vdf_data - model_vdf_data), nan=0.0)

    if vdf.attrs["species"][0].lower() == "e":
        m_s = constants.electron_mass
        v_sc = sc_pot.data.astype(np.float64)
    elif vdf.attrs["species"][0].lower() == "i":
        m_s = constants.proton_mass
        v_sc = -sc_pot.data.astype(np.float64)
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

    # Energy widths may be missing, or set to None (e.g., by get_dist)
    energy_minus = vdf.attrs.get("delta_energy_minus")
    energy_plus = vdf.attrs.get("delta_energy_plus")
    flag_delta_e = energy_minus is not None and energy_plus is not None

    # Upper and lower energy edges of the channels
    if flag_delta_e:
        energy_upper = energy + energy_plus
        energy_lower = energy - energy_minus
    elif flag_same_e:
        # extrapolate one bin before the first and after the last energy column
        temp0 = 2 * energy[:, 0] - energy[:, 1]
        tempend = 2 * energy[:, -1] - energy[:, -2]
        diff_en_all = np.diff(np.column_stack([temp0, energy, tempend]), axis=1)
        energy_upper = 10 ** (np.log10(energy + diff_en_all[:, 1:] / 2))
        energy_lower = 10 ** (np.log10(energy - diff_en_all[:, :-1] / 2))
    else:
        energy_upper0, energy_lower0 = _energy_bin_edges(np.ravel(vdf.attrs["energy0"]))
        energy_upper1, energy_lower1 = _energy_bin_edges(np.ravel(vdf.attrs["energy1"]))

        # esteptable flags which table (0 or 1) applies at each time step
        step_table = np.ravel(vdf.attrs["esteptable"])[:, None] == 1
        energy_upper = np.where(step_table, energy_upper1, energy_upper0)
        energy_lower = np.where(step_table, energy_lower1, energy_lower0)

    def _speed(energy_):
        # Speed after correction for the spacecraft potential; zero below it
        # (as MATLAB's real(sqrt(...)); numpy's sqrt would give NaN and drop the
        # whole channel from the integral)
        energy_corr = np.clip(energy_ - v_sc[:, None], 0.0, None)
        return np.sqrt(2 * q_e * energy_corr / m_s)

    velocity = _speed(energy)
    delta_v = _speed(energy_upper) - _speed(energy_lower)

    # Weights of the integral over velocity space: v^2 dv (time, energy) and
    # sin(theta) dphi dtheta (theta), broadcast instead of tiled
    delta_ang = np.deg2rad(np.median(np.diff(phi[0, :])))
    delta_ang *= np.deg2rad(np.median(np.diff(theta)))
    w_v = (velocity**2 * delta_v)[:, int_energies]
    w_ang = np.sin(np.deg2rad(theta)) * delta_ang

    epsilon = np.einsum("tepk,te,k->t", vdf_diff[:, int_energies, ...], w_v, w_ang)

    epsilon /= 1e6 * (n_s.data * 2)

    return ts_scalar(vdf.time.data, epsilon)
