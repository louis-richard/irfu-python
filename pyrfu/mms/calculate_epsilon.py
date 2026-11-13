#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
import numpy as np
from xarray import DataArray, Dataset

# Local imports
from pyrfu.mms._internal.psd_units import _is_relativistic, _relativistic_factor
from pyrfu.mms._internal.vdf_bins import _charge, _mass, _speed
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar


def _get_si_vdf(vdf: Dataset) -> np.ndarray:
    r"""Relativistic phase-space density of vdf in SI units (s^3 m^-6): that
    of the FPI, HPCA and EIS files (non-relativistic) times 2 gamma^5 /
    (gamma + 1), so that the data and the model are in the same convention.

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

    if not _is_relativistic(vdf):
        out = out * _relativistic_factor(vdf, out)

    return out


def _same_bins(values_1, values_2) -> bool:
    r"""True if the two arrays of bins are equal, a table of each time, e.g.
    phi as (time, phi), being equal to the same table for all times
    (phi,)."""
    try:
        values_1, values_2 = np.broadcast_arrays(values_1, values_2)
    except ValueError:
        return False

    return bool(np.array_equal(values_1, values_2))


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
    en_channels: list[int] | None = None,
) -> DataArray:
    r"""Calculate epsilon parameter using model distribution.

    Parameters
    ----------
    vdf : Dataset
        Observed particle distribution (skymap), in s^3/cm^6, s^3/m^6 or s^3/km^6,
        with phi as (time, phi) or (phi,).
    model_vdf : Dataset
        Model particle distribution (skymap), in s^3/cm^6, s^3/m^6 or s^3/km^6,
        on the grid of `vdf` corrected for the spacecraft potential: its
        energies are those of `vdf` at infinity, E + q V (at least 0) with
        `sc_pot`, at its times, phi and theta, as made by `make_model_vdf` or
        `make_model_kappa` from `vdf` with the same `sc_pot`. The energies of
        `vdf` are corrected here, then the two are compared bin by bin.
    n_s : DataArray
        Time series of the number density in cm^-3 (same times as vdf). It,
        and the moments the model is made from, should be those of the
        distribution corrected for the same spacecraft potential, e.g. from
        ``psd_moments(vdf, sc_pot)``: with other moments (e.g. those of the
        L2 files), the data and the model need not integrate to the same
        density, and epsilon can exceed 1.
    sc_pot : DataArray
        Time series of the spacecraft potential in V. The ions gain q V, with
        q the ``charge`` attribute (1 by default), and the electrons lose
        e V, as `psd_moments`.
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
        If VDF and n_s have different times, or if the model is not on the
        grid of the VDF corrected for the spacecraft potential: most often a
        model on the measured energies, or made with another spacecraft
        potential.
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

    # The moments, and sc_pot resampled to them, at the times of vdf
    if not np.array_equal(vdf.time.data, n_s.time.data):
        raise ValueError("vdf and moments have different times.")

    # Energies of the data at infinity, E + q V (at least 0): the grid of the
    # model, as make_model_vdf and make_model_kappa compute it
    energy_corrected = vdf.energy.data.astype(np.float64)
    energy_corrected = energy_corrected + _charge(vdf) * sc_pot.data[:, None]
    energy_corrected = np.clip(energy_corrected, 0.0, None)

    # The model is compared bin by bin: on exactly the corrected grid of the
    # data (a model on other bins was paired by position, without error)
    for name in ["time", "phi", "theta"]:
        if not _same_bins(vdf[name].data, model_vdf[name].data):
            raise ValueError(f"vdf and model_vdf have different {name}")

    if not _same_bins(energy_corrected, model_vdf.energy.data):
        raise ValueError(
            "model_vdf is not on the energies of vdf corrected for the "
            "spacecraft potential, E + q V (at least 0) with this sc_pot: was "
            "it made on the measured energies, or with another potential? "
            "(make_model_vdf and make_model_kappa of vdf with the same sc_pot "
            "give that grid)"
        )

    # Get vdf and model_vdf in SI units (s^3 m^-6)
    vdf_data = _get_si_vdf(vdf)
    model_vdf_data = _get_si_vdf(model_vdf)

    energy = vdf.energy.data.astype(np.float64)
    phi = vdf.phi.data
    theta = vdf.theta.data

    # NaNs (e.g., fill values) don't contribute to the integral
    vdf_diff = np.nan_to_num(np.abs(vdf_data - model_vdf_data), nan=0.0)

    # Mass of the species of the table of the vdf routines (the species was
    # taken from its first letter: "protons" and "hydrogen+" raised)
    m_s = _mass(vdf)
    # Potential energy q V (eV) at the spacecraft: the ions of charge q gain
    # q V, the electrons lose e V (it was V whatever the charge)
    v_sc = -_charge(vdf) * sc_pot.data.astype(np.float64)

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

    def _speed_corrected(energy_):
        # Relativistic speed after correction for the spacecraft potential;
        # zero below it (as MATLAB's real(sqrt(...)); numpy's sqrt would give
        # NaN and drop the whole channel from the integral). It was
        # non-relativistic.
        energy_corr = np.clip(energy_ - v_sc[:, None], 0.0, None)
        return _speed(energy_corr, m_s)

    velocity = _speed_corrected(energy)
    delta_v = _speed_corrected(energy_upper) - _speed_corrected(energy_lower)

    # Weights of the integral over velocity space: v^2 dv (time, energy) and
    # sin(theta) dphi dtheta (theta), broadcast instead of tiled; phi as
    # (time, phi) or (phi,)
    delta_ang = np.deg2rad(np.median(np.diff(np.atleast_2d(phi)[0, :])))
    delta_ang *= np.deg2rad(np.median(np.diff(theta)))
    w_v = (velocity**2 * delta_v)[:, int_energies]
    w_ang = np.sin(np.deg2rad(theta)) * delta_ang

    epsilon = np.einsum("tepk,te,k->t", vdf_diff[:, int_energies, ...], w_v, w_ang)

    epsilon /= 1e6 * (n_s.data * 2)

    # Dimensionless (no UNITS)
    return ts_scalar(vdf.time.data, epsilon, {"UNITS": ""})
