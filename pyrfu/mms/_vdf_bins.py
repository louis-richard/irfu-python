#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.resample import resample
from ._psd_units import _mass_ratio

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Bins of skymaps and pitch-angle distributions, shared by the vdf_* and pad_*
# functions (irfu-matlab PDist methods)


def _mass(vdf):
    r"""Mass (kg) of the species of the distribution."""
    return _mass_ratio(vdf) * constants.proton_mass


def _charge_sign(vdf):
    r"""Sign of the charge of the species of the distribution."""
    is_electron = str(vdf.attrs["species"]).lower() in ["electrons", "electron", "e"]
    return -1.0 if is_electron else 1.0


def _relativistic(vdf, relativistic):
    r"""Relativistic speeds: the argument, else the attribute, else False."""
    if relativistic is None:
        relativistic = vdf.attrs.get("relativistic", False)

    return bool(relativistic)


def _sc_pot(vdf, sc_pot):
    r"""Spacecraft potential (V) at the times of the distribution (time,)."""
    n_t = len(vdf.time)

    if sc_pot is None:
        return np.zeros(n_t)

    if isinstance(sc_pot, xr.DataArray):
        return resample(sc_pot, vdf.time).data.astype(np.float64)

    sc_pot = np.asarray(sc_pot, dtype=np.float64)

    if sc_pot.ndim == 0:
        return np.full(n_t, float(sc_pot))

    if sc_pot.shape != (n_t,):
        raise ValueError("sc_pot must be a scalar, a time series or (time,) values")

    return sc_pot


def _energy_edges(vdf):
    r"""Lower and upper energy edges (eV) of the channels (time, energy): the
    delta_energy_minus/plus attributes when they match the energies, else the
    midpoints between the channels (extrapolated at both ends)."""
    energy = np.atleast_2d(np.asarray(vdf.energy.data, dtype=np.float64))
    energy = np.broadcast_to(energy, (len(vdf.time), energy.shape[-1]))
    d_minus = vdf.attrs.get("delta_energy_minus")
    d_plus = vdf.attrs.get("delta_energy_plus")

    if d_minus is not None and d_plus is not None:
        d_minus, d_plus = np.asarray(d_minus), np.asarray(d_plus)
        same_table = np.all(energy == energy[:1])
        fits = all(
            d.shape == energy.shape or (d.shape == energy.shape[1:] and same_table)
            for d in [d_minus, d_plus]
        )

        if fits:
            return energy - d_minus, energy + d_plus

    padded = np.concatenate(
        [
            2 * energy[:, :1] - energy[:, 1:2],
            energy,
            2 * energy[:, -1:] - energy[:, -2:-1],
        ],
        axis=1,
    )
    half = np.diff(padded, axis=1) / 2
    return energy - half[:, :-1], energy + half[:, 1:]


def _kinetic_energy(vdf, energy, sc_pot):
    r"""Energy (eV) at infinity, corrected for the spacecraft potential (V),
    as irfu-matlab PDist.v (ions gain, electrons lose q V), at least 0."""
    sc_pot = sc_pot.reshape((-1,) + (1,) * (np.ndim(energy) - 1))
    return np.clip(energy + _charge_sign(vdf) * sc_pot, 0.0, None)


def _speed(energy, mass, relativistic):
    r"""Speed (m/s) of particles of kinetic energy (eV)."""
    if relativistic:
        # beta^2 = 1 - 1 / gamma^2 without cancellation, gamma = 1 + eps
        eps = constants.elementary_charge * energy / (mass * constants.c**2)
        return constants.c * np.sqrt(eps * (2.0 + eps)) / (1.0 + eps)

    return np.sqrt(2.0 * constants.elementary_charge * energy / mass)


def _phi_edges(vdf):
    r"""Lower and upper azimuth edges (degrees) of the bins (time, phi): the
    delta_phi_minus/plus attributes, else half the azimuth spacing."""
    phi = np.asarray(vdf.phi.data, dtype=np.float64)
    phi = np.broadcast_to(phi, (len(vdf.time), phi.shape[-1]))

    if "delta_phi_minus" in vdf.attrs and "delta_phi_plus" in vdf.attrs:
        d_minus = np.asarray(vdf.attrs["delta_phi_minus"], dtype=np.float64)
        d_plus = np.asarray(vdf.attrs["delta_phi_plus"], dtype=np.float64)
    else:
        d_minus = d_plus = np.abs(np.median(np.diff(phi[0]))) / 2

    return phi - d_minus, phi + d_plus


def _polar_edges(angles, attrs, name):
    r"""Lower and upper edges (degrees) of polar bins (theta or pitch angle):
    the delta_{name}_minus/plus attributes, else half the spacing."""
    angles = np.asarray(angles, dtype=np.float64)

    if f"delta_{name}_minus" in attrs and f"delta_{name}_plus" in attrs:
        d_minus = np.asarray(attrs[f"delta_{name}_minus"], dtype=np.float64)
        d_plus = np.asarray(attrs[f"delta_{name}_plus"], dtype=np.float64)
    else:
        d_minus = d_plus = np.abs(np.median(np.diff(angles, axis=-1))) / 2

    return angles - d_minus, angles + d_plus


def _theta_edges(vdf):
    r"""Lower and upper polar edges (degrees) of the bins of a skymap
    (theta,)."""
    return _polar_edges(vdf.theta.data, vdf.attrs, "theta")


def _pitch_angles(pad):
    r"""Pitch angles (degrees) of a pitch-angle distribution (pitch angle,)."""
    pitch = np.asarray(pad.theta.data, dtype=np.float64)
    return pitch[0] if pitch.ndim == 2 else pitch


def _is_skymap(vdf):
    r"""Skymap (time, energy, phi, theta), else pitch-angle distribution
    (time, energy, pitch angle)."""
    if vdf.data.ndim == 4:
        return True

    if vdf.data.ndim == 3 and "theta" in vdf:
        return False

    raise ValueError("vdf must be a skymap or a pitch-angle distribution")


def _band_solid_angles(vdf):
    r"""Solid angle (sr) of each angular bin: (time, phi, theta) for a skymap,
    (pitch angle,) for a pitch-angle distribution (bands of 2 pi)."""
    if _is_skymap(vdf):
        phi_lower, phi_upper = _phi_edges(vdf)
        theta_lower, theta_upper = _theta_edges(vdf)
        d_phi = np.deg2rad(phi_upper - phi_lower)
        d_cos = np.cos(np.deg2rad(theta_lower)) - np.cos(np.deg2rad(theta_upper))
        return d_phi[:, :, np.newaxis] * d_cos[np.newaxis, np.newaxis, :]

    lower, upper = _polar_edges(_pitch_angles(vdf), vdf.attrs, "pitchangle")
    return 2 * np.pi * (np.cos(np.deg2rad(lower)) - np.cos(np.deg2rad(upper)))


def _speed_edges(vdf, sc_pot, relativistic):
    r"""Lower and upper speeds (m/s) of the energy channels (time, energy),
    corrected for the spacecraft potential."""
    sc_pot = _sc_pot(vdf, sc_pot)
    mass = _mass(vdf)
    relativistic = _relativistic(vdf, relativistic)
    return [
        _speed(_kinetic_energy(vdf, edge, sc_pot), mass, relativistic)
        for edge in _energy_edges(vdf)
    ]


def _broadcast_energy(values, data):
    r"""(time, energy) values shaped to broadcast against the data."""
    return values.reshape(values.shape + (1,) * (data.ndim - 2))


def _broadcast_angles(values, data):
    r"""Solid angles of `_band_solid_angles` shaped to broadcast against the
    data."""
    if data.ndim == 4:
        return values[:, np.newaxis, :, :]

    return values[np.newaxis, np.newaxis, :]


def _d3v(vdf, sc_pot, relativistic):
    r"""Velocity-space volume (m^3/s^3) of each bin, shaped as the data."""
    data = vdf.data.data
    v_lower, v_upper = _speed_edges(vdf, sc_pot, relativistic)
    d_v3 = _broadcast_energy((v_upper**3 - v_lower**3) / 3, data)
    d3v = d_v3 * _broadcast_angles(_band_solid_angles(vdf), data)
    return np.broadcast_to(d3v, data.shape)
