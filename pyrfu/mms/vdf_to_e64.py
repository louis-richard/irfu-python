#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.ts_skymap import ts_skymap
from .psd_rebin import psd_rebin

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _seconds(values):
    r"""Durations in seconds as timedelta64[ns]."""
    return (np.asarray(values, dtype=np.float64) * 1e9).astype("timedelta64[ns]")


def vdf_to_e64(vdf_e32):
    r"""Recompile data into 64 energy channels. Time resolution is halved.
    Only applies to skymap.

    Parameters
    ----------
    vdf_e32 : xarray.Dataset
        Time series of the particle distribution with 32 energy channels.

    Returns
    -------
    vdf_e64 : xarray.Dataset
        Time series of the particle distribution with 64 energy channels, or
        a copy of `vdf_e32` if its two energy tables are the same.

    Notes
    -----
    As irfu-matlab PDist.e64, a distribution with a single energy table
    (``energy0`` equal to ``energy1``) is returned unchanged.

    """
    energy0 = vdf_e32.attrs.get("energy0")
    energy1 = vdf_e32.attrs.get("energy1")

    if np.array_equal(np.asarray(energy0), np.asarray(energy1)):
        logger.info("One energy table: the distribution is returned unchanged")
        return vdf_e32.copy(deep=True)

    time_r, vdf_r, energy_r, phi_r = psd_rebin(
        vdf_e32,
        vdf_e32.phi.data,
        vdf_e32.attrs.get("energy0"),
        vdf_e32.attrs.get("energy1"),
        vdf_e32.attrs.get("esteptable"),
    )

    energy_r = np.tile(energy_r, (len(vdf_r), 1))

    # Data attributes
    data_attrs = dict(vdf_e32.data.attrs)

    # Coordinates attributes
    coords_attrs = {
        k: dict(vdf_e32[k].attrs) for k in ["time", "energy", "phi", "theta"]
    }

    # Global attributes (copy: the 64-channel delta energies must not overwrite the
    # caller's)
    glob_attrs = dict(vdf_e32.attrs)

    # Half widths of the accumulations: from the start of the first sample of
    # each pair to the end of the second (twice those of the samples for
    # regular sampling, as irfu-matlab PDist.e64)
    if "delta_time_minus" in glob_attrs and "delta_time_plus" in glob_attrs:
        time = vdf_e32.time.data
        idx = np.arange(0, len(time) - 1, 2)
        t_start = time[idx] - _seconds(vdf_e32.attrs["delta_time_minus"][idx])
        t_stop = time[idx + 1] + _seconds(vdf_e32.attrs["delta_time_plus"][idx + 1])
        glob_attrs["delta_time_minus"] = (time_r - t_start) / np.timedelta64(1, "s")
        glob_attrs["delta_time_plus"] = (t_stop - time_r) / np.timedelta64(1, "s")

    # update delta_energy
    if "delta_energy_plus" in glob_attrs and "delta_energy_minus" in glob_attrs:
        log_energy = np.log10(energy_r[0, :])
        log10_energy = np.diff(log_energy)
        log10_energy_plus = log_energy + 0.5 * np.hstack(
            [log10_energy, log10_energy[-1]],
        )
        log10_energy_minus = log_energy - 0.5 * np.hstack(
            [log10_energy[0], log10_energy],
        )

        energy_plus = 10**log10_energy_plus
        energy_minus = 10**log10_energy_minus
        delta_energy_plus = energy_plus - energy_r
        delta_energy_minus = abs(energy_minus - energy_r)
        # Outer edges (top and bottom channels) from the 32-channel widths. The
        # arrays are (time, energy): [-1] and [0] overwrote time steps.
        delta_energy_plus[:, -1] = np.max(
            np.asarray(vdf_e32.attrs["delta_energy_plus"])[:, -1]
        )
        delta_energy_minus[:, 0] = np.min(
            np.asarray(vdf_e32.attrs["delta_energy_minus"])[:, 0]
        )

        glob_attrs["delta_energy_plus"] = delta_energy_plus
        glob_attrs["delta_energy_minus"] = delta_energy_minus

    vdf_e64 = ts_skymap(
        time_r,
        vdf_r,
        energy_r,
        phi_r,
        vdf_e32.theta.data,
        energy0=energy_r[0, :],
        energy1=energy_r[0, :],
        esteptable=np.zeros(len(time_r)),
        attrs=data_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    return vdf_e64
