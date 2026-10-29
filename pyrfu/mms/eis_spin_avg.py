#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _spin_segments(spin):
    r"""Start and stop indices of the complete spins: the runs of samples with
    the same spin number, in time order, with the full number of samples (that
    of the longest spin between two spin boundaries)."""
    # Runs in time order (np.unique ordered the spins by number, which the
    # wrap of the uint16 spin counter reorders)
    spin = np.asarray(spin).astype(np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(spin) != 0) + 1, len(spin)]

    if len(bounds) < 4:
        raise ValueError(
            "no complete spin: the interval must hold a spin between two spin "
            "boundaries"
        )

    lengths = np.diff(bounds)

    # The partial first and last spins and the spins cut by data gaps are left
    # out
    complete = lengths == np.max(lengths[1:-1])

    return bounds[:-1][complete], bounds[1:][complete]


def eis_spin_avg(eis_allt, method: str = "mean"):
    r"""Calculates spin-averaged fluxes for the EIS instrument.

    Parameters
    ----------
    eis_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes.
    method : {"mean", "sum"}
        Method.
    Returns
    -------
    spin_avg_fluxes : xarray.Dataset
        Spin-averaged fluxes for all 6 telescopes, one per complete spin at the
        time of its first sample.

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", or if the interval has no complete
        spin.

    Notes
    -----
    A spin is a run of samples with the same spin number, in time order (the
    spin counter wraps about every 15 days). Only the complete spins are
    averaged: the partial first and last spins and the spins cut by data gaps
    are left out. Each spin is time stamped at its first sample, as
    :func:`pyrfu.mms.hpca_spin_sum`, the FEEPS spin averages and IDL SPEDAS
    ``mms_eis_pad_spinavg``; IDL SPEDAS and pySPEDAS stamp the spin-averaged
    spectra at the last sample. The time stamp was the 5th of the 8 survey
    samples.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_omni

    Examples
    --------
    >>> from pyrfu import mms

    Define spacecraft index and time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]
    >>> ic = 2

    Get EIS ExTOF all 6 telescopes fluxes

    >>> extof_allt = mms.get_eis_allt("flux_extof_proton_srvy_l2", tint, ic)

    Spin average all 6 telescopes fluxes

    >>> extof_allt_despin = mms.eis_spin_avg(extof_allt)

    """

    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"method must be mean or sum, not {method}")

    spin_nums = eis_allt.spin

    spin_starts, spin_stops = _spin_segments(spin_nums.data)

    scopes = list(filter(lambda x: x[0] == "t", eis_allt))

    spin_avg_flux = {}

    for scope in scopes:
        scope_data = eis_allt[scope]

        time_recs = scope_data.time.data[spin_starts]
        energies_ = scope_data.energy.data
        flux_data = scope_data.data

        flux_avg = np.zeros([len(spin_starts), len(energies_)])

        for i, (spin_strt, spin_stop) in enumerate(zip(spin_starts, spin_stops)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                if method.lower() == "sum":
                    # Sum (for counts)
                    flux_avg[i, :] = np.nansum(flux_data[spin_strt:spin_stop], axis=0)

                else:
                    # Average (for differential partile flux, differential
                    # energy flux, phase-space density, etc.)
                    flux_avg[i, :] = np.nanmean(flux_data[spin_strt:spin_stop], axis=0)

        spin_avg_flux[scope] = xr.DataArray(
            flux_avg,
            coords=[time_recs, energies_],
            dims=["time", "energy"],
            attrs=scope_data.attrs,
        )

    # Get spin index
    spin_avg_flux["spin"] = eis_allt["spin"][spin_starts]

    # Construct Dataset
    out = xr.Dataset(spin_avg_flux, attrs=eis_allt.attrs)
    out.time.attrs = eis_allt.time.attrs
    out.energy.attrs = eis_allt.energy.attrs

    return out
