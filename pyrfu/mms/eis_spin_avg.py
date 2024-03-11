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


def _check_spin(spin):
    _, spin_inds = np.unique(spin, return_index=True)

    len_spin = np.max(np.diff(spin_inds))

    # Check to see if we have a complete first spin,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[0]])[0].size != len_spin:
        spin_inds = spin_inds[1:]

    # Check to see if the last one is complete,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[-1]])[0].size != len_spin:
        spin_inds = spin_inds[:-1]

    return spin_inds, len_spin


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
        Spin-averaged fluxes for all 6 telescopes.

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

    assert method.lower() in ["mean", "sum"]

    spin_nums = eis_allt.spin

    spin_starts, len_spin = _check_spin(spin_nums.data)

    scopes = list(filter(lambda x: x[0] == "t", eis_allt))

    spin_avg_flux = {}

    for scope in scopes:
        scope_data = eis_allt[scope]

        time_recs = scope_data.time.data[spin_starts + len_spin // 2]
        energies_ = scope_data.energy.data
        flux_data = scope_data.data

        flux_avg = np.zeros([len(spin_starts), len(energies_)])

        for i, spin_strt in enumerate(spin_starts):
            t_inds = np.where(spin_nums.data == spin_nums.data[spin_strt])[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                if method.lower() == "sum":
                    # Sum (for counts)
                    flux_avg[i, :] = np.nansum(flux_data[t_inds, :], axis=0)

                else:
                    # Average (for differential partile flux, differential
                    # energy flux, phase-space density, etc.)
                    flux_avg[i, :] = np.nanmean(flux_data[t_inds, :], axis=0)

        spin_avg_flux[scope] = xr.DataArray(
            flux_avg,
            coords=[time_recs, energies_],
            dims=["time", "energy"],
            attrs=scope_data.attrs,
        )

    # Get spin index
    spin_avg_flux["spin"] = eis_allt["spin"][spin_starts + len_spin // 2]

    # Construct Dataset
    out = xr.Dataset(spin_avg_flux, attrs=eis_allt.attrs)
    out.time.attrs = eis_allt.time.attrs
    out.energy.attrs = eis_allt.energy.attrs

    return out
