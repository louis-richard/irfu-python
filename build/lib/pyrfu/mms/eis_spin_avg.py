#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_spin_avg(eis_allt):
    r"""Calculates spin-averaged fluxes for the EIS instrument.

    Parameters
    ----------
    eis_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes.

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

    spin_nums = eis_allt.spin

    spin_starts = np.where(spin_nums.data[1:] > spin_nums.data[:-1])[0]

    scopes = list(filter(lambda x: x[0] == "t", eis_allt))

    spin_avg_flux = {}

    for scope in scopes:
        scope_data = eis_allt[scope]

        time_recs = scope_data.time.data[spin_starts]
        energies_ = scope_data.energy.data
        flux_data = scope_data.data

        flux_ds = np.zeros([len(spin_starts), len(energies_)])

        c_strt = 0

        for i, spin_strt in enumerate(spin_starts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                flux_ds[i, :] = np.nanmean(flux_data[c_strt:spin_strt + 1, :],
                                           axis=0)
            c_strt = spin_strt + 1

        spin_avg_flux[scope] = xr.DataArray(flux_ds,
                                            coords=[time_recs, energies_],
                                            dims=["time", "energy"])

    spin_avg_flux["spin"] = eis_allt["spin"][spin_starts]
    spin_avg_flux = xr.Dataset(spin_avg_flux, attrs=eis_allt.attrs)

    return spin_avg_flux
