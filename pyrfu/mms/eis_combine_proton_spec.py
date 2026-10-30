#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._eis_units import _check_flux

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _channels(energy_phxtof, energy_extof):
    r"""Indices of the PHxTOF channels kept, of the cross-over pairs (PHxTOF,
    ExTOF) averaged and of the ExTOF channels kept, as IDL SPEDAS 6.1
    mms_eis_combine_proton_spec (energy_extof without its lowest channel)."""
    idx_phxtof = np.flatnonzero(energy_phxtof < energy_extof[0])
    idx_phxtof_cross = np.flatnonzero(energy_phxtof > energy_extof[0])
    idx_extof_cross = np.flatnonzero(energy_extof < energy_phxtof[-1])
    idx_extof = np.flatnonzero(energy_extof > energy_phxtof[-1])

    if len(idx_phxtof_cross) != len(idx_extof_cross):
        raise ValueError(
            "The PHxTOF and ExTOF cross-over channels don't pair: "
            f"{energy_phxtof[idx_phxtof_cross]} and {energy_extof[idx_extof_cross]}"
        )

    return idx_phxtof, list(zip(idx_phxtof_cross, idx_extof_cross)), idx_extof


def _combine_attrs(attrs1, attrs2):
    attrs = {}
    for k in attrs1:
        if k.lower() == "global":
            attrs[k] = _combine_attrs(attrs1[k], attrs2[k])
        elif k not in ["delta_energy_plus", "delta_energy_minus"]:
            if attrs1[k] == attrs2[k]:
                attrs[k] = attrs1[k]
            else:
                attrs[k] = [attrs1[k], attrs2[k]]
        else:
            continue

    return attrs


def eis_combine_proton_spec(phxtof_allt, extof_allt):
    r"""Combine ExTOF and PHxTOF proton energy spectra into a single combined
    Dataset.

    Parameters
    ----------
    phxtof_allt : xarray.Dataset
        Dataset containing the PHxTOF energy spectrum of the 6 telescopes.
    extof_allt : xarray.Dataset
        Dataset containing the ExTOF energy spectrum of the 6 telescopes.

    Returns
    -------
    comb_allt : xarray.Dataset
        Dataset containing the combined PHxTOF and ExTOF energy spectrum of
        the 6 telescopes, at the times of both, with the energy widths of the
        combined channels (``delta_energy_minus``, ``delta_energy_plus``).

    Raises
    ------
    ValueError
        If the telescopes of the two datasets differ, if the cross-over
        channels don't pair, or if the data are not fluxes in the same units
        (the ``UNITS`` of the telescopes, when they have them).

    Notes
    -----
    The channels are those of IDL SPEDAS 6.1 ``mms_eis_combine_proton_spec``:
    the lowest ExTOF channel is left out (noise near the detector threshold),
    the PHxTOF channels below the next ExTOF one are kept, the PHxTOF channels
    above it are averaged with the ExTOF channels below the top PHxTOF one
    (pairwise, at the geometric mean of their combined edges), and the ExTOF
    channels above the top PHxTOF one are kept. pyrfu kept the lowest ExTOF
    channel and left out the top PHxTOF one. The samples are those at the
    times of both instruments (they could be paired across times). The energy
    widths are E minus the lower edge and the upper edge minus E (they were
    the edges). Unlike IDL, NaN are kept (IDL sets them to zero). As IDL, only
    fluxes are combined: counts and count rates are not divided by the
    geometric factors and the energy widths of the two instruments (counts and
    cps were combined, and the units became a list when they differed).

    """

    scopes_phxtof = list(filter(lambda x: x[0] == "t", phxtof_allt))
    scopes_extof = list(filter(lambda x: x[0] == "t", extof_allt))

    if scopes_extof != scopes_phxtof:
        raise ValueError(
            f"PHxTOF and ExTOF have different telescopes: {scopes_phxtof} and "
            f"{scopes_extof}"
        )

    _check_flux(
        *[phxtof_allt[s] for s in scopes_phxtof], *[extof_allt[s] for s in scopes_extof]
    )

    # Channels and energy widths, the lowest ExTOF channel left out
    energy_phxtof = phxtof_allt[scopes_phxtof[0]].energy.data
    energy_extof = extof_allt[scopes_extof[0]].energy.data[1:]
    dminus_phxtof = np.asarray(phxtof_allt.attrs["delta_energy_minus"])
    dplus_phxtof = np.asarray(phxtof_allt.attrs["delta_energy_plus"])
    dminus_extof = np.asarray(extof_allt.attrs["delta_energy_minus"])[1:]
    dplus_extof = np.asarray(extof_allt.attrs["delta_energy_plus"])[1:]

    idx_phxtof, cross, idx_extof = _channels(energy_phxtof, energy_extof)

    # Lower and upper edges of the combined channels
    low = np.hstack(
        [
            energy_phxtof[idx_phxtof] - dminus_phxtof[idx_phxtof],
            [
                min(
                    energy_phxtof[i] - dminus_phxtof[i],
                    energy_extof[j] - dminus_extof[j],
                )
                for i, j in cross
            ],
            energy_extof[idx_extof] - dminus_extof[idx_extof],
        ]
    )
    high = np.hstack(
        [
            energy_phxtof[idx_phxtof] + dplus_phxtof[idx_phxtof],
            [
                max(
                    energy_phxtof[i] + dplus_phxtof[i], energy_extof[j] + dplus_extof[j]
                )
                for i, j in cross
            ],
            energy_extof[idx_extof] + dplus_extof[idx_extof],
        ]
    )
    energy_combined = np.hstack(
        [
            energy_phxtof[idx_phxtof],
            np.sqrt(low * high)[len(idx_phxtof) : len(idx_phxtof) + len(cross)],
            energy_extof[idx_extof],
        ]
    )

    # Samples at the times of both instruments (they could be paired across
    # times)
    time_data, i_phx, i_ex = np.intersect1d(
        phxtof_allt.time.data, extof_allt.time.data, return_indices=True
    )

    # The other ExTOF variables (spin, sector, look directions, ...) at those
    # times
    out_keys = list(filter(lambda x: x not in scopes_extof, extof_allt))
    out_dict = {
        k: (
            extof_allt[k].isel(time=i_ex)
            if "time" in extof_allt[k].dims
            else extof_allt[k]
        )
        for k in out_keys
    }

    for scope in scopes_phxtof:
        phxtof_data = phxtof_allt[scope].data[i_phx]
        extof_data = extof_allt[scope].data[i_ex][:, 1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cross_data = [
                np.nanmean(np.stack([phxtof_data[:, i], extof_data[:, j]]), axis=0)
                for i, j in cross
            ]

        blocks = [phxtof_data[:, idx_phxtof], extof_data[:, idx_extof]]

        if cross_data:
            blocks.insert(1, np.stack(cross_data, axis=1))

        data_combined = np.hstack(blocks)

        attrs = _combine_attrs(phxtof_allt[scope].attrs, extof_allt[scope].attrs)

        out_dict[scope] = xr.DataArray(
            data_combined,
            coords=[time_data, energy_combined],
            dims=["time", "energy"],
            attrs=attrs,
        )

    # Combine attributes for all telescopes and set energy widths as attributes
    attrs = _combine_attrs(phxtof_allt.attrs, extof_allt.attrs)
    attrs = {
        "delta_energy_minus": energy_combined - low,
        "delta_energy_plus": high - energy_combined,
        **attrs,
    }

    # Create Dataset
    comb_allt = xr.Dataset(out_dict, attrs=attrs)
    comb_allt.time.attrs = _combine_attrs(
        phxtof_allt.time.attrs,
        extof_allt.time.attrs,
    )
    comb_allt.energy.attrs = _combine_attrs(
        phxtof_allt.energy.attrs,
        extof_allt.energy.attrs,
    )

    return comb_allt
