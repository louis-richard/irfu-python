#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _check_time(proton_phxtof, proton_extof):
    data_size = [len(proton_phxtof), len(proton_extof)]

    if data_size[0] == data_size[1]:
        # identify mismatching timesteps
        cond = proton_phxtof.time.data != proton_extof.time.data
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            extof_inds = []
            phxtof_inds = []
            for i_t in range(data_size[1]):
                dt_dummy = np.min(
                    np.abs(
                        proton_extof.time.data[i_t] - proton_extof.time.data,
                    ),
                )
                t_ind = np.argmin(
                    np.abs(
                        proton_extof.time.data[i_t] - proton_extof.time.data,
                    ),
                )
                if dt_dummy == 0:
                    extof_inds.append(i_t)
                    if not phxtof_inds:
                        phxtof_inds.append(i_t)
                    else:
                        phxtof_inds.append(t_ind)

            time_data = proton_extof.time.data[extof_inds]
            phxtof_data = proton_phxtof.data[phxtof_inds, ...]
            extof_data = proton_extof.data[extof_inds, ...]
        else:
            time_data = proton_extof.time.data
            phxtof_data = proton_phxtof.data
            extof_data = proton_extof.data

    elif data_size[0] > data_size[1]:
        cond = proton_phxtof.time.data[: data_size[1]] != proton_extof.time.data
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            extof_inds = []
            phxtof_inds = []
            for i_t in range(data_size[1]):
                dt_dummy = np.min(
                    np.abs(
                        proton_extof.time.data[i_t] - proton_phxtof.time.data,
                    ),
                )
                t_ind = np.argmin(
                    np.abs(
                        proton_extof.time.data[i_t] - proton_phxtof.time.data,
                    ),
                )
                if dt_dummy == 0:
                    extof_inds.append(i_t)
                    if not phxtof_inds:
                        phxtof_inds.append(i_t)
                    else:
                        phxtof_inds.append(t_ind)

            time_data = proton_extof.time.data[extof_inds]
            phxtof_data = proton_phxtof.data[phxtof_inds, ...]
            extof_data = proton_extof.data[extof_inds, ...]
        else:
            time_data = proton_extof.time.data
            phxtof_data = proton_phxtof.data[: data_size[1], ...]
            extof_data = proton_extof.data

    elif data_size[0] < data_size[1]:
        cond = proton_phxtof.time.data != proton_extof.time.data[: data_size[0]]
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            extof_inds = []
            phxtof_inds = []
            for i_t in range(data_size[0]):
                dt_dummy = np.min(
                    np.abs(
                        proton_phxtof.time.data[i_t] - proton_phxtof.time.data,
                    ),
                )
                t_ind = np.argmin(
                    np.abs(
                        proton_phxtof.time.data[i_t] - proton_phxtof.time.data,
                    ),
                )
                if dt_dummy == 0:
                    phxtof_inds.append(i_t)
                    if not phxtof_inds:
                        extof_inds.append(i_t)
                    else:
                        extof_inds.append(t_ind)

            time_data = proton_phxtof.time.data[phxtof_inds]
            phxtof_data = proton_phxtof.data[phxtof_inds, ...]
            extof_data = proton_extof.data[extof_inds, ...]

        else:
            time_data = proton_phxtof.time.data
            phxtof_data = proton_phxtof.data
            extof_data = proton_extof.data[: data_size[0], ...]

    else:
        raise ValueError

    return time_data, phxtof_data, extof_data


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
        the 6 telescopes.

    """

    scopes_phxtof = list(filter(lambda x: x[0] == "t", phxtof_allt))
    scopes_extof = list(filter(lambda x: x[0] == "t", extof_allt))
    assert scopes_extof == scopes_phxtof

    # Get energy deltas for PHxTOF and ExTOF
    delta_energy_plus_phxtof = phxtof_allt.attrs["delta_energy_plus"]
    delta_energy_minus_phxtof = phxtof_allt.attrs["delta_energy_minus"]

    delta_energy_plus_extof = extof_allt.attrs["delta_energy_plus"]
    delta_energy_minus_extof = extof_allt.attrs["delta_energy_minus"]

    out_keys = list(filter(lambda x: x not in scopes_extof, extof_allt))
    out_dict = {k: extof_allt[k] for k in out_keys if k != "sector"}

    energy_combined_low, energy_combined_hig = [None] * 2

    time_sect, _, extof_sect = _check_time(
        phxtof_allt["sector"],
        extof_allt["sector"],
    )
    sect = xr.DataArray(
        extof_sect,
        coords=[time_sect],
        dims=["time"],
        attrs=extof_allt["sector"].attrs,
    )

    _, _, extof_spin = _check_time(phxtof_allt["spin"], extof_allt["spin"])

    spin = xr.DataArray(
        extof_spin,
        coords=[time_sect],
        dims=["time"],
        attrs=extof_allt["spin"].attrs,
    )

    for scope in scopes_phxtof:
        proton_phxtof = phxtof_allt[scope]
        proton_extof = extof_allt[scope]

        time_data, phxtof_data, extof_data = _check_time(
            proton_phxtof,
            proton_extof,
        )

        energy_phxtof = proton_phxtof.energy.data
        energy_extof = proton_extof.energy.data

        idx_phxtof = np.where(energy_phxtof < energy_extof[0])[0]
        cond_ = np.logical_and(
            energy_phxtof > energy_extof[0],
            energy_phxtof < energy_phxtof[-1],
        )
        idx_phxtof_cross = np.where(cond_)[0]

        idx_extof_cross = np.where(energy_extof < energy_phxtof[-2])[0]
        idx_extof = np.where(energy_extof > energy_phxtof[-2])[0]

        n_phxtof = idx_phxtof.size
        n_phxtof_cross = idx_phxtof_cross.size
        n_extof = idx_extof.size

        n_en = n_phxtof + n_phxtof_cross + n_extof

        energy_combined = np.zeros(n_en)
        energy_combined_low, energy_combined_hig = [np.zeros(n_en) for _ in range(2)]

        data_combined = np.zeros((len(time_data), n_en))
        data_combined[:, :n_phxtof] = phxtof_data[:, idx_phxtof]
        energy_combined[:n_phxtof] = energy_phxtof[idx_phxtof]
        energy_combined_low[:n_phxtof] = (
            energy_combined[:n_phxtof] - delta_energy_minus_phxtof[idx_phxtof]
        )
        energy_combined_hig[:n_phxtof] = (
            energy_combined[:n_phxtof] + delta_energy_plus_phxtof[idx_phxtof]
        )

        for (i, i_phx), i_ex in zip(
            enumerate(idx_phxtof_cross),
            idx_extof_cross,
        ):
            idx_ = n_phxtof + i
            data_combined[:, idx_] = np.nanmean(
                np.vstack([phxtof_data[:, i_phx], extof_data[:, i_ex]]),
                axis=0,
            )
            energy_combined_low[idx_] = np.nanmin(
                [
                    energy_phxtof[idx_] - delta_energy_minus_phxtof[idx_],
                    energy_extof[i] - delta_energy_minus_extof[i],
                ],
            )
            energy_combined_hig[idx_] = np.nanmax(
                [
                    energy_phxtof[idx_] + delta_energy_plus_phxtof[idx_],
                    energy_extof[i] + delta_energy_plus_extof[i],
                ],
            )
            energy_combined[idx_] = np.sqrt(
                energy_combined_low[idx_] * energy_combined_hig[idx_],
            )

        data_combined[:, -n_extof:] = extof_data[:, idx_extof]
        energy_combined[-n_extof:] = energy_extof[idx_extof]
        energy_combined_low[-n_extof:] = (
            energy_extof[idx_extof] - delta_energy_minus_extof[idx_extof]
        )
        energy_combined_hig[-n_extof:] = (
            energy_extof[idx_extof] + delta_energy_plus_extof[idx_extof]
        )

        attrs = _combine_attrs(
            phxtof_allt[scope].attrs,
            extof_allt[scope].attrs,
        )

        out_dict[scope] = xr.DataArray(
            data_combined,
            coords=[time_data, energy_combined],
            dims=["time", "energy"],
            attrs=attrs,
        )

    out_dict["sector"] = sect
    out_dict["spin"] = spin

    # Combine attributes for all telescopes and set energy deltas as attributes
    attrs = _combine_attrs(phxtof_allt.attrs, extof_allt.attrs)
    attrs = {
        "delta_energy_minus": energy_combined_low,
        "delta_energy_plus": energy_combined_hig,
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
