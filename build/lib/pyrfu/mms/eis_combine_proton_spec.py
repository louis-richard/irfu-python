#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from cdflib import cdfread

# Local imports
from ..pyrf import datetime642iso8601

from .list_files import list_files

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _check_time(proton_phxtof, proton_extof):
    data_size = [len(proton_phxtof), len(proton_extof)]

    if data_size[0] == data_size[1]:
        # identify mismatching timesteps
        cond = proton_phxtof.time.data != proton_extof.time.data
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            e_inds = []
            ph_inds = []
            for t in range(data_size[1]):
                dt_dummy = np.min(
                    np.abs(proton_extof.time.data[t] - proton_extof.time.data))
                t_ind = np.argmin(np.abs(proton_extof.time.data[t]
                                         - proton_extof.time.data))
                if dt_dummy == 0:
                    e_inds.append(t)
                    if not ph_inds:
                        ph_inds.append(t)
                    else:
                        ph_inds.append(t_ind)

            time_data = proton_extof.time.data[e_inds]
            phxtof_data = proton_phxtof.data[ph_inds, ...]
            extof_data = proton_extof.data[e_inds, ...]
        else:
            time_data = proton_extof.time.data
            phxtof_data = proton_phxtof.data
            extof_data = proton_extof.data

    elif data_size[0] > data_size[1]:
        cond = proton_phxtof.time.data[:data_size[1]] != proton_extof.time.data
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            e_inds = []
            ph_inds = []
            for t in range(data_size[1]):
                dt_dummy = np.min(np.abs(proton_extof.time.data[t]
                                         - proton_phxtof.time.data))
                t_ind = np.argmin(np.abs(proton_extof.time.data[t]
                                         - proton_phxtof.time.data))
                if dt_dummy == 0:
                    e_inds.append(t)
                    if not ph_inds:
                        ph_inds.append(t)
                    else:
                        ph_inds.append(t_ind)

            time_data = proton_extof.time.data[e_inds]
            phxtof_data = proton_phxtof.data[ph_inds, ...]
            extof_data = proton_extof.data[e_inds, ...]
        else:
            time_data = proton_extof.time.data
            phxtof_data = proton_phxtof.data[:data_size[1], ...]
            extof_data = proton_extof.data

    elif data_size[0] < data_size[1]:
        cond = proton_phxtof.time.data != proton_extof.time.data[:data_size[0]]
        bad_inds = np.where(cond)[0]

        if bad_inds.size:
            e_inds = []
            ph_inds = []
            for t in range(data_size[0]):
                dt_dummy = np.min(np.abs(proton_phxtof.time.data[t]
                                         - proton_phxtof.time.data))
                t_ind = np.argmin(np.abs(
                    proton_phxtof.time.data[t] - proton_phxtof.time.data))
                if dt_dummy == 0:
                    ph_inds.append(t)
                    if not ph_inds:
                        e_inds.append(t)
                    else:
                        e_inds.append(t_ind)

            time_data = proton_phxtof.time.data[ph_inds]
            phxtof_data = proton_phxtof.data[ph_inds, ...]
            extof_data = proton_extof.data[e_inds, ...]

        else:
            time_data = proton_phxtof.time.data
            phxtof_data = proton_phxtof.data
            extof_data = proton_extof.data[:data_size[0], ...]

    else:
        raise ValueError

    return time_data, phxtof_data, extof_data


def _get_energy_dplus_dminus(eis_allt, data_path):
    tint = list(datetime642iso8601(eis_allt.time.data[[0, -1]]))

    name_ = eis_allt.t0.attrs["FIELDNAM"]

    mms_id = int(name_.split("_")[0][-1])

    var = {"inst": "epd-eis", "lev": "l2"}

    if "brst" in name_:
        var["tmmode"] = "brst"
    else:
        var["tmmode"] = "srvy"

    var["dtype"] = name_.split("_")[-5]

    files = list_files(tint, mms_id, var, data_path=data_path)

    with cdfread.CDF(files[0]) as file:
        d_plus = file.varget(eis_allt.t0.energy.attrs["DELTA_PLUS_VAR"])
        d_minus = file.varget(eis_allt.t0.energy.attrs["DELTA_MINUS_VAR"])

    return d_plus, d_minus


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

    data_path = extof_allt.attrs["data_path"]
    dp_phxtof, dm_phxtof = _get_energy_dplus_dminus(phxtof_allt, data_path)
    dp_extof, dm_extof = _get_energy_dplus_dminus(extof_allt, data_path)

    out_keys = list(filter(lambda x: x not in scopes_extof, extof_allt))
    out_dict = {k: extof_allt[k] for k in out_keys if k != "sector"}

    comb_en_low, comb_en_hig = [None] * 2

    time_sect, phxtof_sect, extof_sect = _check_time(phxtof_allt["sector"],
                                                     extof_allt["sector"])
    sect = xr.DataArray(extof_sect, coords=[time_sect], dims=["time"])

    time_spin, phxtof_spin, extof_spin = _check_time(phxtof_allt["spin"],
                                                     extof_allt["spin"])

    spin = xr.DataArray(extof_spin, coords=[time_sect], dims=["time"])

    for scope in scopes_phxtof:
        proton_phxtof = phxtof_allt[scope]
        proton_extof = extof_allt[scope]

        time_data, phxtof_data, extof_data = _check_time(proton_phxtof,
                                                         proton_extof)

        en_phxtof, en_extof = [proton_phxtof.energy.data,
                               proton_extof.energy.data]
        idx_phxtof = np.where(en_phxtof < en_extof[0])[0]
        cond_ = np.logical_and(en_phxtof > en_extof[0],
                               en_phxtof < en_phxtof[-1])
        idx_phxtof_cross = np.where(cond_)[0]

        idx_extof_cross = np.where(en_extof < en_phxtof[-2])[0]
        idx_extof = np.where(en_extof > en_phxtof[-2])[0]

        n_phxtof = idx_phxtof.size
        n_phxtof_cross = idx_phxtof_cross.size
        n_extof = idx_extof.size

        n_en = n_phxtof + n_phxtof_cross + n_extof

        comb_en, comb_en_low, comb_en_hig = [np.zeros(n_en) for _ in range(3)]

        comb_array = np.zeros((len(time_data), n_en))
        comb_array[:, :n_phxtof] = phxtof_data[:, idx_phxtof]
        comb_en[:n_phxtof] = en_phxtof[idx_phxtof]
        comb_en_low[:n_phxtof] = comb_en[:n_phxtof] - dm_phxtof[idx_phxtof]
        comb_en_hig[:n_phxtof] = comb_en[:n_phxtof] + dp_phxtof[idx_phxtof]

        for (i, i_phx), i_ex in zip(enumerate(idx_phxtof_cross),
                                    idx_extof_cross):
            idx_ = n_phxtof + i
            comb_array[:, idx_] = np.nanmean(np.vstack([phxtof_data[:, i_phx],
                                                        extof_data[:, i_ex]]),
                                             axis=0)
            comb_en_low[idx_] = np.nanmin([en_phxtof[idx_] - dm_phxtof[idx_],
                                           en_extof[i] - dm_extof[i]])
            comb_en_hig[idx_] = np.nanmax([en_phxtof[idx_] + dp_phxtof[idx_],
                                           en_extof[i] + dp_extof[i]])
            comb_en[idx_] = np.sqrt(comb_en_low[idx_] * comb_en_hig[idx_])

        comb_array[:, -n_extof:] = extof_data[:, idx_extof]
        comb_en[-n_extof:] = en_extof[idx_extof]
        comb_en_low[-n_extof:] = en_extof[idx_extof] - dm_extof[idx_extof]
        comb_en_hig[-n_extof:] = en_extof[idx_extof] + dp_extof[idx_extof]

        out_dict[scope] = xr.DataArray(comb_array,
                                       coords=[time_data, comb_en],
                                       dims=["time", "energy"])

    out_dict["sector"] = sect
    out_dict["spin"] = spin
    out_dict["energy_dminus"] = comb_en_low
    out_dict["energy_dplus"] = comb_en_hig

    comb_allt = xr.Dataset(out_dict)

    return comb_allt
