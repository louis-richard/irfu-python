#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _idx_closest(lst0, lst1):
    return [(np.abs(np.asarray(lst0) - k)).argmin() for k in lst1]


def eis_skymap_combine_sc(skymaps, method: str = "mean"):
    r"""Generate composite skymap from the EIS sensors across the MMS
    spacecraft.

    Parameters
    ----------
    skymaps : list of xarray.DataArray
        Skymap distribution for all spacecraft.
    method : str, Optional
        Method to combine spectra, "mean" or "sum"

    Returns
    -------
    out : xarray.Dataset
        Composite skymap distribution

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad,
    pyrfu.mms.eis_spec_combine_sc, pyrfu.mms.eis_spec_combine_sc

    """
    assert method.lower() in ["mean", "sum"]

    # Determine spacecraft with smallest number of time steps to use as
    # reference spacecraft
    time_size = [len(probe.time.data) for probe in skymaps]
    ref_sc_time_size, ref_sc_loc = [np.min(time_size), np.argmin(time_size)]
    ref_probe = skymaps[ref_sc_loc]

    # Define common energy grid across EIS instruments
    n_en_chans = [probe.energy.shape[1] for probe in skymaps]
    size_en, loc_ref_en = [np.min(n_en_chans), np.argmin(n_en_chans)]
    ref_energy = skymaps[loc_ref_en].energy.data[0, :]

    energy_data, e_plus, e_minu = [[], [], []]
    for probe in skymaps:
        idx = _idx_closest(probe.energy.data[0, :], ref_energy)
        energy_data.append(probe.energy.data[0, idx])
        e_minu.append(probe.attrs["energy_dminus"][idx])
        e_plus.append(probe.attrs["energy_dplus"][idx])

    energy_data = np.stack(energy_data)
    common_energy = np.nanmean(energy_data, axis=0)
    common_energy = np.tile(common_energy, (ref_sc_time_size, 1))

    #
    e_minu = np.stack(e_minu)
    e_plus = np.stack(e_plus)
    common_minu = np.nanmean(e_minu, axis=0)
    common_plus = np.nanmean(e_plus, axis=0)

    # Use azimuthal and elevation angle from reference spacecraft (in
    # practice they are the same for all spacecraft)
    phi = ref_probe.phi.data
    theta = ref_probe.theta.data

    allmms_skymap = np.zeros([ref_sc_time_size, size_en, phi.shape[1],
                              len(theta), len(skymaps)])

    for p, skymap in enumerate(skymaps):
        idx_en = _idx_closest(skymap.energy.data[0, :], common_energy[0, :])
        allmms_skymap[..., p] = skymap.data[:ref_sc_time_size, idx_en, ...]

    if method.lower() == "mean":
        # Average the four spacecraft
        allmms_skymap_avg = np.nanmean(allmms_skymap, axis=-1)
    else:
        # Sum the four spacecraft
        allmms_skymap_avg = np.nansum(allmms_skymap, axis=-1)

    # Create combined skymap
    out_dict = {"time": ref_probe.time.data,
                "idx0": range(common_energy.shape[1]),
                "idx1": range(phi.shape[1]), "idx2": range(len(theta)),
                "data": (["time", "idx0", "idx1", "idx2"], allmms_skymap_avg),
                "energy": (["time", "idx0"], common_energy),
                "phi": (["time", "idx1"], phi), "theta": (["idx2"], theta)}

    out = xr.Dataset(out_dict)

    out.attrs["energy_dminus"] = common_minu
    out.attrs["energy_dplus"] = common_plus

    return out
