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


def eis_pad_combine_sc(pads):
    r"""Generate composite Pitch Angle Distributions (PAD) from the EIS
    sensors across the MMS spacecraft.

    Parameters
    ----------
    pads : list of xarray.DataArray
        Pitch-angle distribution for all spacecrafts.

    Returns
    -------
    allmms_pad_avg : xarray.DataArray
        Composite pitch angle distribution.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad, pyrfu.mms.eis_spec_combine_sc

    """
    # Determine spacecraft with smallest number of time steps to use as
    # reference spacecraft
    time_size = [len(probe.time.data) for probe in pads]
    ref_sc_time_size, ref_sc_loc = [np.min(time_size), np.argmin(time_size)]
    ref_probe = pads[ref_sc_loc]

    # Define common energy grid across EIS instruments
    n_en_chans = [len(probe.energy.data) for probe in pads]

    size_en, loc_ref_en = [np.min(n_en_chans), np.argmin(n_en_chans)]
    energy_data = [probe.energy.data[:size_en] for probe in pads]
    energy_data = np.stack(energy_data)
    common_energy = np.nanmean(energy_data, axis=0)

    # create PA labels
    n_pabins = len(ref_probe.theta.data)
    size_pabin = 180 / n_pabins
    pa_label = 180. * np.arange(n_pabins) / n_pabins + size_pabin / 2.

    allmms_pad = np.zeros((ref_probe.shape[0], ref_probe.shape[1],
                           ref_probe.shape[2], len(pads)))

    for p, pad_ in enumerate(pads):
        allmms_pad[..., p] = pad_.data[:len(ref_probe.time), ...]

    allmms_pad_avg = np.nanmean(allmms_pad, axis=3)

    allmms_pad_avg = xr.DataArray(allmms_pad_avg,
                                  coords=[ref_probe.time.data, pa_label,
                                          common_energy],
                                  dims=["time", "theta", "energy"])

    return allmms_pad_avg
