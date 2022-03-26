#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .feeps_pitch_angles import feeps_pitch_angles
from .feeps_active_eyes import feeps_active_eyes

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

# Angular response (finite field of view) of instruments electrons can use
# +/- 21.4 deg on each pitch angle as average response angle; ions can start
# with +/-10 deg, but both need to be further refined
angular_repsonse = {"electron": 21.4, "ion": 10}


def _pa_data_map(idx_maps, d_type, d_rate):
    pa_data_map = {}

    if d_rate == "srvy":
        pa_data_map[f"top-{d_type}"] = idx_maps[f"{d_type}-top"]
        pa_data_map[f"bottom-{d_type}"] = idx_maps[f"{d_type}-bottom"]
    else:
        # note: the following are indices of the top/bottom sensors in pa_data
        # they should be consistent with pa_dlimits.labels
        pa_data_map["top-electron"] = np.arange(9)
        pa_data_map["bottom-electron"] = np.arange(9, 18)

        # and ions:
        pa_data_map["top-ion"] = [0, 1, 2]
        pa_data_map["bottom-ion"] = [3, 4, 5]

    return pa_data_map


def _dpa_dflux(inp_dataset, pitch_angles, pa_data_map, energy, d_type, mms_id):
    pa_times = pitch_angles.time
    pa_data = pitch_angles.data

    trange = np.datetime_as_string(np.hstack([np.min(pa_times.data),
                                              np.max(pa_times.data)]), "ns")

    eyes = feeps_active_eyes(inp_dataset.attrs, list(trange), mms_id)

    sensor_types = ["top", "bottom"]

    n_times = len(pa_times)
    n_top = len(pa_data_map[f"top-{d_type}"])
    n_bottom = len(pa_data_map[f"bottom-{d_type}"])

    dflux, dpa = [np.zeros([n_times, n_top + n_bottom]) for _ in range(2)]

    for s_type in sensor_types:
        pa_map = pa_data_map[f"{s_type}-{d_type}"]

        particle_idxs = [eye - 1 for eye in eyes[s_type]]

        for isen, sensor_num in enumerate(particle_idxs):
            var_name = "{}-{:d}".format(s_type, sensor_num + 1)

            data = inp_dataset[var_name].data
            energies = inp_dataset[inp_dataset[var_name].dims[1]].data

            # remove any 0s before averaging
            data[data == 0] = "nan"

            # assumes all energies are NaNs if the first is
            if np.isnan(energies[0]):
                continue

            # energy indices to use:
            idx = np.where(np.logical_and(energies >= energy[0],
                                          energies <= energy[1]))[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dflux[:, pa_map[isen]] = np.nanmean(data[:, idx], axis=1)

            dpa[:, pa_map[isen]] = pa_data[:, pa_map[isen]]

    # we need to replace the 0.0s left in after populating dpa with NaNs;
    # these 0.0s are left in there because these points aren't covered by
    # sensors loaded for this datatype/d_ratee
    dpa[dpa == 0] = "nan"

    return dpa, dflux


def _pa_flux(pa_times, pa_bins, pa_labels, dpa, dflux, d_type):
    n_pabins = len(pa_bins) - 1
    # Account for angular response
    dangresp = angular_repsonse[d_type]

    pa_flux = np.zeros([len(pa_times), int(n_pabins)])
    delta_pa = (pa_bins[1] - pa_bins[0]) / 2.0

    # Now loop through PA bins and time, find the telescopes where there is
    # data in those bins and average it up!
    for (pa_idx, pa_time), ipa in itertools.product(enumerate(pa_times),
                                                    range(n_pabins)):
        if not np.isnan(dpa[pa_idx, :][0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ind = np.where(
                    (dpa[pa_idx, :] + dangresp >= pa_labels[ipa] - delta_pa)
                    & (dpa[pa_idx, :] - dangresp < pa_labels[ipa] + delta_pa))

                if ind[0].size != 0:
                    if len(ind[0]) > 1:
                        pa_flux[pa_idx, ipa] = np.nanmean(
                            dflux[pa_idx, ind[0]], axis=0)
                    else:
                        pa_flux[pa_idx, ipa] = dflux[pa_idx, ind[0]]

    pa_flux[pa_flux == 0] = "nan"  # fill any missed bins with NAN

    return pa_flux


def feeps_pad(inp_dataset, b_bcs, bin_size: float = 16.3636,
              energy: list = None):
    r"""Compute pitch angle distribution using FEEPS data.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Energy spectrum of all eyes.
    b_bcs : xarray.DataArray
        Time series of the magnetic field in spacecraft coordinates.
    bin_size : float, Optional
        Width of the pitch angles bins. Default is 16.3636.
    energy : array_like, Optional
        Energy range of particles. Default is [70., 600.]

    Returns
    -------
    pad : xarray.DataArray
        Time series of the pitch angle distribution.

    """

    if energy is None:
        energy = [70., 600.]

    assert energy[0] > 32., "Please use a starting energy of 32 keV or above"

    time = inp_dataset.time.data
    attrs = inp_dataset.attrs
    mms_id, d_type, d_rate = list(map(attrs.get, ["mmsId", "dtype", "tmmode"]))

    assert d_rate in ["srvy", "brst"]
    assert d_type in ["electron", "ion"]

    n_pabins = int(180 / bin_size)
    pa_bins = [180. * pa_bin / n_pabins for pa_bin in range(n_pabins + 1)]
    pa_labels = [pa_bin  + bin_size / 2. for pa_bin in pa_bins[:-1]]

    pitch_angles, idx_maps = feeps_pitch_angles(inp_dataset, b_bcs)

    pa_data_map = _pa_data_map(idx_maps, d_type, d_rate)

    dpa, dflux = _dpa_dflux(inp_dataset, pitch_angles, pa_data_map, energy,
                            d_type, mms_id)

    pa_flux = _pa_flux(pitch_angles.time, pa_bins, pa_labels, dpa, dflux,
                       d_type)

    pad = xr.DataArray(pa_flux, coords=[time, pa_labels],
                       dims=["time", "theta"], attrs=attrs)

    return pad
