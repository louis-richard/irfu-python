#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_feeps_pad.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
import warnings

from astropy.time import Time

from .feeps_pitch_angles import feeps_pitch_angles
from .get_feeps_active_eyes import get_feeps_active_eyes


def calc_feeps_pad(inp_dset=None, b_bcs=None, bin_size=16.3636, energy=[70, 600]):
	"""
	Compute pitch angle distribution using FEEPS data.

	Parameters :
		inp_dset : Dataset
			Energy spectrum of all eyes

		Bbcs : DataArray
			Time series of the magnetic field in spacecraft coordinates

		bin_size : float
			Width of the pitch angles bins

		energy : list of float
			Energy range of particles

	Returns :
		pad : DataArray
			Time series of the pitch angle distribution
			
	"""
	var = inp_dset.attrs
	mms_id = var["mmsId"]

	if var["dtype"] == "electron":
		dangresp = 21.4  # deg
	elif var["dtype"] == "ion":
		dangresp = 10.0  # deg
	else:
		raise ValueError("Invalid specie")

	if energy[0] < 32.0:
		print("Please select a starting energy of 32 keV or above")
		return

	n_pabins = 180 / bin_size
	pa_bins = [180. * pa_bin / n_pabins for pa_bin in range(0, int(n_pabins)+1)]
	pa_label = [180. * pa_bin / n_pabins + bin_size / 2. for pa_bin in range(0, int(n_pabins))]

	pitch_angles = feeps_pitch_angles(inp_dset, b_bcs)
	pa_times = pitch_angles.time
	pa_data = pitch_angles.data

	trange = Time(np.hstack([pa_times.data.min(), pa_times.data.max()]), format="datetime64").isot

	eyes = get_feeps_active_eyes(var, trange, mms_id)

	pa_data_map = {}

	if var["tmmode"] == "srvy":
		if var["dtype"] == "electron":
			pa_data_map["top-electron"] = idx_maps["electron-top"]
			pa_data_map["bottom-electron"] = idx_maps["electron-bottom"]
		elif var["dtype"] == "ion":
			pa_data_map["top-ion"] = idx_maps["electron-top"]
			pa_data_map["bottom-ion"] = idx_maps["ion-bottom"]
		else:
			raise ValueError("Invalid specie")

	elif var["tmmode"] == "brst":
		# note: the following are indices of the top/bottom sensors in pa_data
		# they should be consistent with pa_dlimits.labels
		pa_data_map["top-electron"], pa_data_map["bottom-electron"] = [np.arange(9), np.arange(9, 18)]

		# and ions:
		pa_data_map["top-ion"], pa_data_map["bottom-ion"] = [[0, 1, 2], [3, 4, 5]]

	sensor_types = ["top", "bottom"]

	if var["dtype"] == "electron":
		dflux = np.zeros([len(pa_times), len(pa_data_map["top-electron"])+len(pa_data_map["bottom-electron"])])

		dpa = np.zeros([len(pa_times), len(pa_data_map["top-electron"])+len(pa_data_map["bottom-electron"])])
	elif var["dtype"] == "ion":
		dflux = np.zeros([len(pa_times), len(pa_data_map["top-ion"])+len(pa_data_map["bottom-ion"])])

		dpa = np.zeros([len(pa_times), len(pa_data_map["top-ion"])+len(pa_data_map["bottom-ion"])])
	else:
		raise TypeError("Invalid specie")

	for s_type in sensor_types:
		pa_map = pa_data_map[s_type + "-" + var["dtype"]]

		particle_idxs = [eye-1 for eye in eyes[s_type]]

		for isen, sensor_num in enumerate(particle_idxs):
			var_name = "{}-{:d}".format(s_type, sensor_num + 1)

			t, data, = inp_dset[var_name].time.data, inp_dset[var_name].data
			energies = inp_dset[var_name].Differential_energy_channels.data

			# remove any 0s before averaging
			data[data == 0] = "nan"

			# assumes all energies are NaNs if the first is
			if np.isnan(energies[0]):
				continue

			# energy indices to use:
			idx = np.where((energies >= energy[0]) & (energies <= energy[1]))
			
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				dflux[:, pa_map[isen]] = np.nanmean(data[:, idx[0]], axis=1)

			dpa[:, pa_map[isen]] = pa_data[:, pa_map[isen]]

	# we need to replace the 0.0s left in after populating dpa with NaNs; these 
	# 0.0s are left in there because these points aren't covered by sensors loaded
	# for this datatype/data_rate
	dpa[dpa == 0] = "nan"

	pa_flux, delta_pa = [np.zeros([len(pa_times), int(n_pabins)]), (pa_bins[1] - pa_bins[0]) / 2.0]

	# Now loop through PA bins and time, find the telescopes where there is data in those bins and average it up!
	for pa_idx, pa_time in enumerate(pa_times):
		for ipa in range(0, int(n_pabins)):
			if not np.isnan(dpa[pa_idx, :][0]):
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					ind = np.where((dpa[pa_idx, :] + dangresp >= pa_label[ipa]-delta_pa)
								& (dpa[pa_idx, :]-dangresp < pa_label[ipa]+delta_pa))
					
					if ind[0].size != 0:
						if len(ind[0]) > 1:
							pa_flux[pa_idx, ipa] = np.nanmean(dflux[pa_idx, ind[0]], axis=0)
						else:
							pa_flux[pa_idx, ipa] = dflux[pa_idx, ind[0]]

	pa_flux[pa_flux == 0] = "nan"  # fill any missed bins with NAN

	pad = xr.DataArray(pa_flux, coords=[t, pa_label], dims=["time", "pa"], attrs=var)

	return pad