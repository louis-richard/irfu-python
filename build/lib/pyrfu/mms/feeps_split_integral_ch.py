#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feeps_split_integral_ch.py

@author : Louis RICHARD
"""

import xarray as xr


def feeps_split_integral_ch(inp_dset=None):
	"""
	This function splits the last integral channel from the FEEPS spectra,
	creating 2 new DataArrays:

	Parameters:
		inp_dset: xr.DataArray
			Energetic particles energy spectrum from FEEPS

	Returns:
		out: xr.DataArray
			Energetic particles energy spectra with the integral channel removed
		out_500kev: xr.DataArray
			Integral channel that was removed
	"""

	outdict, outdict_500kev = [{}, {}]

	for k in inp_dset:
		try:
			# Energy spectra with the integral channel removed
			outdict[k] = inp_dset[k][:, :-1]

			# Integral channel that was removed
			outdict_500kev[k] = inp_dset[k][:, -1]
		except IndexError:
			pass

	out = xr.Dataset(outdict, attrs=inp_dset.attrs)

	out_500kev = xr.Dataset(outdict_500kev, attrs=inp_dset.attrs)

	return out, out_500kev
