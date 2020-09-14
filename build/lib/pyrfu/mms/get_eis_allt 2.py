#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_eis_allt.py

@author : Louis RICHARD
"""

import xarray as xr

from .list_files import list_files
from .db_get_ts import db_get_ts


def get_eis_allt(inp_str="Flux_extof_proton_srvy_l2", trange=None, mms_id=2, silent=False):
	"""
	Read energy spectrum of the selected specie in the selected energy range for all telescopes.
	
	Parameters :
		inp_str : str
			Key of the target variable like {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}

		trange : list of str
			Time interval
	
		mms_id : int/float/str
			Index of the spacecraft

		silent : bool
	
	Returns :
		out : Dataset
			Dataset containing the energy spectrum of the 6 telescopes of the Energy Ion Spectrometer
	"""
	
	# Convert mms_id to integer
	if not isinstance(mms_id, int):
		mms_id = int(mms_id)
	
	data_unit, data_type, specie, data_rate, data_lvl = inp_str.split("_")
	
	var = {"mms_id": mms_id, "inst": "epd-eis", "tmmode": data_rate, "lev": data_lvl}
	
	if data_type == "electronenergy":
		if specie == "electron":
			var["dtype"], var["specie"] = [data_type, specie]
		
			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		else:
			raise ValueError("invalid specie")

	elif data_type == "extof":
		if specie == "proton":
			var["dtype"], var["specie"] = [data_type, specie]
		
			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		elif specie == "oxygen":
			var["dtype"], var["specie"] = [data_type, specie]
		
			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		elif specie == "alpha":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		else:
			raise ValueError("invalid specie")

	elif data_type == "phxtof":
		if specie == "proton":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		elif specie == "oxygen":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
		else:
			raise ValueError("Invalid specie")
	else:
		raise ValueError("Invalid data type")
	
	# EIS includes the version of the files in the cdfname need to read it before.
	files = list_files(trange, mms_id, var)
	
	file_version = int(files[0].split("_")[-1][1])
	var["version"] = file_version
	
	if data_unit.lower() in ["flux", "counts", "cps"]:
		suf = "P{:d}_{}_t".format(file_version, data_unit.lower())
	else:
		raise ValueError("Invalid data unit")
	
	# Name of the data containing index of the probe, instrument, data rate, data level and data type if needed
	dset_name = f"mms{var['mms_id']:d}_{var['inst']}_{var['tmmode']}_{var['lev']}_{var['dtype']}"
	
	# Names of the energy spectra in the CDF (one for each telescope)
	cdfnames = ["{}_{}{:d}".format(pref, suf, t) for t in range(6)]
	
	outdict = {}
	for i, cdfname in enumerate(cdfnames):
		scope_key = f"t{i:d}"
	
		if not silent:
			print(f"Loading {cdfname}...")

		outdict[scope_key] = db_get_ts(dset_name, cdfname, trange)
	
	# Build Dataset
	out = xr.Dataset(outdict, attrs=var)
	
	return out
