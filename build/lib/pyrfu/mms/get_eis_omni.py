#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_eis_omni.py

@author : Louis RICHARD
"""

from .list_files import list_files
from .db_get_ts import db_get_ts


def get_eis_omni(inp_str="Flux_extof_proton_srvy_l2", tint=None, mms_id=2, verbosity=True):
	"""
	Computes omni directional energy spectrum of the target data unit for the target specie over the target energy range

	Parameters :
		inp_str : str
			Key of the target variable like {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}

		tint : list of str
			Time interval

		mms_id : int/float/str
			Index of the spacecraft

	Returns :
		out : DataArray
			Energy spectrum of the target data unit for the target specie in omni direction
	"""

	if not isinstance(mms_id, int):
		mms_id = int(mms_id)

	data_unit, data_type, specie, data_rate, data_lvl = inp_str.split("_")

	var = {"mms_id": mms_id, "inst": "epd-eis"}

	pref = f"mms{mms_id:d}_epd_eis"

	if data_rate == "brst":
		var["tmmode"] = data_rate

		pref = f"{pref}_{data_rate}"
	elif data_rate == "srvy":
		var["tmmode"] = data_rate

		pref = pref
	else:
		raise ValueError("Invalid data rate")

	var["lev"] = data_lvl

	if data_type == "electronenergy":
		if specie == "electron":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		else:
			raise ValueError("invalid specie")
	elif data_type == "extof":
		if specie == "proton":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		elif specie == "oxygen":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		elif specie == "alpha":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		else:
			raise ValueError("invalid specie")
	elif data_type == "phxtof":
		if specie == "proton":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		elif specie == "oxygen":
			var["dtype"], var["specie"] = [data_type, specie]

			pref = f"{pref}_{data_type}_{specie}"
		else:
			raise ValueError("Invalid specie")
	else:
		raise ValueError("Invalid data type")

	files = list_files(tint, mms_id, var)
	
	file_version = int(files[0].split("_")[-1][1])

	var["version"] = file_version

	if data_unit.lower() in ["flux", "counts", "cps"]:
		suf = f"P{file_version:d}_{data_unit.lower()}_t"
	else:
		raise ValueError("Invalid data unit")

	dset_name = f"mms{var['mms_id']:d}_{var['inst']}_{var['tmmode']}_{var['lev']}_{var['dtype']}"
	cdf_names = [f"{pref}_{suf}{t:d}" for t in range(6)]

	out_dict = {}

	flux_omni = None

	for i, cdfname in enumerate(cdf_names):
		scope_key = f"t{i:d}"
		
		if verbosity:
			print(f"Loading {cdfname}...")

		out_dict[scope_key] = db_get_ts(dset_name, cdfname, tint)
		try:
			flux_omni += out_dict[scope_key]
		except TypeError:
			flux_omni = out_dict[scope_key]

	flux_omni /= 6

	return flux_omni
