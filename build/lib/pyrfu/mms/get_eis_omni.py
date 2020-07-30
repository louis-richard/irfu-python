import numpy as np
import xarray as xr

from .list_files import list_files
from .db_get_ts import db_get_ts


def get_eis_omni(inp_str="Flux_extof_proton_srvy_l2",trange=None,mmsId=2,silent=False):
	"""
	Computes omni directional energy spectrum of the target data unit for the target specie over the target energy range

	Parameters :
		inp_str : str
			Key of the target variable like {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}

		trange : list of str
			Time interval

		mmsId : int/float/str
			Index of the spacecraft

	Returns :
		out : DataArray
			Energy spectrum of the target data unit for the target specie in omni direction
	"""

	if not isinstance(mmsId,int):
		mmsId = int(mmsId)

	data_unit, data_type, specie, data_rate, data_lvl = inp_str.split("_")

	Var = {}
	Var["mmsId"] 	= mmsId
	Var["inst"] 	= "epd-eis"

	pref = "mms{:d}_epd_eis".format(mmsId)

	if data_rate == "brst":
		Var["tmmode"] 	= data_rate
		pref = "{}_{}".format(pref,data_rate)
	elif data_rate == "srvy":
		Var["tmmode"] 	= data_rate
		pref = pref
	else :
		raise ValueError("Invalid data rate")

	Var["lev"] 		= data_lvl

	if data_type == "electronenergy":
		if specie == "electron":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		else :
			raise ValueError("invalid specie")
	elif data_type == "extof":
		if specie == "proton":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		elif specie == "oxygen":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		elif specie == "alpha":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		else :
			raise ValueError("invalid specie")
	elif data_type == "phxtof":
		if specie == "proton":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		elif specie == "oxygen":
			Var["dtype"] 	= data_type
			Var["specie"] 	= specie

			pref = "{}_{}_{}".format(pref,data_type,specie)
		else :
			raise ValueError("Invalid specie")
	else :
		raise ValueError("Invalid data type")

	files = list_files(trange,mmsId,Var)
	
	file_version 	= int(files[0].split("_")[-1][1])
	Var["version"] 	= file_version

	if data_unit.lower() in ["flux","counts","cps"]:
		suf = "P{:d}_{}_t".format(file_version,data_unit.lower())
	else :
		raise ValueError("Invalid data unit")

	dsetName = "mms{:d}_{}_{}_{}_{}".format(Var["mmsId"],Var["inst"],Var["tmmode"],Var["lev"],Var["dtype"])
	cdfnames = ["{}_{}{:d}".format(pref,suf,t) for t in range(6)]


	outdict = {}
	flux_omni = None

	for i, cdfname in enumerate(cdfnames):
		scope_key = "t{:d}".format(i)
		
		if silent == False:
			print("Loading "+ cdfname+"...")

		outdict[scope_key] = db_get_ts(dsetName,cdfname,trange)
		try :
			flux_omni += outdict[scope_key]
		except TypeError:
			flux_omni = outdict[scope_key]


	flux_omni /= 6

	return flux_omni
