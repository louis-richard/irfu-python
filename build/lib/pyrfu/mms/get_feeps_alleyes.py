import numpy as np
import xarray as xr
from .db_get_ts import db_get_ts
from .get_feeps_oneeye import get_feeps_oneeye
from .get_feeps_active_eyes import get_feeps_active_eyes


def get_feeps_alleyes(tar_var="fluxe_brst_l2",mmsId=1,trange=None):

	Var = {}

	data_units 		= tar_var.split("_")[0][:-1]
	specie 			= tar_var.split("_")[0][-1]
	Var["tmmode"] 	= tar_var.split("_")[1]
	Var["lev"] 		= tar_var.split("_")[2]
	Var["mmsId"] 	= mmsId

	if specie == "e":
		Var["dtype"] = "electron"
	elif specie == "i":
		Var["dtype"] = "ion"
	else :
		raise ValueError("Invalid specie")

	active_eyes = get_feeps_active_eyes(Var,trange,mmsId)

	eIds = ["{}-{:d}".format(k,s)  for k in active_eyes for s in active_eyes[k]]

	outdict = {}

	for eId in eIds: outdict[eId] = get_feeps_oneeye(tar_var,eId,trange,mmsId)

	out 		= xr.Dataset(outdict)
	out.attrs 	= Var

	return out