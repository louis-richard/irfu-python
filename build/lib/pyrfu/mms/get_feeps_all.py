import numpy as np
import xarray as xr
from .db_get_ts import db_get_ts
from .get_feeps_eye import get_feeps_eye
from .get_feeps_active_eyes import get_feeps_active_eyes


def get_feeps_all(tar_var="fluxe_brst_l2",probe=1,trange=None):

	Var = {}

	data_units 		= tar_var.split("_")[0][:-1]
	specie 			= tar_var.split("_")[0][-1]
	Var["tmmode"] 	= tar_var.split("_")[1]
	Var["lev"] 		= tar_var.split("_")[2]

	if specie == "e":
		Var["dtype"] = "electron"
	elif specie == "i":
		Var["dtype"] = "ion"
	else :
		raise ValueError("Invalid specie")

	active_eyes = get_feeps_active_eyes(trange, probe, Var)

	eIds = ["{}-{:d}".format(k,s)  for k in active_eyes for s in active_eyes[k]]

	outdict = {}

	for eId in eIds: outdict[eId] = get_feeps_eye(tar_var,probe,eId,trange)

	out = xr.Dataset(outdict)

	dsetName = "mms{:d}_feeps_{}_{}_{}".format(probe,Var["tmmode"],Var["lev"],Var["dtype"])
	dsetPref = "mms{:d}_epd_feeps_{}_{}_{}".format(probe,Var["tmmode"],Var["lev"],Var["dtype"])	

	spin_sectors 				= db_get_ts(dsetName,"_".join([dsetPref,"spinsectnum"]),trange)
	out.attrs["spin_sectors"] 	= spin_sectors

	return out