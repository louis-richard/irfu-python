import numpy as np
import xarray as xr
from astropy.time import Time

from .db_get_ts import db_get_ts
from .read_feeps_sector_masks_csv import read_feeps_sector_masks_csv

def feeps_remove_sun(inp_dset):

	Var 	= inp_dset.attrs
	trange 	= list(Time(inp_dset.time.data[[0,-1]],format="datetime64").isot)  

	dsetName = "mms{:d}_feeps_{}_{}_{}".format(Var["mmsId"],Var["tmmode"],Var["lev"],Var["dtype"])
	dsetPref = "mms{:d}_epd_feeps_{}_{}_{}".format(Var["mmsId"],Var["tmmode"],Var["lev"],Var["dtype"])	

	spin_sectors = db_get_ts(dsetName,"_".join([dsetPref,"spinsectnum"]),trange)
	mask_sectors = read_feeps_sector_masks_csv(trange)

	outdict = {}

	for k in inp_dset:
		outdict[k] = inp_dset[k]
		if mask_sectors.get("mms{:d}_imask_{}".format(Var["mmsId"],k)) is not None:
			bad_sectors = mask_sectors["mms{:d}_imask_{}".format(Var["mmsId"],k)]

			for bad_sector in bad_sectors:
				this_bad_sector = np.where(spin_sectors == bad_sector)[0]
				if len(this_bad_sector) != 0:
					outdict[k].data[this_bad_sector] = np.nan

	out 		= xr.Dataset(outdict,attrs=Var)
	out.attrs 	= Var

	return out