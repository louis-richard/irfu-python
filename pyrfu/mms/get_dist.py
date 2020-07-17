import os
import bisect
import numpy as np
import xarray as xr
from spacepy import pycdf
from dateutil import parser
from ..pyrf.ts_skymap import ts_skymap


def get_dist(file_path="",cdfname="",trange=None):

	tmmode = cdfname.split("_")[-1]
	with pycdf.CDF(file_path) as f:
		if tmmode == "brst":

			DEPEND_0 = f[cdfname].attrs["DEPEND_0"]
			DEPEND_1 = f[cdfname].attrs["DEPEND_1"]
			DEPEND_2 = f[cdfname].attrs["DEPEND_2"]
			DEPEND_3 = f[cdfname].attrs["DEPEND_3"]
			
			t = f[DEPEND_0][...]
			idxl = bisect.bisect_left(t,parser.parse(trange[0]))
			idxr = bisect.bisect_left(t,parser.parse(trange[1]))
			t = t[idxl:idxr]
			dist = f[cdfname][idxl:idxr,...]
			dist = np.transpose(dist,[0,3,1,2])
			ph = f[DEPEND_1][idxl:idxr,...]
			th = f[DEPEND_2][...]
			en = f[DEPEND_3][idxl:idxr,...]
			
			denname = "_".join([cdfname.split("_")[0],cdfname.split("_")[1],"energy_delta",cdfname.split("_")[-1]])
			en0name = "_".join([cdfname.split("_")[0],cdfname.split("_")[1],"energy0",cdfname.split("_")[-1]])
			en1name = "_".join([cdfname.split("_")[0],cdfname.split("_")[1],"energy1",cdfname.split("_")[-1]])
			estname = "_".join([cdfname.split("_")[0],cdfname.split("_")[1],"steptable_parity",cdfname.split("_")[-1]])
			stepTable = f[estname][idxl:idxr,...]
			if denname in f.keys():
				delta_plus_var = f[denname][idxl:idxr,...]
				delta_minus_var = f[denname][idxl:idxr,...]
				
			if not en0name in f.keys():
				if stepTable[0]:
					energy0 = en[0,:]
					energy1 = en[1,:]
				else :
					energy0 = en[1,:]
					energy1 = en[0,:]
			else :
				energy0 = f[en0name][...]
				energy1 = f[en1name][...]

			res = ts_skymap(t,dist,None,ph,th,energy0=energy0,energy1=energy1,esteptable=stepTable)
			if "delta_plus_var" in locals() and "delta_minus_var" in locals():
				res.attrs["delta_energy_minus"] = delta_minus_var
				res.attrs["delta_energy_plus"] = delta_plus_var

			for k in f[cdfname].attrs:
				res.attrs[k] = f[cdfname].attrs[k]

			for k in f.attrs:
				res.attrs[k] = f.attrs[k]

			res.attrs["tmmode"] = tmmode
			if "_dis_" in cdfname:
				res.attrs["species"] = "ions"
			else :
				res.attrs["species"] = "electrons"

		elif tmmode == "fast":
			DEPEND_0 = f[cdfname].attrs["DEPEND_0"]
			DEPEND_1 = f[cdfname].attrs["DEPEND_1"]
			DEPEND_2 = f[cdfname].attrs["DEPEND_2"]
			DEPEND_3 = f[cdfname].attrs["DEPEND_3"]
			t = f[DEPEND_0][...]
			idxl = bisect.bisect_left(t,parser.parse(trange[0]))
			idxr = bisect.bisect_left(t,parser.parse(trange[1]))
			t = t[idxl:idxr]
			dist = f[cdfname][idxl:idxr,...]
			dist = np.transpose(dist,[0,3,1,2])
			ph = f[DEPEND_1][...]
			th = f[DEPEND_2][...]
			en = f[DEPEND_3][idxl:idxr,...]
			res = ts_skymap(t,dist,en,ph,th)
	return res

