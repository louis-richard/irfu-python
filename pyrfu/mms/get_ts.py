import os
import bisect
import numpy as np
import xarray as xr
from spacepy import pycdf
from dateutil import parser




def get_ts(file_path="",cdfname="",trange=None):
	"""
	Read field named cdfname in file and convert to time serie

	Parameters :
		- file_path         [str]                   Path of the cdf file
		- cdfname           [str]                   Name of the target variable in the cdf file
		- tramge            [list]                  Time interval

	Returns :
		- out               [xarray]                Time serie of the target variable in the selected time interval

	"""
	if not file_path or not cdfname or trange is None:
		raise ValueError("get_ts requires at least 3 arguments")
	
	if not isinstance(file_path,str):
		raise TypeError("file_path must be a str in UNIX format")
	
	if not isinstance(cdfname,str):
		raise TypeError("cdfname must be a str")
	
	x = {}
	y = {}
	z = {}
	w = {}
	outdict = {}
	with pycdf.CDF(file_path) as f:
		depend0_key = f[cdfname].attrs["DEPEND_0"]
		start_ind   = bisect.bisect_left(f[depend0_key], parser.parse(trange[0]))
		stop_ind    = bisect.bisect_left(f[depend0_key], parser.parse(trange[1]))
		x["data"]   = f[depend0_key][start_ind:stop_ind]
		x["attrs"]  = {}
		for k in f[depend0_key].attrs.keys():
			x["attrs"][k] = f[depend0_key].attrs[k]
			if isinstance(x["attrs"][k],str) and x["attrs"][k] in f.keys() and not k == "LABLAXIS":
				try :
					# If array
					x["attrs"][k] = f[x["attrs"][k]][start_ind:stop_ind,...]
				except IndexError:
					# If scalar
					x["attrs"][k] = f[x["attrs"][k]][...]

		
		if "DEPEND_1" in f[cdfname].attrs or "REPRESENTATION_1" in f[cdfname].attrs:
			try :
				depend1_key = f[cdfname].attrs["DEPEND_1"]
			except KeyError:
				depend1_key = f[cdfname].attrs["REPRESENTATION_1"]

			#pdb.set_trace()

			if depend1_key == "x,y,z":
				y["data"]   = np.array(depend1_key.split(","))
				y["attrs"]  = {"LABLAXIS": "comp"}
			else :
				try :
					y["data"] = f[depend1_key][start_ind:stop_ind,:]
				except IndexError:
					y["data"] = f[depend1_key][...]

				# If vector componenents remove magnitude index
				
				if len(y["data"]) == 4 and all(y["data"] == ['x', 'y', 'z', 'r']):
					y["data"] = y["data"][:-1]
				# if y is 2d get only first row assuming that the bins are the same
				elif y["data"].ndim == 2:
					try :  
						y["data"] = y["data"][0,:]
					except IndexError :
						pass


				y["attrs"] = {}
				# Get attributes
				for k in f[depend1_key].attrs.keys():
					y["attrs"][k] = f[depend1_key].attrs[k]

					if isinstance(y["attrs"][k],str) and y["attrs"][k] in f.keys():
						if not k in ["DEPEND_0","LABLAXIS"]:
							try :
								y["attrs"][k] = f[y["attrs"][k]][start_ind:stop_ind,...]
							except :
								y["attrs"][k] = f[y["attrs"][k]][...]
							# If attrs is 2D get only first row
							if y["attrs"][k].ndim == 2:
								try :
									y["attrs"][k] = y["attrs"][k][0,:]
								except IndexError:
									pass

				# Remove spaces in label
				try:
					y["attrs"]["LABLAXIS"] = y["attrs"]["LABLAXIS"].replace(" ","_")
				except KeyError:
					y["attrs"]["LABLAXIS"] = "comp"


		if "DEPEND_2" in f[cdfname].attrs or "REPRESENTATION_2" in f[cdfname].attrs:
			try :
				depend2_key = f[cdfname].attrs["DEPEND_2"]
			except KeyError:
				depend2_key = f[cdfname].attrs["REPRESENTATION_2"]

			if depend2_key == "x,y,z":
				z["data"]   = np.array(depend2_key.split(","))
				z["attrs"]  = {"LABLAXIS": "comp"}
			else :
				z["data"]   = f[depend2_key][...]
				z["attrs"]  = {}
				for k in f[depend2_key].attrs.keys():
					z["attrs"][k] = f[depend2_key].attrs[k]

					if isinstance(z["attrs"][k],str) and z["attrs"][k] in f.keys() and not k == "DEPEND_0":
						z["attrs"][k] = f[z["attrs"][k]][start_ind:stop_ind,...]

				if not "LABLAXIS" in z["attrs"].keys():
					z["attrs"]["LABLAXIS"] = "comp"

		if ("DEPEND_3" in f[cdfname].attrs 
			or "REPRESENTATION_3" in f[cdfname].attrs 
			and f[cdfname].attrs["REPRESENTATION_3"] != "x,y,z"):

			try :
				depend3_key = f[cdfname].attrs["DEPEND_3"]
			except KeyError:
				depend3_key = f[cdfname].attrs["REPRESENTATION_3"]

			w["data"] = f[depend3_key][...]

			if w["data"].ndim == 2:
				try :  
					w["data"] = w["data"][0,:]
				except IndexError :
					pass
			
			w["attrs"] = {}
			for k in f[depend3_key].attrs.keys():
				w["attrs"][k] = f[depend3_key].attrs[k]

				if isinstance(w["attrs"][k],str) and w["attrs"][k] in f.keys() and not k == "DEPEND_0":
					w["attrs"][k] = f[w["attrs"][k]][start_ind:stop_ind,...]

			if not "LABLAXIS" in w["attrs"].keys():
				w["attrs"]["LABLAXIS"] = "comp"


		if "sector_mask" in cdfname:
			y["data"]   = f[f[cdfname.replace("sector_mask","intensity")].attrs["DEPEND_1"]][...]
			y["attrs"]  = {}
			for k in f[f[cdfname.replace("sector_mask","intensity")].attrs["DEPEND_1"]].attrs.keys():
				y["attrs"][k] = f[f[cdfname.replace("sector_mask","intensity")].attrs["DEPEND_1"]].attrs[k]

			y["attrs"]["LABLAXIS"] = y["attrs"]["LABLAXIS"].replace(" ","_")


		if "edp_dce_sensor" in cdfname:
			y["data"] = ["x","y","z"]
			y["attrs"] = {"LABLAXIS": "comp"}
			
		outdict["data"] = f[cdfname][start_ind:stop_ind,...]

		if outdict["data"].ndim == 2 and outdict["data"].shape[1] == 4:
			outdict["data"] = outdict["data"][:,:-1]

		outdict["attrs"] = {}

		for k in f[cdfname].attrs:
			outdict["attrs"][k] = f[cdfname].attrs[k]


	if x and not y and not z and not w:
		dims    = ["time"]
		coords  = [x["data"]]
		out     = xr.DataArray(outdict["data"],coords=coords,dims=dims,attrs=outdict["attrs"])
		exec("out."+dims[0]+".attrs = x['attrs']")
	elif x and y and not z and not w:
		dims    = ["time",y["attrs"]["LABLAXIS"]]
		coords  = [x["data"],y["data"]]
		out     = xr.DataArray(outdict["data"],coords=coords,dims=dims,attrs=outdict["attrs"])
		exec("out."+dims[0]+".attrs = x['attrs']")
		exec("out."+dims[1]+".attrs = y['attrs']")
	elif x and y and z and not w:
		dims    = ["time",y["attrs"]["LABLAXIS"],z["attrs"]["LABLAXIS"]]
		coords  = [x["data"],y["data"],z["data"]]
		out     = xr.DataArray(outdict["data"],coords=coords,dims=dims,attrs=outdict["attrs"])
		exec("out."+dims[0]+".attrs = x['attrs']")
		exec("out."+dims[1]+".attrs = y['attrs']")
		exec("out."+dims[2]+".attrs = z['attrs']")
	elif x and y and z and w:
		dims    = ["time",y["attrs"]["LABLAXIS"],z["attrs"]["LABLAXIS"],w["attrs"]["LABLAXIS"]]
		coords  = [x["data"],y["data"],z["data"],w["data"]]
		out     = xr.DataArray(outdict["data"],coords=coords,dims=dims,attrs=outdict["attrs"])
		exec("out."+dims[0]+".attrs = x['attrs']")
		exec("out."+dims[1]+".attrs = y['attrs']")
		exec("out."+dims[2]+".attrs = z['attrs']")
		exec("out."+dims[3]+".attrs = w['attrs']")
		
	return out