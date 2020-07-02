import numpy as np
import xarray as xr


def ts_skymap(time,data,energy,phi,theta,**kwargs):
	"""
	Createes a skymap of the distribution function

	Parameters :
		- time              [ndarray]               List of times
		- data              [ndarray]               Values of the distribution function
		- energy            [ndarray]               Energy levels 
		- phi               [ndarray]               Azymutal angles
		- theta             [ndarray]               Elevation angles

	Returns :
		- out               [xarray]                Skymap of the distribution function

	"""
	
	if not isinstance(time,np.ndarray):
		raise TypeError("time must be an array")
	#if time.dtype != "<M8[ns]":
	#    raise TypeError("time must be datetime64 array")
	epoch = time
	


	if energy is None:
		energy0_ok = False
		energy1_ok = False
		esteptable_ok = False
		if "energy0" in kwargs:
			energy0_ok = True
			energy0 = kwargs["energy0"]
		if "energy1" in kwargs:
			energy1_ok = True
			energy1 = kwargs["energy1"]
		if "esteptable" in kwargs:
			esteptable_ok = True
			esteptable = kwargs["esteptable"]
		if not energy0_ok and not energy1_ok and not esteptable_ok:
			raise ValueError("Energy input required")
			
		energy = np.tile(energy0,(len(esteptable),1))
		energy[esteptable==1] = np.tile(energy1,(int(np.sum(esteptable)),1))

	mydict  = {"data": (["time","idx0","idx1","idx2"], data),\
				"phi":(["time","idx1"],phi),"theta":(["idx2"],theta),\
				"energy":(["time","idx0"],energy),"time":time,\
				"idx0":np.arange(32),"idx1":np.arange(32),"idx2":np.arange(16)}
	
	out = xr.Dataset(mydict)

	if energy0_ok:
		out.attrs["energy0"] = energy0

	if energy1_ok:
		out.attrs["energy1"] = energy1

	if energy0_ok:
		out.attrs["esteptable"] = esteptable
	return out