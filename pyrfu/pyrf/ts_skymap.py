import numpy as np
import xarray as xr


def ts_skymap(time,data,energy,phi,theta,**kwargs):
	"""
	Creates a skymap of the distribution function

	Parameters :
		time : np.ndarray
			List of times

		data : np.ndarray
			Values of the distribution function

		energy : np.ndarray
			Energy levels

		phi : np.ndarray
			Azimuthal angles
			
		theta : np.ndarray
			Elevation angles

	Returns :
		out : DataArray
			Skymap of the distribution function

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