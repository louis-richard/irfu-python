import numpy as np
import xarray as xr

from astropy.time import Time



def ts_time(t=None,fmt="unix"):
	"""
	Creates time line in DataArray

	Parameters :
		t : array
			Input time line

	Options :
		fmt : str
			Format of the input time line

	Returns :
		out : DataArray
			Time series of the time line

	"""

	if t is None:
		raise ValueError("ts_time requires at least one argument")

	if not isinstance(t,np.ndarray):
		raise TypeError("t must be an array")

	t = Time(t,format=fmt).datetime64
	
	out = xr.DataArray(t,coords=[t],dims=["time"])
	
	return out
