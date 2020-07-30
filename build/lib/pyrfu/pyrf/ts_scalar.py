import numpy as np
import xarray as xr



def ts_scalar(t=None,data=None,attrs=None):
	"""
	Create a time series containing a 0th order tensor

	Parameters :
		t : np.ndarray
			Array of times

		data : np.ndarray
			Data corresponding to the time list
	
	Options :
		attrs : dict
			Attributes of the data list

	Returns :
		out DataArray
			0th order tensor time series
	"""

	# Check inputs are not empty
	if t is None:
		raise ValueError("ts_scalar requires at least two inputs")
	if data is None:
		raise ValueError("ts_scalar requires at least two inputs")
		
	# Check inputs are numpy arrays
	if not isinstance(t,np.ndarray):
		raise TypeError("Time must be a np.datetime64 array")
	
	if not isinstance(data,np.ndarray):
		raise TypeError("Data must be a np array")
	
	if data.ndim != 1:
		raise TypeError("Input must be a scalar")
	
	if len(t) != len(data):
		raise IndexError("Time and data must have the same length")
	flagAttrs = False
	if attrs != None:
		flagAttrs = True
	
	out = xr.DataArray(data,coords=[t],dims="time")
	
	if flagAttrs :
		out.attrs = attrs
	out.attrs["TENSOR_ORDER"] = 0
	return out