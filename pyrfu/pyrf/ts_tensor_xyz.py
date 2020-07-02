import numpy as np
import xarray as xr



def ts_tensor_xyz(t,data,attrs=None):
	"""
	Create a time serie containing a 2nd order tensor

	Parameters :
		- t                 [ndarray]               Array of times  
		- data              [ndarray]               Data corresponding to the time list
		- attrs             [dict]                  Attributes of the data list (optionnal)

	Returns :
		- out               [xarray]                2nd order tensor time serie
	"""

	# Check inputs are not empty
	if t is None:
		raise ValueError("ts_tensor_xyz requires at least two inputs")
	if data is None:
		raise ValueError("ts_tensor_xyz requires at least two inputs")
		
	# Check inputs are numpy arrays
	if not isinstance(t,np.ndarray):
		raise TypeError("Time must be a np.datetime64 array")
	
	if not isinstance(data,np.ndarray):
		raise TypeError("Data must be a np array")
	
	if data.ndim != 3:
		raise TypeError("Input must be a tensor time serie")
	
	if data.shape[1] != 3 or data.shape[2] != 3:
		raise TypeError("Input must be a xyz tensor")
	
	if len(t) != len(data):
		raise IndexError("Time and data must have the same length")
	
	flagAttrs = False
	if attrs != None:
		flagAttrs = True
	
	out = xr.DataArray(data,coords=[t,["x","y","z"],["x","y","z"]],dims=["time","comph","compv"])
	
	if flagAttrs :
		out.attrs = attrs

	out.attrs["TENSOR_ORDER"] = 2

	return out