import xarray as xr
import numpy as np



def norm(inp=None):
	"""
	Normalizes the input field

	Parameter :
		- inp               [xarray]                Input field

	Returns :
		- out               [xarray]                Normalized input field
	"""
	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")
	
	out = inp/np.linalg.norm(inp,axis=1,keepdims=True)
	return out