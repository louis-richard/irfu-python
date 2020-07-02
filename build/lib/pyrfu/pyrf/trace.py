import numpy as np
import xarray as xr



def trace(inp=None):
	"""
	Computes trace of the time serie of 2nd order tensors

	Paramters :
		- inp               [xarray]                Input time serie of 2nd order tensor

	Returns :
		- out               [xarray]                Trace of the input tensor

	"""

	if not isinstance(inp, xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if len(inp.shape) != 3:
		raise TypeError("Input must be a 2nd order tensor")

	inp_data    = inp.data
	outdata     = inp_data[:,0,0] + inp_data[:,1,1] + inp_data[:,2,2]
	
	# Attributes
	attrs                   = inp.attrs
	attrs["TENSOR_ORDER"]   = 0

	out = xr.DataArray(outdata,coords=[inp.time],dims=["time"],attrs=attrs)
	
	return out