import numpy as np
import xarray as xr



def trace(inp=None):
	"""
	Computes trace of the time series of 2nd order tensors

	Parameters :
		inp : DataArray
			Time series of the input 2nd order tensor.

	Returns :
		out : DataArray
			Time series of the trace of the input tensor

	"""

	if not isinstance(inp, xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if inp.data.ndim != 3:
		raise TypeError("Input must be a 2nd order tensor")

	inp_data    = inp.data
	outdata     = inp_data[:,0,0] + inp_data[:,1,1] + inp_data[:,2,2]
	
	# Attributes
	attrs                   = inp.attrs
	attrs["TENSOR_ORDER"]   = 0

	out = xr.DataArray(outdata,coords=[inp.time],dims=["time"],attrs=attrs)
	
	return out