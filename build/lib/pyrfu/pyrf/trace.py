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

	Example :
		>>> # Time interval
		>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
		>>> 
		>>> # Spacecraft index
		>>> ic = 1
		>>> 
		>>> # Load magnetic field and ion temperature
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
		>>> Tixyz = mms.get_data("Ti_gse_fpi_fast_l2",Tint,ic)
		>>> 
		>>> # Rotate to ion temperature tensor to field aligned coordinates
		>>> Tixyzfac = pyrf.rotate_tensor(Tixyz,"fac",Bxyz,"pp")
		>>> 
		>>> # Compute scalar temperature
		>>> Ti = pyrf.trace(Tixyzfac)

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