import numpy as np
import xarray as xr



def gradient(inp=None):
	"""
	Computes time derivative of the input time serie

	Parameters :
		- inp               [xarray]                Input time serie

	Returns :
		- out               [xarray]                Time derivative of the input time serie

	"""


	if inp is None:
		raise ValueError("gradient requires at least 1 argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")


	# guess time step
	dt = np.median(np.diff(inp.time.data)).astype(float)*1e-9

	dinpdt = np.gradient(inp.data,axis=0)/dt

	out = xr.DataArray(dinpdt,coords=inp.coords,dims=inp.dims, attrs=inp.attrs)
	

	if "UNITS" in out.attrs:
		out.attrs["UNITS"] += "/s"

	return out