import numpy as np
import xarray as xr


def calc_dt(inp=None):
	"""
	Compute time step of the input time series

	Parameters :
		- inp               [xarray]                Input time series

	Returns :
		- out               [float]                 Time step in seconds

	"""

	if inp is None:
		raise ValueError("calc_dt requires at least one argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")

	out = np.median(np.diff(inp.time.data)).astype(float)*1e-9
	return out