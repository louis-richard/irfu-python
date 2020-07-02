import xarray as xr
from astropy.time import Time



def end(inp=None,fmt="unix"):
	"""
	Gives the last time of the time series

	Parameters :
		- inp               [xarray]                Time series
		- fmt               [str]                   Format of the output

	Returns :
		- out               [float/str]             Value of the first time in the desired format
	"""

	if not isinstance(inp,xr.DataArray):
		raise TypeError("in must be a time series")

	out = Time(inp.time.data[-1],format="datetime64").unix
	return out 