import xarray as xr
import numpy as np


def abs(inp=None):
	"""
	Computes the magnitude of the input field

	Parameters :
		- inp               [xarray]                Input field

	Returns :
		- out               [xarray]                Magnitude of the input field

	"""

	if type(inp) != xr.DataArray:
		raise TypeError('Input must be a DataArray')

	out = np.sqrt(np.sum(inp**2,axis=1))
	return out