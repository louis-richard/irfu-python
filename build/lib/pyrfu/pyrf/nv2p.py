import numpy as np
import xarray as xr

from .ts_scalar import ts_scalar


def nv2p(n=None, v=None):
	"""
	Calculate plasma dynamic pressure

	Parameters :
		- n                 [xarray]                Number density
		- v                 [xarray]                Velocity field
	
	Returs :
		- out               [xarray]                Dynamic pressure

	"""

	if n is None or v is None:
		raise ValueError("nv2p requires at least 2 arguments")
	
	if not isinstance(n,xr.DataArray) or not isinstance(v,xr.DataArray):
		raise TypeError("n and v must be DataArrays")


	outdata = 1.6726*1e-6*n*np.linalg.norm(v,axis=1)**2

	out = ts_scalar(n.time.data,outdata)

	return out