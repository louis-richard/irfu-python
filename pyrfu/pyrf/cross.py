import xarray as xr
import numpy as np

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def cross(inp1=None,inp2=None):
	"""
	Computes cross product of two fields z = xxy

	Parameters :
		- inp1              [xarray]                Time serie of the first field x
		- inp2              [xarray]                Time serie of the second field y

	Returns :
		- out               [xarray]                Cross product

	"""
	if not isinstance(inp1,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	if not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
		
	if len(inp1) != len(inp2):
		inp2 = resample(inp2,inp1)
		
	outdata = np.cross(inp1.data,inp2.data,axis=1)

	out = ts_vec_xyz(inp1.time.data,outdata)
	
	return out


