import xarray as xr
import numpy as np

from .resample import resample
from .ts_scalar import ts_scalar



def dot(inp1=None,inp2=None):
	"""
	Computes do product of two fields

	Parameters : 
		- inp1              [xarray]                First field x
		- inp2              [xarray]                Second field y

	Returns :
		- out               [xarray]                Dot product of inp1 and inp2 z = x.y

	"""
	if not isinstance(inp1,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	if not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	inp2 = resample(inp2,inp1)

	outdata = np.sum(inp1.data*inp2.data,axis=1)

	out = ts_scalar(inp1.time.data,outdata)
	return out