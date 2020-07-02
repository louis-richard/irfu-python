import xarray as xr
import numpy as np




def new_xyz(inp=None,M=None):
	"""
	Transform the input field to the new frame

	Paramters:
		- inp               [xarray]                Input field
		- M                 [ndarray]               Transformation matrix

	Returns :
		- out               [xarray]                Input in the new frame

	"""

	if inp is None:
		raise ValueError("new_xyz equires at least two argument")
	
	if M is None:
		raise ValueError("new_xyz equires at least two argument")
		
	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")
	
	if not isinstance(M,np.ndarray):
		raise TypeError("M must be a ndarray")
		
	if inp.data.ndim == 3:
		outdata = np.matmul(np.matmul(M.T,inp.data),M)
	else :
		outdata = (M.T @ inp.data.T).T
	
	out = xr.DataArray(outdata,coords=inp.coords,dims=inp.dims,attrs=inp.attrs)
	
	return out