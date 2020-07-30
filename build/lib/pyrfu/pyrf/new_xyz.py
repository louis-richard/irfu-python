import xarray as xr
import numpy as np




def new_xyz(inp=None,M=None):
	"""
	Transform the input field to the new frame

	Parameters:
		inp : DataArray
			Time series of the input field in the original coordinate system

		M : array
			Transformation matrix

	Returns :
		out : DataArray
			Time series of the input in the new frame

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = 1
		>>> # Load magnetic field and electric field
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
		>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,ic)
		>>> # Compute MVA frame
		>>> Blmn, l, V = pyrf.minvar(Bxyz)
		>>> # Move electric field to the MVA frame
		>>> Elmn = pyrf.new_xyz(Exyz,V)

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