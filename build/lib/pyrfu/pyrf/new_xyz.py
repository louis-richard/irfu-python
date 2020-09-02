import xarray as xr
import numpy as np


def new_xyz(inp=None, trans_mat=None):
	"""
	Transform the input field to the new frame

	Parameters:
		inp : DataArray
			Time series of the input field in the original coordinate system

		trans_mat : array
			Transformation matrix

	Returns :
		out : DataArray
			Time series of the input in the new frame

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> mms_id = 1
		>>> # Load magnetic field and electric field
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)
		>>> # Compute MVA frame
		>>> b_lmn, l, mva = pyrf.minvar(b_xyz)
		>>> # Move electric field to the MVA frame
		>>> e_lmn = pyrf.new_xyz(e_xyz, mva)

	"""

	if inp is None:
		raise ValueError("new_xyz requires at least two argument")
	
	if trans_mat is None:
		raise ValueError("new_xyz requires at least two argument")
		
	if not isinstance(inp, xr.DataArray):
		raise TypeError("inp must be a DataArray")
	
	if not isinstance(trans_mat, np.ndarray):
		raise TypeError("M must be a ndarray")
		
	if inp.data.ndim == 3:
		out_data = np.matmul(np.matmul(trans_mat.T, inp.data), trans_mat)
	else:
		out_data = (trans_mat.T @ inp.data.T).T
	
	out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims, attrs=inp.attrs)
	
	return out
