import xarray as xr
import numpy as np

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def cross(x=None, y=None):
	"""
	Computes cross product of two fields.

	Parameters :
		x : DataArray
			Time series of the first field X

		y : DataArray
			Time series of the second field Y

	Returns :
		out : DataArray
			Time series of the cross product Z = XxY

	Example :
		>>> from pyrfu import mms, pyrf
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, 1)
		>>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, 1)
		>>> b_mag = pyrf.norm(b_xyz)
		>>> ExBxyz = pyrf.cross(e_xyz, b_xyz)/b_mag ** 2
	"""

	if (x is None) or (y is None):
		raise ValueError("cross requires 2 arguments")

	if not isinstance(x, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	if not isinstance(y, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
		
	if len(x) != len(y):
		y = resample(y, x)
		
	outdata = np.cross(x.data, y.data, axis=1)

	out = ts_vec_xyz(x.time.data, outdata)
	
	return out
