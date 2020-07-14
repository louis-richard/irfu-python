import xarray as xr
import numpy as np


def norm(inp=None):
	"""
	Computes the magnitude of the input field

	Parameters :
		- inp : DataArray
			Time series of the input field

	Returns :
		- out : DataArray
			Time series of the magnitude of the input field

	Example : 
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
		>>> Bmag = pyrf.norm(Bxyz)

	"""

	if type(inp) != xr.DataArray:
		raise TypeError('Input must be a DataArray')

	out = np.sqrt(np.sum(inp**2,axis=1))
	return out