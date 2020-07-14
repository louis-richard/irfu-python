import xarray as xr
import numpy as np


def normalize(inp=None):
	"""
	Normalizes the input field

	Parameter :
		- inp : DataArray
			Time series of the input field

	Returns :
		- out : DataArray
			Time series of the normalized input field
	
	Example :
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> gseB = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
		>>> b = pyrf.normalize(gseB)

	"""

	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")
	
	out = inp/np.linalg.norm(inp,axis=1,keepdims=True)

	return out