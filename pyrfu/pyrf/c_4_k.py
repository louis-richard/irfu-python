import numpy as np
import xarray as xr

from .cross import cross
from .dot import dot


def c_4_k(r_list=None):
	"""
	Calculate reciprocal vectors in barycentric coordinates. Reference: ISSI book 14.7
	
	Parameters :
		R : list of DataArray
			Position of the spacecrafts

	Returns :
		K : list of DataArray
			Reciprocal vectors in barycentric coordinates
	
	Note : 
		The units of reciprocal vectors are the same as [1/r]

	"""

	if r_list is None:
		raise ValueError("c_4_k requires one argument")

	if not isinstance(r_list, list) or len(r_list) != 4:
		raise TypeError("R must be a list of 4SC position")

	for i in range(4):
		if not isinstance(r_list[i], xr.DataArray):
			raise TypeError("Spacecraft position must be DataArray")

	mms_list = np.arange(4)

	k_list = [None]*4

	for i, j, k, l in zip(mms_list, np.roll(mms_list, 1), np.roll(mms_list, 2), np.roll(mms_list, 3)):
		cc 			= cross(r_list[k]-r_list[j], r_list[l]-r_list[j])
		dr12 		= r_list[i]-r_list[j]
		k_list[j] 	= cc/dot(cc, dr12)

	return k_list
