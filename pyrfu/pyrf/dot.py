#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dot.py

@author : Louis RICHARD
"""

import xarray as xr
import numpy as np

from .resample import resample
from .ts_scalar import ts_scalar


def dot(x=None, y=None):
	"""
	Computes dot product of two fields

	Parameters : 
		inp1 : DataArray
			Time series of the first field X

		inp2 : DataArray
			Time series of the second field Y

	Returns :
		out : DataArray
			Time series of the dot product Z = X.Y

	Example :
		>>> import numpy as np
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> mms_list = np.arange(1,5)
		>>> # Load magnetic field, electric field and spacecraft position
		>>> b_mms = [mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id) for mms_id in mms_list]
		>>> e_mms = [mms.get_data("E_gse_edp_fast_l2", tint, mms_id) for mms_id in mms_list]
		>>> r_mms = [mms.get_data("R_gse", tint, mms_id) for mms_id in mms_list]
		>>> j_xyz, div_b, b_avg, jxb, div_t_shear, div_pb = pyrf.c_4_j(r_mms, b_mms)
		>>> # Compute the electric at the center of mass of the tetrahedron
		>>> e_xyz = pyrf.avg_4sc(e_mms)
		>>> # Compute J.E dissipation
		>>> je = pyrf.dot(j_xyz, e_xyz)

	"""

	if (x is None) or (y is None):
		raise ValueError("dot requires 2 arguments")

	if not isinstance(x, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not isinstance(y, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	y = resample(y, x)

	outdata = np.sum(x.data * y.data, axis=1)

	out = ts_scalar(x.time.data, outdata)
	
	return out
