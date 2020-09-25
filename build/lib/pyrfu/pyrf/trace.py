#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trace.py
@author : Louis RICHARD
"""

import xarray as xr

from .ts_scalar import ts_scalar


def trace(inp=None):
	"""
	Computes trace of the time series of 2nd order tensors

	Parameters :
		inp : DataArray
			Time series of the input 2nd order tensor.

	Returns :
		out : DataArray
			Time series of the trace of the input tensor

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load magnetic field and ion temperature
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)
		>>> # Rotate to ion temperature tensor to field aligned coordinates
		>>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")
		>>> # Compute scalar temperature
		>>> t_i = pyrf.trace(t_xyzfac_i)

	"""

	if not isinstance(inp, xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if inp.data.ndim != 3:
		raise TypeError("Input must be a 2nd order tensor")

	inp_data = inp.data
	out_data = inp_data[:, 0, 0] + inp_data[:, 1, 1] + inp_data[:, 2, 2]
	
	# Attributes
	attrs = inp.attrs

	# Change tensor order from 2 (matrix) to 0 (scalar)
	attrs["TENSOR_ORDER"] = 0

	out = ts_scalar(inp.time.data, out_data, attrs)

	return out
