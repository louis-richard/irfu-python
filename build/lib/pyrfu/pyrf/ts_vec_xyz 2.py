#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts_vec_xyz.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def ts_vec_xyz(t=None, data=None, attrs=None):
	"""
	Create a time series containing a 1st order tensor

	Parameters :
		t : np.ndarray
			Array of times  
		data : np.ndarray
			Data corresponding to the time list
	
	Options :
		attrs : dict
			Attributes of the data list

	Returns :
		out : DataArray
			1st order tensor time series

	"""

	# Check inputs are not empty
	if t is None:
		raise ValueError("ts_vec_xyz requires at least two inputs")
	if data is None:
		raise ValueError("ts_vec_xyz requires at least two inputs")
		
	# Check inputs are numpy arrays
	if not isinstance(t, np.ndarray):
		raise TypeError("Time must be a np.datetime64 array")
	
	if not isinstance(data, np.ndarray):
		raise TypeError("Data must be a np array")
	
	if data.ndim != 2:
		raise TypeError("Input must be a vector field")
	
	if data.shape[1] != 3:
		raise TypeError("Input must be a xyz vector field")
	
	if len(t) != len(data):
		raise IndexError("Time and data must have the same length")
	
	flag_attrs = True

	if attrs is None:
		flag_attrs = False
	
	out = xr.DataArray(data, coords=[t[:], ["x", "y", "z"]], dims=["time", "comp"])
	
	if flag_attrs:
		out.attrs = attrs

		out.attrs["TENSOR_ORDER"] = 1
	else:
		out.attrs["TENSOR_ORDER"] = 1

	return out
