#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
normalize.py

@author : Louis RICHARD
"""

import xarray as xr
import numpy as np


def normalize(inp=None):
	"""
	Normalizes the input field

	Parameter :
		inp : DataArray
			Time series of the input field

	Returns :
		out : DataArray
			Time series of the normalized input field
	
	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load magnetic field
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> # Compute the normalized magnetic field
		>>> b = pyrf.normalize(b_xyz)

	"""

	if inp is None:
		raise ValueError("normalize requires one argument")

	if not isinstance(inp, xr.DataArray):
		raise TypeError("Input must be a DataArray")
	
	out = inp / np.linalg.norm(inp, axis=1, keepdims=True)

	return out