#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_4_k.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .cross import cross
from .dot import dot


def c_4_k(r_list=None):
	"""
	Calculates reciprocal vectors in barycentric coordinates.
	
	Parameters
	----------
	r_list : list of xarray.DataArray
		Position of the spacecrafts

	Returns
	-------
	k_list : list of xarray.DataArray
		Reciprocal vectors in barycentric coordinates

	Reference
	---------
	ISSI book  Eq. 14.16, 14.17 p. 353

	Note
	----
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
		cc = cross(r_list[k]-r_list[j], r_list[l]-r_list[j])

		dr12 = r_list[i]-r_list[j]

		k_list[i] = cc/dot(cc, dr12)

	return k_list
