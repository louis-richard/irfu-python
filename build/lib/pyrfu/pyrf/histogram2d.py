#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
histogram2d.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample


def histogram2d(inp1=None, inp2=None, bins=100):
	"""
	Computes 2d histogram of inp2 vs inp1 with nbins number of bins

	Parameters :
		inp1 : DataArray
			Time series of the x values

		inp2 : DataArray
			Time series of the y values
		
		nbins : int
			Number of bins

	Returns :
		out : DataArray
			2D map of the density of inp2 vs inp1

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> mms_id = 1
		>>> # Load magnetic field and electric field
		>>> b = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> r = mms.get_data("R_gse", tint, mms_id)
		>>> # Compute current density, etc
		>>> j_xyz, div_b, b_xyz, jxb, div_t_shear, div_pb = pyrf.c_4_j(r, b)
		>>> # Compute magnitude of B and J
		>>> b_mag = pyrf.norm(b_xyz)
		>>> j_mag = pyrf.norm(j_xyz)
		>>> # Histogram of |J| vs |B|
		>>> h2d_b_j = pyrf.histogram2d(b_mag, j_mag)

	"""

	if inp1 is None or inp2 is None:
		raise ValueError("histogram2d requires at least 2 arguments")

	if not isinstance(inp1, xr.DataArray) or not isinstance(inp2, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	# resample inp2 with respect to inp1
	if len(inp2) != len(inp1):
		inp2 = resample(inp2, inp1)

	h2d, x_edges, y_edges = np.histogram2d(inp1.data, inp2.data, bins=bins)

	x, y = [x_edges[:-1] + np.median(np.diff(x_edges)) / 2, y_edges[:-1] + np.median(np.diff(y_edges)) / 2]

	out = xr.DataArray(h2d, coords=[x, y], dims=["x_bins", "y_bins"])

	return out
