#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t_eval.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
import bisect


def t_eval(inp=None, t=None):
	"""Evaluates the input time series at the target time.

	Parameters
	----------
	inp : xarray.DataArray
		Time series if the input to evaluate.

	t : numpy.ndarray
		Times at which the input will be evaluated.

	Returns
	-------
	out : DataArray
		Time series of the input at times t.

	"""

	if (inp is None) or (t is None):
		raise ValueError("t_eval requires at least 2 arguments")

	idx = np.zeros(len(t))

	for i in range(len(t)):
		idx[i] = bisect.bisect_left(inp.time.data, t[i])
		
	idx = idx.astype(int)

	if inp.ndim == 2:
		out = xr.DataArray(inp.data[idx, :], coords=[t, inp.comp], dims=["time", "comp"])
	else:
		out = xr.DataArray(inp.data[idx], coords=t, dims=["time"])

	return out
