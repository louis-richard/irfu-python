#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
end.py

@author : Louis RICHARD
"""

import xarray as xr
from astropy.time import Time


def end(inp=None):
	"""
	Gives the last time of the time series

	Parameters :
		inp : DataArray
			Time series of the input variable

		fmt : str
			Format of the output time (see Rots et al. 2015 https://arxiv.org/pdf/1409.7583.pdf)

	Returns :
		out : float/str
			Value of the last time in the desired format
	"""

	if inp is None:
		raise ValueError("end requires at least one argument")

	if not isinstance(inp, xr.DataArray):
		raise TypeError("in must be a time series")

	out = Time(inp.time.data[-1], format="datetime64").unix

	return out
