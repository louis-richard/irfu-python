#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dynamic_press.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from astropy import constants


def dynamic_press(n=None, v=None, s="i"):
	"""
	Computes dynamic pressure

	Parameters :
		n : DataArray
			Time series of the number density of the specie
		v : DataArray
			Time series of the bulk velocity of the specie
	
	Options :
		s : "i"/"e"
			Specie (default "i")
	
	Returns :
		Pdyn : DataArray
			Time series of the dynamic pressure of the specie

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load ion bulk velocity and remove spintone
		>>> v_xyz_i = mms.get_data("Vi_gse_fpi_fast_l2", tint, mms_id)
		>>> st_xyz_i = mms.get_data("STi_gse_fpi_fast_l2", tint, mms_id)
		>>> v_xyz_i = v_xyz_i - st_xyz_i
		>>> # Ion number density
		>>> n_i = mms.get_data("Ni_fpi_fast_l2", tint, mms_id)
		>>> # Compute dynamic pressure
		>>> p = pyrf.dynamic_press(n_i, v_xyz_i, s="i")
	"""

	if (n is None) or (v is None):
		raise ValueError("dynamic_press requires at least 2 arguments")

	if not isinstance(n, xr.DataArray):
		raise TypeError("n must be a DataArray")

	if not isinstance(v, xr.DataArray):
		raise TypeError("v must be a DataArray")

	if s == "i":
		m = constants.m_p.value
	elif s == "e":
		m = constants.m_e.value
	else:
		raise ValueError("Unknown specie")

	p_dyn = n * m * np.linalg.norm(v, axis=0) ** 2

	return p_dyn
