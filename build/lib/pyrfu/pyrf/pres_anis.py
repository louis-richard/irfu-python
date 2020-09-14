#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pres_anis.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from astropy import constants

from .resample import resample
from ..mms import rotate_tensor


def pres_anis(p_xyz=None, b_xyz=None):
	"""
	Compute pressure anisotropy factor: (P_para - P_perp) * mu0 / B^2

	Parameters :
		p_xyz : DataArray
			Time series of the pressure tensor
		b_xyz : DataArray
			Time series of the background magnetic field

	Returns :
		p_anis : DataArray
			Time series of the pressure anisotropy

	See also :
		rotate_tensor

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load magnetic field, ion/electron temperature and number density
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> p_xyz_i = mms.get_data("Pi_gse_fpi_fast_l2", tint, mms_id)
		>>> # Compute pressure anistropy
		>>> p_anis = pyrf.pres_anis(p_xyz_i, b_xyz)

	"""

	if (p_xyz is None) or (b_xyz is None):
		raise ValueError("pres_anis requires at least 2 arguments")

	if not isinstance(p_xyz, xr.DataArray):
		raise TypeError("p_xyz must be a DataArray")

	if not isinstance(b_xyz, xr.DataArray):
		raise TypeError("b_xyz must be a DataArray")

	b_xyz = resample(b_xyz, p_xyz)

	# rotate pressure tensor to field aligned coordinates
	p_xyzfac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")
	
	# Get parallel and perpendicular pressure
	p_para = p_xyzfac[:, 0, 0]
	p_perp = (p_xyzfac[:, 1, 1] + p_xyzfac[:, 2, 2]) / 2
	
	# Load permitivity
	mu0 = constants.mu0.value

	# Compute pressure anistropy
	p_anis = 1e9 * mu0 * (p_para - p_perp) / np.linalg.norm(b_xyz) ** 2
	
	return p_anis
