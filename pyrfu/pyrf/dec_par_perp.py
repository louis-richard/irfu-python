#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dec_par_perp.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .dot import dot


def dec_par_perp(inp=None, b0=None, flag_spin_plane=False):
	"""
	Decomposes a vector into par/perp to B components. If flagspinplane decomposes components to the projection of B 
	into the XY plane. Alpha_XY gives the angle between B0 and the XY plane.
	
	Parameters :
		inp : DataArray
			Time series of the field to decompose

		b0 : DataArray
			Time series of the background magnetic field
	
	Options :
		flagspinplane : bool
			Flag if True gives the projection in XY plane
		
	Returns :
		apara : DataArray
			Time series of the input field parallel to the background magnetic field

		aperp : DataArray
			Time series of the input field perpendicular to the background magnetic field

		alpha : DataArray
			Time series of the angle between the background magnetic field and the XY plane

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load magnetic field (FGM) and electric field (EDP)
		>>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
		>>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)
		>>> # Decompose e_xyz into parallel and perpendicular to b_xyz components
		>>> e_para, e_perp, _ = pyrf.dec_par_perp(e_xyz, b_xyz)

	"""

	if (inp is None) or (b0 is None):
		raise ValueError("dec_par_perp requires at least 2 arguments")
	
	if not isinstance(inp, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not isinstance(b0, xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not flag_spin_plane:
		b_mag = np.linalg.norm(b0, axis=1, keepdims=True)

		ii = np.where(b_mag < 1e-3)[0]

		if ii.size > 0:
			b_mag[ii] = np.ones(len(ii))*1e-3

		b_hat = b0 / b_mag
		b_hat = resample(b_hat, inp)

		a_para = dot(b_hat, inp)
		a_perp = inp.data - (b_hat * np.tile(a_para.data, (3, 1)).T)
		alpha = []
	else:
		b0 = resample(b0, inp)
		bt = np.sqrt(b0[:, 0] ** 2 + b0[:, 1] ** 2)
		b0 /= bt[:, np.newaxis]

		a_para = inp[:, 0] * b0[:, 0] + inp[:, 1] * b0[:, 1]
		a_perp = inp[:, 0] * b0[:, 1] - inp[:, 1] * b0[:, 0]
		alpha = np.arctan2(b0[:, 2], bt)

	return a_para, a_perp, alpha