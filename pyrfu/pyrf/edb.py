#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
edb.py

@author : Louis RICHARD
"""

import numpy as np

from .resample import resample
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz


def edb(e=None, b0=None, angle_lim=20, flag_method="E.B=0"):
	"""
	Compute Ez under assumption E.B=0 or E.B~=0

	Parameters :
		e : DataArray
			Time series of the electric field

		b0 : DataArray
			Time series of the background magnetic field

		flag_method : str
			Assumption on the direction of the measured electric field :
				"E.B=0" -> E.B = 0
				"Epar" 	-> E field along the B projection is coming from parallel electric field

		angle_lim : float
			B angle with respect to the spin plane should be less than angle_lim degrees otherwise Ez is set to 0
	
	Returns :
		ed : DataArray
			Time series of the electric field output

		d : DataArray
			Time series of the B elevation angle above spin plane

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> mms_id = 1
		>>> # Load magnetic field, electric field and spacecraft position
		>>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
		>>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)
		>>> # Compute Ez
		>>> ed, d = pyrf.edb(e_xyz, b_xyz)

	"""

	if e is None:
		raise ValueError("edb requires at least two inputs")
	if b0 is None:
		raise ValueError("edb requires at least two inputs")
	
	default_value = 0
	if flag_method.lower() == "eperp+nan":
		default_value = np.nan

		flag_method = "e.b=0"

	if len(b0) != len(e):
		b0 = resample(b0, e)

	bd = b0.data
	ed = e.data
	ed[:, -1] *= default_value

	if flag_method.lower() == "e.b=0":
		# Calculate using assumption E.B=0
		d = np.arctan2(bd[:, 2], np.sqrt(bd[:, 0] ** 2 + bd[:, 1] ** 2)) * 180 / np.pi

		ind = np.abs(d) > angle_lim

		if True in ind:
			ed[ind, 2] = -(ed[ind, 0] * bd[ind, 0] + ed[ind, 1] * bd[ind, 1]) / bd[ind, 2]

	elif flag_method.lower() == "epar":
		# Calculate using assumption that E field along the B projection is coming from parallel electric field
		d = np.arctan2(bd[:, 2], np.sqrt(bd[:, 0] ** 2 + bd[:, 1] ** 2)) * 180 / np.pi

		ind = np.abs(d) < angle_lim

		if True in ind:
			ed[ind, 2] = (ed[ind, 0] * bd[ind, 0] + ed[ind, 1] * bd[ind, 1])
			ed[ind, 2] = ed[ind, 2] * bd[ind, 2] / (bd[ind, 0] ** 2 + bd[ind, 1] ** 2)

	else:
		raise ValueError("Invalid flag")

	ed, d = [ts_vec_xyz(e.time.data, ed, {"UNITS": e.attrs["UNITS"]}), ts_scalar(e.time.data, d, {"UNITS": "degres"})]

	return ed, d
