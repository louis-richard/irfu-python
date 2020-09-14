#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_4_v.py

@author : Louis RICHARD
"""

import numpy as np
from scipy import interpolate
from astropy.time import Time


def get_vol_ten(r=None, t=None):

	if len(t) == 1:
		t = np.array([t, t, t, t])

	tckr_x, tckr_y, tckr_z = [[], [], []]

	for i in range(4):
		tckr_x.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 0]))
		tckr_y.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 1]))
		tckr_z.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 2]))

		r[i] = np.array([tckr_x[i](t[0]), tckr_y[i](t[0]), tckr_z[i](t[0])])

	# Volumetric tensor with SC1 as center.
	dr_mat = (np.vstack(r[1:]) - np.tile(r[0], (3, 1))).T

	return dr_mat


def c_4_v(r=None, x=None):
	"""
	Calculate velocity or time shift of discontinuity.

	Parameters : 
		r : list of DataArray
			Time series of the positions of the spacecraft
		x : list
			Crossing times or time and velocity
	
	Returns :
		out : ndarray
			Discontinuity velocity or time shift with respect to mms1
	"""
	if isinstance(x, np.ndarray) and x.dtype == np.datetime64:
		flag = "v_from_t"

		x = Time(x, format="datetime64").unix
	elif x[1] > 299792.458:
		flag = "v_from_t"
	else:
		flag = "dt_from_v"

	if flag.lower() == "v_from_t":
		# Time input, velocity output
		t = x
		dr_mat = get_vol_ten(r, t)
		tau = np.array(t[1:]) - t[0]
		m = np.linalg.solve(dr_mat, tau)

		# "1/v vector"
		out = m / np.linalg.norm(m) ** 2

	elif flag.lower() == "dt_from_v":
		# Time and velocity input, time output
		tc = x[0]  # center time
		v = np.array(x[1:])  # Input velocity
		m = v / np.linalg.norm(v) ** 2

		dr_mat = get_vol_ten(r, tc)
		
		dt = np.matmul(dr_mat, m)
		out = np.hstack([0, dt])

	else:
		raise ValueError("Invalid flag")

	return out
