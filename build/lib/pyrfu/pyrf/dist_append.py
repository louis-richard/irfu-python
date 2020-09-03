# -*- coding: utf-8 -*-
"""
dist_append.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from .ts_skymap import ts_skymap


def dist_append(inp0=None, inp1=None):
	"""
	Concatenate two distribution skymaps along the time axis

	Note : the time series have to be in the correct time order

	Parameters :
		inp1 : DataArray
			3D skymap distribution at early times 

		inp2 : DataArray
			3D skymap distribution at late times 

	Returns :
		out : DataArray
			3D skymap of the concatenated 3D skymaps 

	"""

	if inp0 is None or inp1 is None:
		raise ValueError("dist_append requires at least two arguments")
		
	if not isinstance(inp0, xr.Dataset) or not isinstance(inp1, xr.Dataset):
		raise TypeError("inputs must be datasets")
		
	n_t0 = len(inp0.data)
	n_t1 = len(inp1.data)

	n_en = inp0.energy.shape[1]
	try:
		n_ph = inp0.phi.shape[1]
	except IndexError:
		n_ph = len(inp0.phi)

	n_th = len(inp0.theta)

	# time
	time = np.zeros(n_t0 + n_t1)
	time[:n_t0] = inp0.time.data
	time[n_t0:n_t0 + n_t1] = inp1.time.data
	time[n_t0:n_t0 + n_t1] = inp1.time.data

	# Azymutal angle
	if inp0.phi.ndim == 2:
		# TODO replace by stacking phi of inp0 and inp1
		phi = np.zeros((n_t0 + n_t1, n_ph))
		phi[:n_t0, ...] = inp0.phi.data
		phi[n_t0:n_t0 + n_t1, ...] = inp1.phi.data
	else:
		phi = inp0.phi.data

	# Elevation angle
	theta = inp0.theta.data

	# distribution
	# TODO replace by stacking data of inp0 and inp1
	data = np.zeros((n_t0 + n_t1, n_en, n_ph, n_th))
	data[:n_t0, ...] = inp0.data
	data[n_t0:n_t0 + n_t1, ...] = inp1.data

	if "delta_energy_plus" in inp0.attrs:
		# TODO replace by stacking data of inp0 and inp1
		delta_energy_plus = np.zeros((n_t0 + n_t1, n_en))
		delta_energy_plus[:n_t0, ...] = inp0.attrs["delta_energy_plus"].data
		delta_energy_plus[n_t0:n_t0 + n_t1, ...] = inp1.attrs["delta_energy_plus"].data
	else:
		delta_energy_plus = None

	if "delta_energy_minus" in inp0.attrs:
		# TODO replace by stacking data of inp0 and inp1
		delta_energy_minus = np.zeros((n_t0 + n_t1, n_en))
		delta_energy_minus[:n_t0, ...] = inp0.attrs["delta_energy_minus"].data
		delta_energy_minus[n_t0:n_t0 + n_t1, ...] = inp1.attrs["delta_energy_minus"].data
	else:
		delta_energy_minus = None

	# Energy
	if inp0.attrs["tmmode"] == "brst":
		step_table = np.hstack([inp0.attrs["esteptable"], inp1.attrs["esteptable"]])
		out = ts_skymap(time, data, None, phi, theta, energy0=inp0.energy0, energy1=inp0.energy1, esteptable=step_table)
	else:
		# TODO replace by stacking data of inp0 and inp1
		energy = np.zeros((n_t0 + n_t1, n_en))
		energy[:n_t0, ...] = inp0.energy.data
		energy[n_t0:n_t0 + n_t1, ...] = inp1.energy.data
		out = ts_skymap(time, data, energy, phi, theta)
	
	# attributes
	attrs = inp0.attrs
	attrs.pop("esteptable")

	attrs["delta_energy_minus"] = delta_energy_minus
	attrs["delta_energy_plus"] = delta_energy_plus

	for k in attrs:
		out.attrs[k] = attrs[k]

	return out
