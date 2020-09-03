# -*- coding: utf-8 -*-
"""
ts_skymap.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def ts_skymap(t=None, data=None, energy=None, phi=None, theta=None, **kwargs):
	"""
	Creates a skymap of the distribution function

	Parameters :
		time : np.ndarray
			List of times

		data : np.ndarray
			Values of the distribution function

		energy : np.ndarray
			Energy levels

		phi : np.ndarray
			Azimuthal angles

		theta : np.ndarray
			Elevation angles

	Returns :
		out : DataArray
			Skymap of the distribution function

	"""

	if t is None or data is None or phi is None or theta is None:
		raise ValueError("ts_skymap requires at least 4 arguments")

	if not isinstance(t, np.ndarray):
		raise TypeError("time must be an array")

	if energy is None:
		if "energy0" in kwargs:
			energy0, energy0_ok = [kwargs["energy0"], True]
		else:
			energy0, energy0_ok = [None, False]

		if "energy1" in kwargs:
			energy1, energy1_ok = [kwargs["energy1"], True]
		else:
			energy1, energy1_ok = [None, False]

		if "esteptable" in kwargs:
			esteptable, esteptable_ok = [kwargs["esteptable"], True]
		else:
			esteptable, esteptable_ok = [None, False]

		if not energy0_ok and not energy1_ok and not esteptable_ok:
			raise ValueError("Energy input required")

		energy = np.tile(energy0, (len(esteptable), 1))

		energy[esteptable == 1] = np.tile(energy1, (int(np.sum(esteptable)), 1))

	else:
		energy0, energy1, esteptable = [None] * 3

		energy0_ok, energy1_ok, esteptable_ok = [False] * 3

	mydict = {"data": (["time", "idx0", "idx1", "idx2"], data), "phi": (["time", "idx1"], phi),
			  "theta": (["idx2"], theta), "energy": (["time", "idx0"], energy), "time": t, "idx0": np.arange(32),
			  "idx1": np.arange(32), "idx2": np.arange(16)}

	out = xr.Dataset(mydict)

	if energy0_ok:
		out.attrs["energy0"] = energy0

	if energy1_ok:
		out.attrs["energy1"] = energy1

	if energy0_ok:
		out.attrs["esteptable"] = esteptable

	return out
