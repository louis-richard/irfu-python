# -*- coding: utf-8 -*-
"""
calc_disprel_tm.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from scipy import optimize


def calc_disprel_tm(v=None, dv=None, tau=None, dtau=None):
	"""
	Computes dispersion relation from velocities and period given by the timing method

	Parameters :
		V : DataArray
			Time series of the velocities

		dV : DataArray
			Time series of the error on velocities

		T : DataArray
			Time series of the periods
		
		dT : DataArray
			Time series of the error on period

	Returns :
		out : Dataset
			DataSet containing the frequency, the wavelength, the wavenumber. Also includes the errors and the fit 
			(e.g Vph phase velocity)

	See also :
		c_4_v_xcorr

	"""

	if (v is None) or (dv is None) or (tau is None) or (dtau is None):
		raise ValueError("calc_disprel_tm requires at least 4 arguments")

	if not isinstance(v, xr.DataArray):
		raise TypeError("V must a DataArray")

	if not isinstance(dv, xr.DataArray):
		raise TypeError("dV must a DataArray")

	if not isinstance(tau, xr.DataArray):
		raise TypeError("T must a DataArray")

	if not isinstance(dtau, xr.DataArray):
		raise TypeError("dT must a DataArray")

	omega, lamb, k = [2 * np.pi / tau, v * tau, 2 * np.pi/(v * tau)]  # Frequency, wavelength, wave number

	# Estimate propagation of the errors
	# Error on frequency
	domega = omega*((dtau / tau) / (1 + dtau / tau))

	# Error on wavelength
	dlamb = dv * tau

	# Error on wave number
	dk = k*((dlamb/lamb)/(1+dlamb/lamb))

	def model_tau_v(x, a):
		return a / x

	fit_tau_v, cov_tau_v = optimize.curve_fit(model_tau_v, tau, v, 1, sigma=np.sqrt(dv ** 2 + dtau ** 2))
	sigma_tau_v = np.sqrt(np.diagonal(cov_tau_v))
	hires_tau = np.logspace(np.log10(5), np.log10(2e3), int(1e4))
	bound_upper_v = model_tau_v(hires_tau, *(fit_tau_v + 1.96*sigma_tau_v))
	bound_lower_v = model_tau_v(hires_tau, *(fit_tau_v - 1.96*sigma_tau_v))
	pred_v = model_tau_v(hires_tau, *fit_tau_v)

	def model_k_w(x, a):
		return a * x

	fit, cov = optimize.curve_fit(model_k_w, k, omega, 1, sigma=np.sqrt(domega ** 2 + dk ** 2))
	sigma_k_w = np.sqrt(np.diagonal(cov))
	hires_k = np.linspace(0, 0.003, int(1e4))
	bound_upper_w = model_k_w(hires_k, *(fit + 1.96 * sigma_k_w))
	bound_lower_w = model_k_w(hires_k, *(fit - 1.96 * sigma_k_w))
	pred_w = model_k_w(hires_k, *fit)

	outdict = {"T": tau, "dT": (["T"], dtau), "V": (["T"], v), "dV": (["T"], dv), "lamb": (["T"], lamb),
			   "dlamb": (["T"], dlamb), "k": k, "dk": (["k"], dk), "omega": (["k"], omega), "domega": (["k"], domega),
			   "hires_k": hires_k, "pred_omega": (["hires_k"], pred_w), "bound_upper": (["hires_k"], bound_upper_w),
			   "bound_lower": (["hires_k"], bound_lower_w), "hires_tau": hires_tau, "pred_v": (["hires_tau"], pred_v),
			   "bound_upper_v": (["hires_tau"], bound_upper_v), "bound_lower_v": (["hires_tau"], bound_lower_v),
			   "l": fit_tau_v, "Vph": fit}

	out = xr.Dataset(outdict)

	return out
