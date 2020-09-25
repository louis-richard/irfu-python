#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_disprel_tm.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from scipy import optimize


def calc_disprel_tm(v=None, v_err=None, tau=None, tau_err=None):
	"""
	Computes dispersion relation from velocities and period given by the timing method

	Parameters :
		v : DataArray
			Time series of the velocities

		v_err : DataArray
			Time series of the error on velocities

		tau : DataArray
			Time series of the periods
		
		tau_err : DataArray
			Time series of the error on period

	Returns :
		out : Dataset
			DataSet containing the frequency, the wavelength, the wavenumber. Also includes the errors and the fit 
			(e.g Vph phase velocity)

	See also :
		c_4_v_xcorr

	"""

	if (v is None) or (v_err is None) or (tau is None) or (tau_err is None):
		raise ValueError("calc_disprel_tm requires at least 4 arguments")

	if not isinstance(v, xr.DataArray):
		raise TypeError("V must a DataArray")

	if not isinstance(v_err, xr.DataArray):
		raise TypeError("dV must a DataArray")

	if not isinstance(tau, xr.DataArray):
		raise TypeError("T must a DataArray")

	if not isinstance(tau_err, xr.DataArray):
		raise TypeError("dT must a DataArray")

	omega, lamb, k = [2 * np.pi / tau.data, v * tau.data, 2 * np.pi/(v * tau.data)]  # Frequency, wavelength, wave number

	# Estimate propagation of the errors
	# Error on frequency
	omega_err = omega*((tau_err / tau) / (1 + tau_err / tau))

	# Error on wavelength
	lamb_err = v_err * tau

	# Error on wave number
	k_err = k*((lamb_err/lamb)/(1+lamb_err/lamb))

	def model_tau_v(x, a):
		return a / x

	fit_tau_v, cov_tau_v = optimize.curve_fit(model_tau_v, tau, v, 1, sigma=np.sqrt(v_err ** 2 + tau_err ** 2))
	sigma_tau_v = np.sqrt(np.diagonal(cov_tau_v))

	# High resolution prediction
	hires_tau = np.logspace(np.log10(5), np.log10(2e3), int(1e4))
	predict_v = model_tau_v(hires_tau, *fit_tau_v)

	# 95% confidence interval
	bound_upper_v = model_tau_v(hires_tau, *(fit_tau_v + 1.96*sigma_tau_v))
	bound_lower_v = model_tau_v(hires_tau, *(fit_tau_v - 1.96*sigma_tau_v))

	def model_k_w(x, a):
		return a * x

	fit, cov = optimize.curve_fit(model_k_w, k, omega, 1, sigma=np.sqrt(omega_err ** 2 + k_err ** 2))
	sigma_k_w = np.sqrt(np.diagonal(cov))

	# High resolution prediction
	hires_k = np.linspace(0, 0.003, int(1e4))
	predict_w = model_k_w(hires_k, *fit)

	# 95% confidence interval
	bound_upper_w = model_k_w(hires_k, *(fit + 1.96 * sigma_k_w))
	bound_lower_w = model_k_w(hires_k, *(fit - 1.96 * sigma_k_w))

	out_dict = {'tau': tau, 'tau_err': (["tau"], tau_err), 'v': (["tau"], v), 'v_err': (["tau"], v_err),
				'lamb': (["tau"], lamb),
				'lamb_err': (["tau"], lamb_err), 'k': k, 'k_err': (["k"], k_err), 'omega': (["k"], omega),
				'omega_err': (["k"], omega_err), 'hires_k': hires_k, 'pred_omega': (["hires_k"], predict_w),
				'bound_upper': (["hires_k"], bound_upper_w), 'bound_lower': (["hires_k"], bound_lower_w),
				'hires_tau': hires_tau, 'pred_v': (["hires_tau"], predict_v),
				'bound_upper_v': (["hires_tau"], bound_upper_v), 'bound_lower_v': (["hires_tau"], bound_lower_v),
				'l': fit_tau_v, 'vph': fit, 'sigma_k_w': sigma_k_w}

	out = xr.Dataset(out_dict)

	return out
