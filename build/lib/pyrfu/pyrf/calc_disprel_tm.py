#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""calc_disprel_tm.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from scipy import optimize


def calc_disprel_tm(vel, vel_err, tau, tau_err):
    """Computes dispersion relation from velocities and period given by
    the timing method.

    Parameters
    ----------
    vel : xarray.DataArray
        Time series of the velocities.

    vel_err : xarray.DataArray
        Time series of the error on velocities.

    tau : xarray.DataArray
        Time series of the periods.

    tau_err : xarray.DataArray
        Time series of the error on period.

    Returns
    -------
    out : xarray.Dataset
        DataSet containing the frequency, the wavelength, the wavenumber.
        Also includes the errors and the fit (e.g Vph phase velocity).

    See also
    --------
    pyrfu.pyrf.c_4_v : Calculates velocity or time shift of discontinuity.

    """

    # Frequency, wavelength, wave number
    omega = 2 * np.pi / tau.data
    lamb, k = [vel * tau.data, 2 * np.pi / (vel * tau.data)]

    # Estimate propagation of the errors
    # Error on frequency
    omega_err = omega*((tau_err / tau) / (1 + tau_err / tau))

    # Error on wavelength
    lamb_err = vel_err * tau

    # Error on wave number
    k_err = k*((lamb_err/lamb)/(1+lamb_err/lamb))

    def model_tau_v(period, numerator):
        return numerator / period

    curve_fit_options = dict(sigma=np.sqrt(vel_err ** 2 + tau_err ** 2))
    res_tau = optimize.curve_fit(model_tau_v, tau, vel, 1, **curve_fit_options)
    fit_tau_v, cov_tau_v = res_tau[:2]
    sigma_tau_v = np.sqrt(np.diagonal(cov_tau_v))

    # High resolution prediction
    hires_tau = np.logspace(np.log10(5), np.log10(2e3), int(1e4))
    predict_v = model_tau_v(hires_tau, *fit_tau_v)

    # 95% confidence interval
    bound_upper_v = model_tau_v(hires_tau, *(fit_tau_v + 1.96*sigma_tau_v))
    bound_lower_v = model_tau_v(hires_tau, *(fit_tau_v - 1.96*sigma_tau_v))

    def model_k_w(wavenumber, factor):
        return factor * wavenumber

    curve_fit_options = dict(sigma=np.sqrt(omega_err ** 2 + k_err ** 2))
    res = optimize.curve_fit(model_k_w, k, omega, 1, **curve_fit_options)
    fit, cov = res[:2]
    sigma_k_w = np.sqrt(np.diagonal(cov))

    # High resolution prediction
    hires_k = np.linspace(0, 0.003, int(1e4))
    predict_w = model_k_w(hires_k, *fit)

    # 95% confidence interval
    bound_upper_w = model_k_w(hires_k, *(fit + 1.96 * sigma_k_w))
    bound_lower_w = model_k_w(hires_k, *(fit - 1.96 * sigma_k_w))

    out_dict = {'tau': tau, 'tau_err': (["tau"], tau_err),
                'v': (["tau"], vel), 'v_err': (["tau"], vel_err),
                'lamb': (["tau"], lamb), 'lamb_err': (["tau"], lamb_err),
                'k': k, 'k_err': (["k"], k_err),
                'omega': (["k"], omega), 'omega_err': (["k"], omega_err),
                'hires_k': hires_k, 'pred_omega': (["hires_k"], predict_w),
                'bound_upper': (["hires_k"], bound_upper_w),
                'bound_lower': (["hires_k"], bound_lower_w),
                'hires_tau': hires_tau, 'pred_v': (["hires_tau"], predict_v),
                'bound_upper_v': (["hires_tau"], bound_upper_v),
                'bound_lower_v': (["hires_tau"], bound_lower_v),
                'l': fit_tau_v, 'vph': fit, 'sigma_k_w': sigma_k_w}

    out = xr.Dataset(out_dict)

    return out
