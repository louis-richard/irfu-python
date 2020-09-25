#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fk_powerspec_4sc.py

@author : Louis RICHARD
"""

import numpy as np
import bisect
import xarray as xr

from ..pyrf.resample import resample
from ..pyrf.avg_4sc import avg_4sc
from ..pyrf.time_clip import time_clip
from ..pyrf.wavelet import wavelet


def fk_powerspec_4sc(e=None, r=None, b=None, tints=None, cav=8, num_k=500, num_f=200, df=None, w_width=1, f_range=None):
	"""
	Calculates the frequency-wave number power spectrum using the four MMS spacecraft. Uses a generalization of 
	mms.fk_powerspectrum. Wavelet based cross-spectral analysis is used to calculate the phase difference each 
	spacecraft pair and determine 3D wave vector. A generalization of the method used in mms.fk_powerspectrum 
	to four point measurements. 

	Parameters :
		e : list of DataArray
			Fields to apply 4SC cross-spectral analysis to. e.g., e or b fields 
			(if multiple components only the first is used).

		r : list of DataArray
			Positions of the four spacecraft

		b : list of DataArray
			background magnetic field in the same coordinates as r. Used to determine the parallel and perpendicular 
			wave numbers using 4SC average.

		tints : list of str
			Time interval over which the power spectrum is calculated. To avoid boundary effects use a longer time
			interval for e and b. 
		
		cav : int
			(Optional) Number of points in time series used to estimate phase. (default cav = 8)

		num_k : int
			(Optional) Number of wave numbers used in spectrogram. (default num_k = 500)

		df : float
			(Optional) Linear spacing of frequencies (default log spacing).

		num_f : int
			(Optional) Number of frequencies used in spectrogram. (default num_f = 200)

		w_width : float
			(Optional) Multiplier for Morlet wavelet width. (default w_width = 1)

		f_range : list of float
			(Optional) Frequency range for k-k plots. [minf maxf]

	returns :
		out : Dataset
			Dataset of array of powers as a function of frequency and wavenumber. Power is normalized to the maximum 
			value.

	Notes: 
		Wavelength must be larger than twice the spacecraft separations, otherwise spatial aliasing will occur. 

	Example:
		>>> from pyrfu import mms
		>>> power = mms.fk_powerspec_4sc(e_par, r_xyz, b_xyz, tints)
		>>> power = mms.fk_powerspec_4sc(b_scmfac_x, r_xyz, b_xyz, tints, linear=10, num_k=500, cav=4, w_width=2)

	Example to plot:
		>>> import matplotlib.pyplot as plt
		>>> from pyrfu.plot import plot_spectr
		>>> fig, ax = plt.subplots(1)
		>>> ax, cax = plot_spectr(ax, power.kmagf, cscale="log", cmap="viridis")
		>>> ax.set_xlabel("$|k|$ [m$^{-1}$]")
		>>> ax.set_ylabel("$f$ [Hz]")

	"""

	if (e is None) or (r is None) or (b is None) or (tints is None):
		raise ValueError("fk_powerspec4SC requires at least 4 arguments")

	ic = np.arange(1, 5)

	e = [resample(e[i - 1], e[0]) for i in ic]
	r = [resample(r[i - 1], e[0]) for i in ic]
	b = [resample(b[i - 1], e[0]) for i in ic]

	b_avg = avg_4sc(b)

	times = e[0].time
	use_linear = not(df is None)

	idx = time_clip(e[0].time, tints)

	# If odd, remove last data point (as is done in irf_wavelet)
	if len(idx) % 2:
		idx = idx[:-1]

	if use_linear:
		w = [wavelet(e[i], linear=df, returnpower=False, wavelet_width=5.36 * w_width) for i in range(4)]
	else:
		w = [wavelet(e[i], nf=num_f, returnpower=False, wavelet_width=5.36 * w_width) for i in range(4)]

	num_f = len(w[0].frequency)

	times = time_clip(times, tints)
	nt = len(times)

	w = [time_clip(w[i], tints) for i in range(4)]

	fk_power = 0
	for i in range(4):
		fk_power += w[i].data * np.conj(w[i].data) / 4

	n = int(np.floor(nt/cav)-1)
	posav = cav / 2 + np.arange(n) * cav
	avtimes = times[posav.astype(int)]

	b_avg = resample(b_avg, avtimes)

	r = [resample(r[i], avtimes) for i in range(4)]

	cx12, cx13, cx14, cx23, cx24, cx34 = [np.zeros((n, num_f), dtype="complex128") for _ in range(6)]

	power_avg = np.zeros((n, num_f), dtype="complex128")

	for m, posavm in enumerate(posav):
		lb, ub = [int(posavm - cav / 2 + 1), int(posavm + cav / 2)]

		cx12[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[1].data[lb:ub, :]), axis=0)
		cx13[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[2].data[lb:ub, :]), axis=0)
		cx14[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)
		cx23[m, :] = np.nanmean(w[1].data[lb:ub, :] * np.conj(w[2].data[lb:ub, :]), axis=0)
		cx24[m, :] = np.nanmean(w[1].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)
		cx34[m, :] = np.nanmean(w[2].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)

		power_avg[m, :] = np.nanmean(fk_power[lb:ub, :], axis=0)

	# Compute phase differences between each spacecraft pair
	th12 = np.arctan2(np.imag(cx12), np.real(cx12))
	th13 = np.arctan2(np.imag(cx13), np.real(cx13))
	th14 = np.arctan2(np.imag(cx14), np.real(cx14))
	th23 = np.arctan2(np.imag(cx23), np.real(cx23))
	th24 = np.arctan2(np.imag(cx24), np.real(cx24))
	th34 = np.arctan2(np.imag(cx34), np.real(cx34))

	w_mat = 2 * np.pi * np.tile(w[0].frequency.data, (n, 1))

	# Convert phase difference to time delay
	dt12, dt13, dt14, dt23, dt24, dt34 = [th / w_mat for th in [th12, th13, th14, th23, th24, th34]]

	# Weighted averaged time delay using all spacecraft pairs
	dt2 = 0.5 * dt12 + 0.2 * (dt13 - dt23) + 0.2 * (dt14 - dt24) + 0.1 * (dt14 - dt34 - dt23)
	dt3 = 0.5 * dt13 + 0.2 * (dt12 + dt23) + 0.2 * (dt14 - dt34) + 0.1 * (dt12 + dt24 - dt34)
	dt4 = 0.5 * dt14 + 0.2 * (dt12 + dt24) + 0.2 * (dt13 + dt34) + 0.1 * (dt12 + dt23 + dt34)

	# Compute phase speeds
	r = [r[i].data for i in range(4)]

	k_x, k_y, k_z = [np.zeros((n, num_f)) for _ in range(3)]

	# Volumetric tensor with SC1 as center.
	dr = np.reshape(np.hstack(r[1:]), (n, 3, 3)) - np.reshape(np.tile(r[0], (1, 3)), (n, 3, 3))
	dr = np.transpose(dr, [0, 2, 1])

	# Delay tensor with SC1 as center.
	# dT = np.reshape(np.hstack([dt2,dt3,dt4]),(N,num_f,3))
	tau = np.dstack([dt2, dt3, dt4])

	for ii in range(num_f):
		m = np.linalg.solve(dr, np.squeeze(tau[:, ii, :]))

		k_x[:, ii], k_y[:, ii], k_z[:, ii] = [2 * np.pi * w[0].frequency[ii].data * m[:, i] for i in range(3)]

	k_x, k_y, k_z = [k / 1e3 for k in [k_x, k_y, k_z]]

	k_mag = np.linalg.norm(np.array([k_x, k_y, k_z]), axis=0)

	b_avg_x_mat = np.tile(b_avg.data[:, 0], (num_f, 1)).T
	b_avg_y_mat = np.tile(b_avg.data[:, 1], (num_f, 1)).T
	b_avg_z_mat = np.tile(b_avg.data[:, 2], (num_f, 1)).T

	b_avg_abs = np.linalg.norm(b_avg, axis=1)
	b_avg_abs_mat = np.tile(b_avg_abs, (num_f, 1)).T

	k_par = (k_x * b_avg_x_mat + k_y * b_avg_y_mat + k_z * b_avg_z_mat) / b_avg_abs_mat
	k_perp = np.sqrt(k_mag ** 2 - k_par ** 2)

	k_max = np.max(k_mag) * 1.1
	k_min = -k_max
	k_vec = np.linspace(-k_max, k_max, num_k)
	k_mag_vec = np.linspace(0, k_max, num_k)

	dkmag = k_max / num_k
	dk = 2 * k_max / num_k

	# Sort power into frequency and wave vector
	print("notice : Computing power versus kx,f; ky,f, kz,f")
	power_k_x_f, power_k_y_f, power_k_z_f = [np.zeros((num_f, num_k)) for _ in range(3)]
	power_k_mag_f = np.zeros((num_f, num_k))

	for nn in range(num_f):
		k_x_number = np.floor((k_x[:, nn] - k_min) / dk).astype(int)
		k_y_number = np.floor((k_y[:, nn] - k_min) / dk).astype(int)
		k_z_number = np.floor((k_z[:, nn] - k_min) / dk).astype(int)
		k_number = np.floor((k_mag[:, nn]) / dkmag).astype(int)

		power_k_x_f[nn, k_x_number] += np.real(power_avg[:, nn])
		power_k_y_f[nn, k_y_number] += np.real(power_avg[:, nn])
		power_k_z_f[nn, k_z_number] += np.real(power_avg[:, nn])

		power_k_mag_f[nn, k_number] += np.real(power_avg[:, nn])

	# powerkxf[powerkxf == 0] 	= np.nan
	# powerkyf[powerkyf == 0] 	= np.nan
	# powerkzf[powerkzf == 0] 	= np.nan
	# powerkmagf[powerkmagf == 0] = np.nan

	power_k_x_f /= np.max(power_k_x_f)
	power_k_y_f /= np.max(power_k_y_f)
	power_k_z_f /= np.max(power_k_z_f)
	power_k_mag_f /= np.max(power_k_mag_f)

	# powerkxf[powerkxf < 1.0e-6] 		= 1e-6
	# powerkyf[powerkyf < 1.0e-6] 		= 1e-6
	# powerkzf[powerkzf < 1.0e-6] 		= 1e-6
	# powerkmagf[powerkmagf < 1.0e-6] 	= 1e-6

	freqs = w[0].frequency.data
	idxf = np.arange(num_f)

	if f_range is not None:
		idx_minfreq, idx_maxfreq = [bisect.bisect_left(np.min(f_range)), bisect.bisect_left(np.max(f_range))]

		idxf = idxf[idx_minfreq:idx_maxfreq]

	print("notice : Computing power versus kx,ky; kx,kz; ky,kz\n")
	power_k_x_k_y = np.zeros((num_k, num_k))
	power_k_x_k_z = np.zeros((num_k, num_k))
	power_k_y_k_z = np.zeros((num_k, num_k))
	power_k_perp_k_par = np.zeros((num_k, num_k))

	for nn in idxf:
		k_x_number = np.floor((k_x[:, nn] - k_min) / dk).astype(int)
		k_y_number = np.floor((k_y[:, nn] - k_min) / dk).astype(int)
		k_z_number = np.floor((k_z[:, nn] - k_min) / dk).astype(int)

		k_par_number = np.floor((k_par[:, nn] - k_min) / dk).astype(int)
		k_perp_number = np.floor((k_perp[:, nn]) / dkmag).astype(int)

		power_k_x_k_y[k_y_number, k_x_number] += np.real(power_avg[:, nn])
		power_k_x_k_z[k_z_number, k_x_number] += np.real(power_avg[:, nn])
		power_k_y_k_z[k_z_number, k_y_number] += np.real(power_avg[:, nn])

		power_k_perp_k_par[k_par_number, k_perp_number] += np.real(power_avg[:, nn])

	# powerkxky[powerkxky == 0] = np.nan
	# powerkxkz[powerkxkz == 0] = np.nan
	# powerkykz[powerkykz == 0] = np.nan
	# powerkperpkpar[powerkperpkpar == 0] = np.nan

	power_k_x_k_y /= np.max(power_k_x_k_y)
	power_k_x_k_z /= np.max(power_k_x_k_z)
	power_k_y_k_z /= np.max(power_k_y_k_z)
	power_k_perp_k_par /= np.max(power_k_perp_k_par)

	# powerkxky(powerkxky < 1.0e-6) 				= 1e-6
	# powerkxkz(powerkxkz < 1.0e-6) 				= 1e-6
	# powerkykz(powerkykz < 1.0e-6) 				= 1e-6
	# powerkperpkpar[powerkperpkpar < 1.0e-6] 	= 1e-6

	outdict = {"kxf": (["kx", "f"], power_k_x_f.T), "kyf": (["kx", "f"], power_k_y_f.T),
			   "kzf": (["kx", "f"], power_k_z_f.T), "kmagf": (["kmag", "f"], power_k_mag_f.T),
			   "kxky": (["kx", "ky"], power_k_x_k_y.T), "kxkz": (["kx", "kz"], power_k_x_k_z.T),
			   "kykz": (["ky", "kz"], power_k_y_k_z.T), "kperpkpar": (["kperp", "kpar"], power_k_perp_k_par.T),
			   "kx": k_vec, "ky": k_vec, "kz": k_vec, "kmag": k_mag_vec, "kperp": k_mag_vec, "kpar": k_vec, "f": freqs}

	out = xr.Dataset(outdict)

	return out
