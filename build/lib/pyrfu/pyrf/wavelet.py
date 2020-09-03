# -*- coding: utf-8 -*-
"""
wavelet.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
import pyfftw
import multiprocessing as mp
from astropy.time import Time

import matplotlib.pyplot as plt


def wavelet(inp=None, **kwargs):
	"""
	Calculate wavelet spectrogram based on fast FFT algorithm
	
	Parameters :
		inp : DataArray
			Input quantity

	Options :
		fs : int/float
			Sampling frequency of the input time series

		f : list/np.ndarray
			Vector [fmin fmax], calculate spectra between frequencies fmin and fmax

		nf : int/float
			Number of frequency bins

		wavelet_width : int/float
			Width of the Morlet wavelet, default 5.36

		linear : float
			Linear spacing between frequencies of df

		returnpower : bool
			Set to True (default) to return the power, False for complex wavelet transform

		cutedge : bool
			Set to True (default) to set points affected by edge effects to NaN, False to keep edge affect points
	
	Returns :
		out : DataArray/Dataset
			Wavelet transform of the input

	"""

	# Default values
	# Fs

	if isinstance(inp, xr.DataArray):
		# Time bounds
		tstart, tstop = [Time(t_bound, format="datetime64").unix for t_bound in [inp.time.data[0], inp.time.data[-1]]]

		# Time interval
		timeint = tstop - tstart

		# Sampling frequency
		fs = len(inp) / timeint

		# Unpack time and data
		t, data = [inp.time.data.view("i8") * 1e-9, inp.data]

	else:
		raise TypeError("Input must be a DataArray")

	# f
	amin, amax = [0.01, 2]
	fmin, fmax = [.5 * fs / 10 ** amax, .5 * fs / 10 ** amin]

	# nf
	nf = 200

	# wavelet_width
	wavelet_width, deltaf = [5.36, 100]

	return_power, cut_edge, linear_df, plot_flag = [True, True, False, False]

	if "returnpower" in kwargs:
		return_power = kwargs["returnpower"]

	if "cutedge" in kwargs:
		cut_edge = kwargs["cutedge"]

	if "fs" in kwargs:
		if isinstance(kwargs["fs"], (int, float)):
			fs = kwargs["fs"]
		else:
			raise TypeError("fs must be numeric")

	if "nf" in kwargs:
		if isinstance(kwargs["nf"], (int, float)):
			nf = kwargs["nf"]
		else:
			raise TypeError("nf must be numeric")

	if "linear" in kwargs:
		linear_df = True
		if isinstance(kwargs["linear"], (int, float)):
			deltaf = kwargs["linear"]
		else:
			raise Warning("Unknow input for linear deltaf set to 100")

	if "wavelet_width" in kwargs:
		if isinstance(kwargs["wavelet_width"], (int, float)):
			wavelet_width = kwargs["wavelet_width"]
		else:
			raise TypeError("wavelet_width must be numeric")

	if "f" in kwargs:
		if isinstance(kwargs["f"], (np.ndarray, list)):
			if len(kwargs["f"]) == 2:
				fmin = kwargs["f"][0]
				fmax = kwargs["f"][1]
			else:
				raise IndexError("f should have vector with 2 elements as parameter value")
		else:
			raise TypeError("f must be a list or array")

	if "plot" in kwargs:
		plot_flag = kwargs["plot"]

	sampl, w0, anumber, sigma = [fs, fs / 2, nf, wavelet_width / (fs / 2)]

	if linear_df:
		anumber = np.floor(w0 / deltaf).astype(int)

		fmin, fmax = [deltaf, anumber * deltaf]

		a = w0 / (np.linspace(fmax, fmin, anumber))
	else:
		amin, amax = [np.log10(.5 * fs / fmax), np.log10(.5 * fs / fmin)]

		a = np.logspace(amin, amax, anumber)

	# Remove the last sample if the total number of samples is odd
	if len(data) / 2 != np.floor(len(data) / 2):
		t, data = [t[:-1], data[:-1, ...]]

	# Check for NaNs
	a[np.isnan(a)] = 0

	# Find the frequencies for an FFT of all data
	nd2, nyq = [len(data) / 2, 1 / 2]

	freq = sampl * np.arange(1, nd2 + 1) / nd2 * nyq

	# The frequencies corresponding to FFT
	w = np.hstack([0, freq, -np.flip(freq[:-1])])
	
	# Get the correct frequencies for the wavelet transform
	newfreq = w0 / a

	if len(inp.shape) == 1:
		outdict, power2 = [None, np.zeros((len(inp.data), nf))]
	elif len(inp.shape) == 2:
		outdict, power2 = [{}, None]
	else:
		raise TypeError("Invalid shape of the inp")

	newfreqmat, temp = np.meshgrid(newfreq, w)

	temp, ww = np.meshgrid(a, w)  # Matrix form

	# if scalar add virtual axis
	if len(inp.shape) == 1:
		data = data[:, np.newaxis]

	# go through all the datacolumns
	for i in range(data.shape[1]):
		# Make the FFT of all data
		datacol = data[:, i]

		"""
		Wavelet transform of the data
		"""

		# Forward FFT
		s_w = pyfftw.interfaces.numpy_fft.fft(datacol, threads=mp.cpu_count())

		aa, s_ww = np.meshgrid(a, s_w)  # Matrix form
		
		# Calculate the FFT of the wavelet transform
		w_w = np.sqrt(1) * s_ww * np.exp(-sigma * sigma * ((aa * ww - w0) ** 2) / 2)

		# Backward FFT
		power = pyfftw.interfaces.numpy_fft.ifft(w_w, axis=0, threads=mp.cpu_count())

		# Calculate the power spectrum
		if return_power:
			power = np.absolute((2 * np.pi) * np.conj(power) * power / newfreqmat)
		else:
			power = np.sqrt(np.absolute((2 * np.pi) / newfreqmat)) * power

		# Remove data possibly influenced by edge effects
		power2 = power

		if cut_edge:
			censur = np.floor(2 * a).astype(int)

			for j in range(anumber):
				power2[1:censur[j], j] = np.nan

				power2[len(datacol) - censur[j]:len(datacol), j] = np.nan
		else:
			continue
				
		if len(inp.shape) == 2:
			outdict[inp.comp.data[i]] = (["time", "frequency"], power2)
		else:
			continue
			
	if len(inp.shape) == 1:
		out = xr.DataArray(power2, coords=[Time(t, format="unix").datetime64, newfreq], dims=["time", "frequency"])
	elif len(inp.shape) == 2:
		out = xr.Dataset(outdict, coords={"time": Time(t, format="unix").datetime64, "frequency": newfreq})
	else:
		raise TypeError("Invalid shape")

	if plot_flag:
		if isinstance(out, xr.Dataset):
			fig, axs = plt.subplots(3, sharex="all")
			# fig.subplots_adjust(hspace=0)
			axs[0].pcolormesh(out.time, out.frequency, out.x.data, cmap="jet")
			axs[0].set_yscale('log')
			axs[0].set_ylabel("f [Hz]")
			axs[1].pcolormesh(out.time, out.frequency, out.y.data, cmap="jet")
			axs[1].set_yscale('log')
			axs[1].set_ylabel("f [Hz]")
			axs[2].pcolormesh(out.time, out.frequency, out.z.data, cmap="jet")
			axs[2].set_yscale('log')
			axs[2].set_ylabel("f [Hz]")
		else:
			fig, axs = plt.subplots(1)
			axs.pcolormesh(out.time, out.frequency, out.data.T, cmap="jet")
			axs.set_yscale('log')
			axs.set_ylabel("f [Hz]")

	return out
