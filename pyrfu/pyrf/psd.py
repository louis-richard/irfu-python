import numpy as np
import xarray as xr
import warnings
from scipy import signal


def psd(inp=None, nfft=256, n_overlap=128, window="hamming", dflag="constant", scalling="density"):
	"""
	Estimate power spectral density using Welch's method.
	
	Welch's method [1]_ computes an estimate of the power spectral density by dividing the data into overlapping 
	segments, computing a modified periodogram for each segment and averaging the periodograms.
	
	Parameters : 
		- inp : DataArray
			Time series of measurement values
	
	Options :
		- window str
			Desired window to use. It is passed to `get_window` to 
													generate the window values, which are DFT-even by default. See 
													"get_window" or a list of windows and required parameters. 
													Defaults Hanning
		- noverlap          [int]                   (optional) Number of points to overlap between segments. 
													Defaults to 128.
		- nfft              [int]                   (optional) Length of the FFT used, if a zero padded FFT is desired 
													Defaults to 256
		- dflag             [str]                   (optional) Specifies how to detrend each segment. It is passed as 
													the "type" argument to the "detrend" function. 
													Defaults to "constant".
		- scaling           [str]                   (optional) Selects between computing the power spectral density 
													('density') where `Pxx` has units of V**2/Hz and computing the 
													power spectrum ("spectrum") where "Pxx" has units of V**2, if `x`
													is measured in V and "fs" is measured in Hz. Defaults to 'density'

	Returns : 
		- out               [xarray]                Power spectral density or power spectrum of inp
		
	References : 
		- [1] :     P. Welch, "The use of the fast Fourier transform for the estimation of power spectra: A method
					based on time averaging over short, modified periodograms", IEEE Trans. Audio Electroacoust. 
					vol. 15, pp. 70-73, 1967.
		- [2] :     M.S. Bartlett, "Periodogram Analysis and Continuous Spectra", Biometrika, vol. 37, pp. 1-16, 1950.

	"""
	if not isinstance(inp, xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if inp.ndim == 2 and inp.shape[-1] == 3:
		inp = np.abs(inp)

	if n_overlap is None:
		n_persegs = 256
		n_overlap = n_persegs / 2
	else:
		n_persegs = 2 * n_overlap

	if nfft < n_persegs:
		nfft = n_persegs
		warnings.warn("nfft < n_persegs. set to n_persegs", UserWarning)

	dt = np.median(np.diff(inp.time.data)).astype(float) * 1e-9
	fs = 1 / dt

	f, p_xx = signal.welch(inp.data, nfft=nfft, fs=fs, window=window, noverlap=n_overlap, detrend=dflag, \
							nperseg=n_persegs, scaling=scalling, return_onesided=True, axis=-1)

	out = xr.DataArray(p_xx, coords=[f], dims=["f"])

	return out
