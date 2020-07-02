import numpy as np
import xarray as xr
import warnings
from scipy import signal
from astropy.time import Time
import pyfftw
import multiprocessing as mp



def psd(inp=None, nfft=256, noverlap=128, window="hamming", dflag="constant", scalling="density"):
	"""
	Estimate power spectral density using Welch's method.
	
	Welch's method [1]_ computes an estimate of the power spectral density by dividing the data into overlapping 
	segments, computing a modified periodogram for each segment and averaging the periodograms.
	
	Parameters : 
		- inp               [xarray]                Time series of measurement values
	
		- window            [str]                   (optional) Desired window to use. It is passed to `get_window` to 
													generate the window values, which are DFT-even by default. See 
													`get_window` or a list of windows and required parameters. 
													Defaults Hanning
		- noverlap          [int]                   (optional) Number of points to overlap between segments. 
													Defaults to 128.
		- nfft              [int]                   (optional) Length of the FFT used, if a zero padded FFT is desired 
													Defaults to 256
		- dflag             [str]                   (optional) Specifies how to detrend each segment. It is passed as 
													the `type` argument to the `detrend` function. 
													Defaults to 'constant'.
		- scaling           [str]                   (optional) Selects between computing the power spectral density 
													('density') where `Pxx` has units of V**2/Hz and computing the 
													power spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
													is measured in V and `fs` is measured in Hz. Defaults to 'density'

	Returns : 
		- out               [xarray]                Power spectral density or power spectrum of inp
		
	References : 
		- [1] :     P. Welch, "The use of the fast Fourier transform for the estimation of power spectra: A method
					based on time averaging over short, modified periodograms", IEEE Trans. Audio Electroacoust. 
					vol. 15, pp. 70-73, 1967.
		- [2] :     M.S. Bartlett, "Periodogram Analysis and Continuous Spectra", Biometrika, vol. 37, pp. 1-16, 1950.

	"""
	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if inp.ndim == 2 and inp.shape[-1] == 3:
		inp = abs(inp)

	if not noverlap is None:
		nperseg = 2*noverlap
	else :
		nperseg     = 256
		noverlap    = nperseg/2


	if nfft < nperseg:
		nfft = nperseg
		warnings.warn("nfft < nperseg. set to nperseg",UserWarning)

	dt = np.median(np.diff(inp.time.data)).astype(float)*1e-9
	Fs = 1/dt

	f, Pxx = signal.welch(inp.data, nfft=nfft, fs=Fs, window=window, noverlap=noverlap, detrend=dflag, \
							nperseg=nperseg, scaling=scalling, return_onesided=True, axis=-1)

	out = xr.DataArray(Pxx,coords=[f],dims=["f"])

	return out
#-----------------------------------------------------------------------------------------------------------------------
def psd_m(inp=None, comp="", window="hanning", noverlap=0, nfft=256, dflag=None):
	"""
	Estimates the Power Spectral Density of signal vector X using Welch's averaged periodogram method.  X is divided
	into overlapping sections, each of which is detrended, then windowed by the WINDOW parameter, then zero-padded to 
	length NFFT. The magnitude squared of the length NFFT DFTs of the sections are averaged to form Pxx.  Pxx is length 
	NFFT/2+1 for NFFT even, (NFFT+1)/2 for NFFT odd, or NFFT if the signal X is complex.  If you specify a scalar for
	WINDOW, a Hanning window of that length is used.  Fs is the sampling frequency which doesn't affect the spectrum 
	estimate but is used for scaling of plots.  See Page 556, A.V. Oppenheim and R.W. Schafer, Digital Signal 
	Processing, Prentice-Hall, 1975.

	Parameters :
		- inp               [xarray]                Input time serie
		- comp              [str]                   Target component. If not given (default) compute power spectral 
													density of the magnitude
		- window            [str]                   Type of window applied hannng (default), etc.
		- noverlap          [int]                   Number of samples of overlapping of the sections of inp
		- nfft              [int]                   Number of frequencies of fft
		- dflag             [str]                   Detrending mode for the prewindowed sections of inp. 
													None (default), linear or mean
	
	Returns :
		- out               [xarray]                Power Spectrum of the input
	"""

	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")

	t   = Time(inp.time.data,format="datetime64").unix
	N   = len(inp)
	dt  = np.median(np.diff(t))
	fs  = 1/dt

	if comp:
		x = inp.sel(comp=comp).data
	else :
		x = abs(inp).data


	window = eval("np."+window+"(nfft)")

	n       = len(x)
	nwind   = len(window)

	k = np.fix((n-noverlap)/(nwind-noverlap)).astype(int)

	index = np.arange(nwind)

	KMU = k*np.linalg.norm(window)**2

	Spec = np.zeros(nfft)
	for i in range(k):
		if dflag == None:
			xw = window*(x[index])
		elif isinstance(dflag,str) and dflag == "linear":
			xw = window*signal.detrend(x[index],type='linear')
		else :
			xw = window*signal.detrend(x[index],type='constant')

		index   = index + (nwind - noverlap)
		Xx      = np.abs(pyfftw.interfaces.numpy_fft.fft(xw,nfft,axis=0,threads=mp.cpu_count()))**2
		Spec    = Spec + Xx

	if not any(np.imag(x) != 0):
		if np.remainder(nfft,2):
			select = np.arange((nff+1)/2)
		else :
			select = np.arange(nfft/2+1).astype(int)
		Spec = Spec[select]
	else :
		select = np.arange(nfft)

	freq_vector = select*fs/nfft

	# I hate decibells, added to make correct units - AV 97.12.20
	Spec=Spec/fs*2
	#confid=confid/Fs*2;
	#
	# original line
	Spec = Spec*(1/KMU)   # normalize
	f = freq_vector

	out = xr.DataArray(Spec,coords=[f],dims="freq")
	return out