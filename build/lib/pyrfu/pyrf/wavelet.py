import numpy as np
import xarray as xr
import pyfftw
import multiprocessing as mp
from astropy.time import Time

import seaborn as sns
from cycler import cycler
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

color 			= ["k","b","r","g","c","m","y"]
default_cycler 	= cycler(color=color)
date_form 		= mdates.DateFormatter("%H:%M:%S")
sns.set()
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8,'axes.grid': True,\
                        'font.family': ['sans-serif']})
sns.set_context("paper")
plt.rc('axes',prop_cycle=default_cycler)
plt.rc('lines', linewidth=0.8)
plt.close("all")


def wavelet(inp=None,**kwargs):
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
	if isinstance(inp,xr.DataArray):
		tstart  = Time(inp.time.data[0],format="datetime64").unix
		tstop   = Time(inp.time.data[-1],format="datetime64").unix
		timeint = tstop - tstart 

		Fs  = len(inp)/timeint
		t   = inp.time.data.view("i8")*1e-9
		#t = Time(inp.time.data,format="datetime64").unix
		data = inp.data
	else :
		raise TypeError("Input must be a DataArray")

	# f
	amin = 0.01             # The highest frequency to consider is 0.5*sampl/10^amin
	fmax = 0.5*Fs/10**amin
	amax = 2                # The lowest frequency to consider is 0.5*sampl/10^amax
	fmin = 0.5*Fs/10**amax

	# nf
	nf = 200

	# wavelet_width
	wavelet_width = 5.36

	#Other
	returnpower = True
	cutedge     = True
	lineardf    = False
	plot_flag   = True


	if "returnpower" in kwargs:
		returnpower = kwargs["returnpower"]

	if "cutedge" in kwargs:
		cutedge = kwargs["cutedge"]

	if "fs" in kwargs:
		if isinstance(kwargs["fs"],int) or isinstance(kwargs["fs"],float):
			fs = kwargs["fs"]
		else :
			raise TypeError("fs must be numeric")

	if "nf" in kwargs:
		if isinstance(kwargs["nf"],int) or isinstance(kwargs["nf"],float):
			nf = kwargs["nf"]
		else :
			raise TypeError("nf must be numeric")

	if "linear" in kwargs:
		lineardf = True
		if isinstance(kwargs["linear"],int) or isinstance(kwargs["linear"],float):
			deltaf = kwargs["linear"]
		else :
			deltaf = 100
			raise Warning("Unknow input for linear deltaf set to 100")

	if "wavelet_width" in kwargs:
		if isinstance(kwargs["wavelet_width"],int) or isinstance(kwargs["wavelet_width"],float):
			wavelet_width = kwargs["wavelet_width"]
		else :
			raise TypeError("wavelet_width must be numeric")

	if "f" in kwargs:
		if isinstance(kwargs["f"],np.ndarray) or isinstance(kwargs["f"],list):
			if len(kwargs["f"]) == 2:
				fmin = kwargs["f"][0]
				fmax = kwargs["f"][1]
			else :
				raise IndexError("f should have vector with 2 elements as parameter value")
		else :
			raise TypeError("f must be a list or array")

	if "plot" in kwargs:
		plot_flag =  kwargs["plot"]

	sampl   = Fs
	w0      = sampl/2               # The maximum frequency
	anumber = nf                    # The number of frequencies
	sigma   = wavelet_width/(Fs/2)  # The width of the Morlet wavelet


	if lineardf :
		fmin    = deltaf
		anumber = np.floor(w0/deltaf).astype(int)
		fmax    = anumber*deltaf
		a       = w0/(np.linspace(fmax,fmin,anumber))
	else :
		amin    = np.log10(0.5*Fs/fmax)             # The highest frequency to consider is 0.5*sampl/10^amin
		amax    = np.log10(0.5*Fs/fmin)             # The lowest frequency to consider is 0.5*sampl/10^amax
		a       = np.logspace(amin,amax,anumber)

	# Remove the last sample if the total number of samples is odd
	if len(data)/2 != np.floor(len(data)/2):
		data    = data[:-1,...]
		t       = t[:-1]


	# Check for NaNs
	a[np.isnan(a)] = 0

	# Find the frequencies for an FFT of all data
	nd2     = len(data)/2
	nyq     = 1/2 
	freq    = sampl*np.arange(1,nd2+1)/(nd2)*nyq
	w       = np.hstack([0,freq,-np.flip(freq[:-1])]) # The frequencies corresponding to FFT


	# construct specrec
	#specrec.p=cell(1,size(data,2));
	#specrec.t=t;
	
	# Get the correct frequencies for the wavelet transform
	newfreq = w0/a;
	if len(data.shape) == 2:
		outdict = {}

	#specrec.f=newfreq(:);
	[newfreqmat,temp]   = np.meshgrid(newfreq,w)
	[temp,ww]           = np.meshgrid(a,w); # Matrix form

	# if scalar add virtual axis
	if len(inp.shape) == 1:
		data = data[:,np.newaxis]

	

	#a = pyfftw.empty_aligned(npts, dtype='complex128')
	#Sw = pyfftw.empty_aligned(npts, dtype='complex128')
	#W = pyfftw.empty_aligned(npts, dtype='complex128')
	
	#fft_object = pyfftw.FFTW(a, b)
	#ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')

	for i in range(data.shape[1]): # go through all the datacolumns

		# Make the FFT of all data
		datacol = data[:,i]
		
		#nthread = multiprocessing.cpu_count()
		#Sw = pyfftw.empty_aligned(data[:,i].shape, dtype='complex128')
		#fft_object = pyfftw.FFTW(data[:,i].astype("complex128"), Sw,axes=(0,), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=nthread)
		Sw = pyfftw.interfaces.numpy_fft.fft(datacol,threads=mp.cpu_count())
		
		#  Sw=fft(detrend(Ex));
		[aa,Sww] = np.meshgrid(a,Sw) # Matrix form
		
		# Calculate the FFT of the wavelet transform
		#pdb.set_trace()
		Ww = np.sqrt(1)*Sww*np.exp(-sigma*sigma*((aa*ww-w0)**2)/2)
		
		
		## Get the wavelet transform by IFFT of the FFT
		#W = pyfftw.empty_aligned(Ww.shape, dtype='complex128')
		#fft_objectb = pyfftw.FFTW(Ww, W,axes=(0,), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=nthread)
		W = pyfftw.interfaces.numpy_fft.ifft(Ww,axis=0,threads=mp.cpu_count())
		
		power = W
		
		# Calculate the power spectrum
		if returnpower :
			power = np.absolute((2*np.pi)*np.conj(W)*W/newfreqmat)
		else :
			power = np.sqrt(np.absolute((2*np.pi)/newfreqmat))*power

		
		# Remove data possibly influenced by edge effects
		
		power2 = power
		if cutedge :
			censur = np.floor(2*a).astype(int)
			for j in range(anumber):
				power2[1:censur[j],j] = np.nan
				power2[len(datacol)-censur[j]:len(datacol),j] = np.nan
				
		if len(inp.shape) == 2:
			outdict[inp.comp.data[i]] = (["time","frequency"], power2)
			
	if len(inp.shape) == 1:
		out = xr.DataArray(power2,coords=[Time(t,format="unix").datetime64,newfreq], dims=["time","frequency"])
	elif len(inp.shape) == 2:
		out = xr.Dataset(outdict,coords={"time" : Time(t,format="unix").datetime64,"frequency" : newfreq})

	if plot_flag:
		if isinstance(out,xr.Dataset):
			fig, axs = plt.subplots(3,sharex=True)
			#fig.subplots_adjust(hspace=0)
			axs[0].pcolormesh(out.time,out.frequency,out.x.data,cmap="jet")
			axs[0].set_yscale('log')
			axs[0].set_ylabel("f [Hz]")
			axs[1].pcolormesh(out.time,out.frequency,out.y.data,cmap="jet")
			axs[1].set_yscale('log')
			axs[1].set_ylabel("f [Hz]")
			axs[2].pcolormesh(out.time,out.frequency,out.z.data,cmap="jet")
			axs[2].set_yscale('log')
			axs[2].set_ylabel("f [Hz]")
		else :
			fig, axs = plt.subplots(1)
			axs.pcolormesh(out.time,out.frequency,out.data.T,cmap="jet")
			axs.set_yscale('log')
			axs.set_ylabel("f [Hz]")
	return out