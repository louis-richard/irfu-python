# -*- coding: utf-8 -*-
"""
wavepolarize_means.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from astropy.time import Time
import warnings

from .resample import resample


def wavepolarize_means(Bwave=None, Bbgd= None, **kwargs):
	"""
	Analysis the polarization of magnetic wave using "means" method

	Parameters :
		Bwave : DataArray
			Time series of the magnetic field from SCM

		Bbgd : DataArray
			Time series of the magnetic field from FGM.
		
	Options :
		minPsd : float
			Threshold for the analysis (e.g 1.0e-7). Below this value, the SVD analysis is meaningless if minPsd is 
			not given, SVD analysis will be done for all waves. (default 1e-25)

		nopfft : int
			Number of points in FFT. (default 256)

	Returns :
		Bpsd : DataArray
			Power spectrum density of magnetic filed wave.

		degpol : DataArray
			Spectrogram of the degree of polarization (form 0 to 1).

		waveangle : DataArray
			(form 0 to 90)

		elliptict : DataArray
			Spectrogram of the ellipticity (form -1 to 1)

		helict    : DataArray
			Spectrogram of the helicity (form -1 to 1)

	Example :
		>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd)
		>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd,1.0e-7)
		>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd,1.0e-7,256)

	Notice: Bwave and Bbgd should be from the same satellite and in the same coordinates 

	WARNING: If one component is an order of magnitude or more  greater than the other two then the polarization 
	results saturate and erroneously indicate high degrees of polarization at all times and frequencies. Time 
	series should be eyeballed before running the program.
	For time series containing very rapid changes or spikes the usual problems with Fourier analysis arise.
	Care should be taken in evaluating degree of polarization results.
	For meaningful results there should be significant wave power at the frequency where the polarization 
	approaches 100%. Remember comparing two straight lines yields 100% polarization.

	"""

	if (Bwave is None) or (Bbgd is None):
		raise ValueError("wavepolarize_means requires at least two arguments")

	minPsd = 1e-25
	nopfft = 256

	if "minPsd" in kwargs: minPsd = kwargs["minPsd"]
	if "nopfft" in kwargs: minPsd = kwargs["nopfft"]


	steplength=nopfft/2
	nopoints = len(Bwave)
	nosteps = (nopoints-nopfft)/steplength # total number of FFTs
	nosmbins = 7 # No. of bins in frequency domain
	aa = np.array([0.024,0.093,0.232,0.301,0.232,0.093,0.024]) # smoothing profile based on Hanning


	# change wave to MFA coordinates
	Bbgd = resample(Bbgd,Bwave)
	for ii in range(len(Bwave)):
		nb = Bbgd[ii,:]/np.linalg.norm(Bbgd[ii,:])
		nperp1 = np.cross(nb,[0,1,0]) 
		nperp1 = nperp1/np.linalg.norm(nperp1)
		nperp2 = np.cross(nb,nperp1)
		
		Bz[ii] = np.sum(Bwave[ii,:]*nb)
		Bx[ii] = np.sum(Bwave[ii,:]*nperp1)
		By[ii] = np.sum(Bwave[ii,:]*nperp2)

	ct = Time(Bwave.time.data,format="datetime64").unix

	# DEFINE ARRAYS
	xs = Bx
	ys = By
	zs = Bz
	sampfreq = 1/(ct[1]-ct[0])
	endsampfreq = 1/(ct[-1]-ctp[-2])
	if sampfreq != endsampfreq:
	   warnings.warn("file sampling frequency changes {} Hz to {} Hz".format(sampfreq,endsampfreq),UserWarning)
	else :
	   print("ac file sampling frequency {} Hz".format(sampfreq))


	for j in range(nosteps):
		# FFT CALCULATION
		smooth  = 0.08+0.46*(1-np.cos(2*np.pi*np.arange(1,nopfft+1)/nopfft))
		tempx   = smooth*xs[((j-1)*steplength+1):((j-1)*steplength+nopfft)]
		tempy   = smooth*ys[((j-1)*steplength+1):((j-1)*steplength+nopfft)]
		tempz   = smooth*zs[((j-1)*steplength+1):((j-1)*steplength+nopfft)]

		specx[j,:] = np.fft.fft(tempx)
		specy[j,:] = np.fft.fft(tempy)
		specz[j,:] = np.fft.fft(tempz)

		halfspecx[j,:] = specx[j,:(nopfft/2)]
		halfspecy[j,:] = specy[j,:(nopfft/2)]
		halfspecz[j,:] = specz[j,:(nopfft/2)]

		xs = np.roll(xs,-steplength)
		ys = np.roll(ys,-steplength)
		zs = np.roll(zs,-steplength)

		# CALCULATION OF THE SPECTRAL MATRIX
		matspec[j,:,0,0] = halfspecx[j,:]*np.conj(halfspecx[j,:])
		matspec[j,:,1,0] = halfspecx[j,:]*np.conj(halfspecy[j,:])
		matspec[j,:,2,0] = halfspecx[j,:]*np.conj(halfspecz[j,:])
		matspec[j,:,0,1] = halfspecy[j,:]*np.conj(halfspecx[j,:])
		matspec[j,:,1,1] = halfspecy[j,:]*np.conj(halfspecy[j,:])
		matspec[j,:,2,1] = halfspecy[j,:]*np.conj(halfspecz[j,:])
		matspec[j,:,0,2] = halfspecz[j,:]*np.conj(halfspecx[j,:])
		matspec[j,:,1,2] = halfspecz[j,:]*np.conj(halfspecy[j,:])
		matspec[j,:,2,2] = halfspecz[j,:]*np.conj(halfspecz[j,:])

		# CALCULATION OF SMOOTHED SPECTRAL MATRIX
		ematspec[j,:,:,:] = matspec[j,:,:,:]*np.nan
		for k in range(((nosmbins-1)/2),(nopfft/2-(nosmbins-1)/2)):
			ematspec[j,k,0,0] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),0,0])
			ematspec[j,k,1,0] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),1,0])
			ematspec[j,k,2,0] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),2,0])
			ematspec[j,k,0,1] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),0,1])
			ematspec[j,k,1,1] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),1,1])
			ematspec[j,k,2,1] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),2,1])
			ematspec[j,k,0,2] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),0,2])
			ematspec[j,k,1,2] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),1,2])
			ematspec[j,k,2,2] = np.sum(aa[:nosmbins]*matspec[j,(k-(nosmbins-1)/2):(k+(nosmbins-1)/2),2,2])


		# CALCULATION OF THE MINIMUM VARIANCE DIRECTION AND WAVENORMAL ANGLE
		aaa2[j,:]       = np.sqrt(np.imag(ematspec[j,:,0,1])**2+np.imag(ematspec[j,:,0,2])**2\
									+np.imag(ematspec[j,:,1,2])**2)
		wnx[j,:]        = -np.abs(np.imag(ematspec[j,:,1,2])/aaa2[j,:])
		wny[j,:]        = -np.abs(np.imag(ematspec[j,:,0,2])/aaa2[j,:])
		wnz[j,:]        = np.imag(ematspec[j,:,0,1])/aaa2[j,:]
		waveangle[j,:]  = np.arctan(np.sqrt(wnx[j,:]**2 + wny[j,:]**2)/np.abs(wnz[j,:]))

		# CALCULATION OF THE DEGREE OF POLARISATION
		# calc of square of smoothed spec matrix
		matsqrd[j,:,0,0] = ematspec[j,:,0,0]*ematspec[j,:,0,0]+ematspec[j,:,0,1]*ematspec[j,:,1,0]\
							+ematspec[j,:,0,2]*ematspec[j,:,2,0]
		matsqrd[j,:,0,1] = ematspec[j,:,0,0]*ematspec[j,:,0,1]+ematspec[j,:,0,1]*ematspec[j,:,1,1]\
							+ematspec[j,:,0,2]*ematspec[j,:,2,1]
		matsqrd[j,:,0,2] = ematspec[j,:,0,0]*ematspec[j,:,0,2]+ematspec[j,:,0,1]*ematspec[j,:,1,2]\
							+ematspec[j,:,0,2]*ematspec[j,:,2,2]
		matsqrd[j,:,1,0] = ematspec[j,:,1,0]*ematspec[j,:,0,0]+ematspec[j,:,1,1]*ematspec[j,:,1,0]\
							+ematspec[j,:,1,2]*ematspec[j,:,2,0]
		matsqrd[j,:,1,1] = ematspec[j,:,1,0]*ematspec[j,:,0,1]+ematspec[j,:,1,1]*ematspec[j,:,1,1]\
							+ematspec[j,:,1,2]*ematspec[j,:,2,1]
		matsqrd[j,:,1,2] = ematspec[j,:,1,0]*ematspec[j,:,0,2]+ematspec[j,:,1,1]*ematspec[j,:,1,2]\
							+ematspec[j,:,1,2]*ematspec[j,:,2,2]
		matsqrd[j,:,2,0] = ematspec[j,:,2,0]*ematspec[j,:,0,0]+ematspec[j,:,2,1]*ematspec[j,:,1,0]\
							+ematspec[j,:,2,2]*ematspec[j,:,2,0]
		matsqrd[j,:,2,1] = ematspec[j,:,2,0]*ematspec[j,:,0,1]+ematspec[j,:,2,1]*ematspec[j,:,1,1]\
							+ematspec[j,:,2,2]*ematspec[j,:,2,1]
		matsqrd[j,:,2,2] = ematspec[j,:,2,0]*ematspec[j,:,0,2]+ematspec[j,:,2,1]*ematspec[j,:,1,2]\
							+ematspec[j,:,2,2]*ematspec[j,:,2,2]


		Trmatsqrd[j,:]  = matsqrd[j,:,0,0]+matsqrd[j,:,1,1]+matsqrd[j,:,2,2]
		Trmatspec[j,:]  = ematspec[j,:,0,0]+ematspec[j,:,1,1]+ematspec[j,:,2,2]
		degpol[j,:]     = Trmatspec[j,:]*np.nan
		degpol[j,(nosmbins-1)/2:(nopfft/2-(nosmbins-1)/2)] = (3*Trmatsqrd[j,(nosmbins-1)/2:(nopfft/2-(nosmbins-1)/2)]\
															-Trmatspec[j,(nosmbins-1)/2:(nopfft/2-(nosmbins-1)/2)]**2)
		degpol[j,(nosmbins-1)/2:(nopfft/2-(nosmbins-1)/2)]/= 2*Trmatspec[j,(nosmbins-1)/2:(nopfft/2-(nosmbins-1)/2)]**2


		# CALCULATION OF HELICITY, ELLIPTICITY AND THE WAVE STATE VECTOR
		alphax[j,:]         = np.sqrt(ematspec[j,:,0,0])
		alphacos1x[j,:]     = np.real(ematspec[j,:,0,1])/np.sqrt(ematspec[j,:,0,0])
		alphasin1x[j,:]     = -np.imag(ematspec[j,:,0,1])/np.sqrt(ematspec[j,:,0,0])
		alphacos2x[j,:]     = np.real(ematspec[j,:,0,2])/np.sqrt(ematspec[j,:,0,0])
		alphasin2x[j,:]     = -np.imag(ematspec[j,:,0,2])/np.sqrt(ematspec[j,:,0,0])
		lambdau[j,:,0,0]    = alphax[j,:]
		lambdau[j,:,0,1]    = np.complex(alphacos1x[j,:],alphasin1x[j,:])
		lambdau[j,:,0,2]    = np.complex(alphacos2x[j,:],alphasin2x[j,:])

		alphay[j,:]         = np.sqrt(ematspec[j,:,0,0])
		alphacos1y[j,:]     = np.real(ematspec[j,:,0,1])/np.sqrt(ematspec[j,:,0,0])
		alphasin1y[j,:]     = -np.imag(ematspec[j,:,0,1])/np.sqrt(ematspec[j,:,0,0])
		alphacos2y[j,:]     = np.real(ematspec[j,:,0,2])/np.sqrt(ematspec[j,:,0,0])
		alphasin2y[j,:]     = -np.imag(ematspec[j,:,0,2])/np.sqrt(ematspec[j,:,0,0])
		lambdau[j,:,1,0]    = alphay[j,:]
		lambdau[j,:,1,1]    = np.complex(alphacos1y[j,:],alphasin1y[j,:])
		lambdau[j,:,1,2]    = np.complex(alphacos2y[j,:],alphasin2y[j,:])

		alphaz[j,:]         = np.sqrt(ematspec[j,:,2,0])
		alphacos1z[j,:]     = np.real(ematspec[j,:,2,1])/np.sqrt(ematspec[j,:,2,2])
		alphasin1z[j,:]     = -np.imag(ematspec[j,:,2,1])/np.sqrt(ematspec[j,:,2,2])
		alphacos2z[j,:]     = np.real(ematspec[j,:,2,2])/np.sqrt(ematspec[j,:,2,2])
		alphasin2z[j,:]     = -np.imag(ematspec[j,:,2,2])/np.sqrt(ematspec[j,:,2,2])
		lambdau[j,:,2,0]    = alphaz[j,:]
		lambdau[j,:,2,1]    = np.complex(alphacos1z[j,:],alphasin1z[j,:])
		lambdau[j,:,2,2]    = np.complex(alphacos2z[j,:],alphasin2z[j,:])

		
		for k in range(nopfft/2):
			for xyz in range(3):
				# HELICITY CALCULATION
				upper[j,k] = np.sum(2*np.real(lambdau[j,k,xyz,:3])*(np.imag(lambdau[j,k,xyz,:3])))
				lower[j,k] = np.sum((np.real(lambdau[j,k,xyz,:3]))**2-(np.imag(lambdau[j,k,xyz,:3]))**2)

				if upper[j,k] > 0:
					gamma[j,k] = np.atan(upper[j,k]/lower[j,k])
				else:
					gamma[j,k] = np.pi+(np.pi+np.atan(upper[j,k]/lower[j,k]))
				
				lambdau[j,k,xyz,:]  = np.exp(np.complex(0,-0.5*gamma[j,k]))*lambdau[j,k,xyz,:]
				helicity[j,k,xyz]   = np.sqrt(np.sum(np.real(lambdau[j,k,xyz,:3])**2))
				helicity[j,k,xyz]   /= np.sqrt(np.sum(np.imag(lambdau[j,k,xyz,:3])**2))
				helicity[j,k,xyz]   = np.divide(1,helicity[j,k,xyz])


				# ELLIPTICITY CALCULATION
				uppere = np.sum(np.imag(lambdau[j,k,xyz,:3])*np.real(lambdau[j,k,xyz,:3]))
				lowere = np.sum(np.real(lambdau[j,k,xyz,:2])**2) - np.sum(np.imag(lambdau[j,k,xyz,:2])**2)
				
				if uppere > 0:
					gammarot[j,k] = np.arctan(uppere/lowere)
				else:
					gammarot[j,k] = np.pi + np.pi + np.atan(uppere/lowere)
				

				lam = lambdau[j,k,xyz,:2]
				lambdaurot[j,k,:]   = np.exp(np.complex(0,-0.5*gammarot[j,k]))*lam[:]
				ellip[j,k,xyz]      = np.sqrt(np.sum(np.imag(lambdaurot[j,k,:2])**2))
				ellip[j,k,xyz]      /= np.sqrt(np.sum(np.real(lambdaurot[j,k,:2])**2))
				ellip[j,k,xyz]      *= -(np.imag(ematspec[j,k,0,1])*np.sin(waveangle[j,k]))
				ellip[j,k,xyz]      /= np.abs(np.imag(ematspec[j,k,0,1])*np.sin(waveangle[j,k]))
			
	# AVERAGING HELICITY AND ELLIPTICITY RESULTS
	elliptict   = np.mean(ellip,axis=-1)
	helict      = np.mean(helicity,axis=-1)

	# CREATING OUTPUT PARAMETER
	timeline = ct[0] + np.abs(nopfft/2)/sampfreq + np.arange(1,nosteps+1)*steplength/sampfreq
	binwidth = sampfreq/nopfft
	freqline = binwidth*np.arange(1,nopfft/2+1)

	# scaling power results to units with meaning
	W                       = nopfft*np.sum(smooth**2)
	powspec[:,1:nopfft/2-1] = 1/W*2*Trmatspec[:,1:nopfft/2-1]/binwidth
	powspec[:,1]            = 1/W*Trmatspec[:,0]/binwidth
	powspec[:,nopfft/2]     = 1/W*Trmatspec[:,nopfft/2]/binwidth

	"""
	# KICK OUT THE ANALYSIS OF THE WEAK SIGNALS
	index = find(powspec<minPsd);
	waveangle(index)=nan;
	degpol(index)=nan;
	elliptict(index)=nan;
	helict(index)=nan;
	"""

	# SAVE DATA AS DATAARRAY FORMAT 
	Bpsd        = xr.DataArray(powspec,coords=[timeline,freqline],dims=["t","f"])
	waveangle   = xr.DataArray(waveangle*180/np.pi,coords=[timeline,freqline],dims=["t","f"])
	degpol      = xr.DataArray(degpol,coords=[timeline,freqline],dims=["t","f"])
	elliptict   = xr.DataArray(elliptict,coords=[timeline,freqline],dims=["t","f"])
	helict      = xr.DataArray(helict,coords=[timeline,freqline],dims=["t","f"])

	Bpsd.f.attrs["units"]       = "Hz"
	waveangle.f.attrs["units"]  = "Hz"
	degpol.f.attrs["units"]     = "Hz"
	elliptict.f.attrs["units"]  = "Hz"
	helict.f.attrs["units"]     = "Hz"

	return(Bpsd,waveangle,degpol,elliptict,helict)