import numpy as np
import xarray as xr
from scipy import optimize


def calc_disprel_tm(V=None, dV=None, T=None, dT=None):
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

	if (V is None) or (dV is None) or (T is None) or (dT is None):
		raise ValueError("calc_disprel_tm requires at least 4 arguments")

	if not isinstance(V,xr.DataArray):
		raise TypeError("V must a DataArray")

	if not isinstance(dV,xr.DataArray):
		raise TypeError("dV must a DataArray")

	if not isinstance(T,xr.DataArray):
		raise TypeError("T must a DataArray")

	if not isinstance(dT,xr.DataArray):
		raise TypeError("dT must a DataArray")

	omega   = 2*np.pi/T                             # Frequency
	lamb    = V*T                                   # Wave length
	k       = 2*np.pi/lamb                          # Wave number

	# Estimate errors
	domega  = omega*((dT/T)/(1+dT/T))               # Error on frequency
	dlamb   = dV*T                                  # Error on wave length
	dk      = k*((dlamb/lamb)/(1+dlamb/lamb))       # Error on wave number
	
	
	modelTV         = lambda x,a: a/x
	fitTV, covTV    = optimize.curve_fit(modelTV,T,V,1,sigma=np.sqrt(dV**2+dT**2))
	sigmaTV         = np.sqrt(np.diagonal(covTV))
	hires_T         = np.logspace(np.log10(5),np.log10(2e3),int(1e4))
	bound_upper_V   = modelTV(hires_T, *(fitTV + 1.96*sigmaTV))
	bound_lower_V   = modelTV(hires_T, *(fitTV - 1.96*sigmaTV))
	pred_V          = modelTV(hires_T,*fitTV)
	
	
	model       = lambda x,a: a*x
	fit, cov    = optimize.curve_fit(model,k,omega,1,sigma=np.sqrt(domega**2+dk**2))
	sigma       = np.sqrt(np.diagonal(cov))
	hires_k     = np.linspace(0,0.003,int(1e4))
	bound_upper = model(hires_k, *(fit + 1.96*sigma))
	bound_lower = model(hires_k, *(fit - 1.96*sigma))
	pred_omega  = model(hires_k,*fit)
	
	
	outdict     = { "T"             : T                             ,\
					"dT"            : (["T"], dT)                   ,\
					"V"             : (["T"], V)                    ,\
					"dV"            : (["T"], dV)                   ,\
					"lamb"          : (["T"], lamb)                 ,\
					"dlamb"         : (["T"], dlamb)                ,\
					"k"             : k                             ,\
					"dk"            : (["k"], dk)                   ,\
					"omega"         : (["k"], omega)                ,\
					"domega"        : (["k"], domega)               ,\
					"hires_k"       : hires_k                       ,\
					"pred_omega"    : (["hires_k"],pred_omega)      ,\
					"bound_upper"   : (["hires_k"], bound_upper)    ,\
					"bound_lower"   : (["hires_k"], bound_lower)    ,\
					"hires_T"       : hires_T                       ,\
					"pred_V"        : (["hires_T"], pred_V)         ,\
					"bound_upper_V" : (["hires_T"], bound_upper_V)  ,\
					"bound_lower_V" : (["hires_T"], bound_lower_V)  ,\
					"l"             : fitTV                         ,\
					"Vph"           : fit                           ,\
					}

	
	out = xr.Dataset(outdict)
	return out