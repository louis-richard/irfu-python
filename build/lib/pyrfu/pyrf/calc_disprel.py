import numpy as np
import xarray as xr
from scipy import optimize


def calc_disprel(V,dV,T,dT):
	"""
	Computes diespertion relation from velocities and period

	Parameters :
		- V                 [xarray]                Time serie of the velocities
		- dV                [xarray]                Time serie of the error on velocities
		- T                 [xarray]                Time serie of the periods
		- dT                [xarray]                Time serie of the error on period

	Returns :
		- out               [xarray]                DataSet containing the frequency, the wavelenth, the wavenumber.
													Also includes the errors and the fit (e.g .Vph phase velocity)

	"""

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