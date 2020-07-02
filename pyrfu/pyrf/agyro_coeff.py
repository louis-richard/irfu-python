import numpy as np
import xarray as xr


def agyro_coeff(P=None):
	"""
	Compues agyrotropy coefficient (Swidak2016 https://doi.org/10.1002/2015GL066980)
	
	Parameters :
		- P                 [xarray]                Time serie of the pressure tensor
		
	Returns :
		- Q                 [xarray]                Time serie of the agyrotropy coefficient of the specie
		
	"""
	
	if P is None:
		raise ValueError("agyro_coeff requires at least one argument")

	if not isinstance(P,xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if P.ndim != 3:
		raise TypeError("Input must be a second order tensor")

	P_para  = P[:,0,0]
	P_perp  = (P[:,1,1]+P[:,2,2])/2
	P_12    = P[:,0,1]
	P_13    = P[:,0,2]
	P_23    = P[:,1,2]

	Q = (P_12**2+P_13**2+P_23**2)/(P_perp**2+2*P_perp*P_para)

	return Q