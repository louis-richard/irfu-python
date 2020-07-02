import numpy as np
import xarray as xr
from astropy import constants


def dynamicp(N=None, V=None, s="i"):
	"""
	Computes dynamic pressure

	Parameters :
		- N                 [xarray]                Time serie of the number density of the specie
		- V                 [xarray]                Time serie of the bulk velocity of the specie
		- s                 [i/e]                   Specie (default i)
	
	Returns :
		- Pdyn              [xarray]                Time serie of the dynamic pressure of the specie

	"""

	if N is None or V is None:
		raise ValueError("dynamicp requires at least 2 arguments")

	if not isinstance(N, xr.DataArray):
		raise TypeError("N must be a DataArray")

	if not isinstance(V, xr.DataArray):
		raise TypeError("V must be a DataArray")

	if s == "i":
		m = constants.m_p.value
	elif s == "e":
		m = constants.m_e.value
	else :
		raise ValueError("Unknown specie")
	
	V2 = np.linalg.norm(V,axis=0)**2

	Pdyn = N*V2

	return Pdyn