import numpy as np
import xarray as xr
from astropy import constants


def dynamicp(N=None, V=None, s="i"):
	"""
	Computes dynamic pressure

	Parameters :
		N : DataArray
			Time series of the number density of the specie
		V : DataArray
			Time series of the bulk velocity of the specie
	
	Options :
		s : "i"/"e"
			Specie (default "i")
	
	Returns :
		Pdyn : DataArray
			Time series of the dynamic pressure of the specie

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Ion bluk velocity
		>>> Vixyz = mms.get_data("Vi_gse_fpi_fast_l2",Tint,ic)
		>>> # Remove spintone
		>>> STixyz 	= mms.get_data("STi_gse_fpi_fast_l2",Tint,ic)
		>>> Vixyz 	= Vixyz-STixyz
		>>> # Ion number density
		>>> Ni = mms.get_data("Ni_fpi_fast_l2",Tint,ic)
		>>> # Compute dynamic pressure
		>>> Pdyn = pyrf.dynamicp(Ni,Vixyz, s="i")

	"""

	if (N is None) or (V is None):
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