import numpy as np
import xarray as xr
from astropy import constants


from .resample import resample
from .ts_scalar import ts_scalar
from .rotate_tensor import rotate_tensor


def pres_anis(P=None, B=None):
	"""
	Compute pressure anisotropy factor: (P_para - P_perp) * mu0 / B^2

	Parameters :
		P : DataArray
			Time series of the pressure tensor
		B : DataArray
			Time series of the background magnetic field

	Returns :
		p_anis : DataArray
			Time series of the pressure anisotropy

	See also :
		rotate_tensor

	Example :
		>>> # Time interval
		>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
		>>> 
		>>> # Spacecraft index
		>>> ic = 1
		>>> 
		>>> # Load magnetic field, ion/electron temperature and number density
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
		>>> Pixyz = mms.get_data("Pi_gse_fpi_fast_l2",Tint,ic)
		>>> 
		>>> # Compute pressure anistropy
		>>> p_anis = pyrf.pres_anis(Pxyz,Bxyz)

	"""

	if (P is None) or (B is None):
		raise ValueError("pres_anis requires at least 2 arguments")

	if not isinstance(P,xr.Datarray):
		raise TypeError("P must be a DataArray")

	if not isinstance(B,xr.Datarray):
		raise TypeError("B must be a DataArray")

	B  = resample(B,Pi)

	# rotate pressure tensor to field aligned coordinates
	P = rotate_tensor(P,"fac",B,"pp")
	
	# Get parallel and perpendicular pressure
	P_para 	= Pi[:,0,0]
	P_perp 	= (Pi[:,1,1]+Pi[:,2,2])/2
	
	# Compute pressure anistropy
	mu0 	= constants.mu0.value
	p_anis 	= 1e9*mu0*(Pi_para - Pi_perp)/np.linalg.norm(Bres)**2
	
	return p_anis