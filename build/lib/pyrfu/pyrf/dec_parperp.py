import numpy as np
import xarray as xr

from .resample import resample
from .dot import dot



def dec_parperp(inp=None, b0=None, flagspinplane=False):
	"""
	Decomposes a vector into par/perp to B components. If flagspinplane decomposes components to the projection of B 
	into the XY plane. Alpha_XY gives the angle between B0 and the XY plane.
	
	Parameters :
		inp : DataArray
			Time series of the field to decompose

		b0 : DataArray
			Time series of the background magnetic field
	
	Options :
		flagspinplane : bool
			Flag if True gives the projection in XY plane
		
	Returns :
		apar : DataArray
			Time series of the input field parallel to the background magnetic field

		aperp : DataArray
			Time series of the input field perpendicular to the background magnetic field

		alpha : DataArray
			Time series of the angle between the background magnetic field and the XY plane

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field (FGM) and electric field (EDP)
		>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
		>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
		>>> # Decompose Exyz into parallel and perpendicular to Bxyz components
		>>> Epar, Eperp, alpha = pyrf.dec_parperp(Exyz,Bxyz)

	"""

	if (inp is None) or (b0 is None):
		raise ValueError("dec_parperp requires at least 2 arguments")
	
	if not isinstance(inp,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not isinstance(b0,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not flagspinplane:
		btot    = np.linalg.norm(b0,axis=1,keepdims=True)
		ii      = np.where(btot<1e-3)[0]
		if ii.size > 0 : 
			btot[ii] = np.ones(len(ii))*1e-3

		normb   = b0/btot
		normb   = resample(normb,inp)

		apar    = dot(normb,inp)
		aperp   = inp.data - (normb*np.tile(apar.data,(3,1)).T)
		alpha   = []
	else :
		b0      = resample(b0,inp)
		btot    = np.sqrt(b0[:,0]**2 + b0[:,1]**2)
		alpha   = np.arctan2(b0[:,2],btot)
		b0[:,0] = b0[:,0]/btot
		b0[:,1] = b0[:,1]/btot
		apar    = inp[:,0]*b0[:,0] + inp[:,1]*b0[:,1]
		aperp   = inp[:,0]*b0[:,1] - inp[:,1]*b0[:,0]

	return (apar, aperp, alpha)