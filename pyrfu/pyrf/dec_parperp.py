import numpy as np
import xarray as xr

from .resample import resample
from .dot import dot




def dec_parperp(inp=None, b0=None, flagspinplane=False):
	"""
	Decompose a vector into par/perp to B components. If flagspinplane decomposes components to the projection of B 
	into the XY plane. Alpha_XY gives the angle between B0 and the XY plane.
	
	Parameters :
		- inp               [xarray]                Field to decompose
		- b0                [xarray]                Background magnetic field
		- flagspinplane     [bool]                  Flag if True gives the projection in XY plane
		
	Returns :
		- apar              [xarray]                Parrallel component
		- aperp             [xarray]                Perpandicular component
		- alpha             [xarray]                Angle between B0 and the XY plane

	"""
	
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