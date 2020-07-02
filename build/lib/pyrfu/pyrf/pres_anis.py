import numpy as np
import xarray as xr

from .resample import resample
from .ts_scalar import ts_scalar



def pres_anis(Pi=None, B=None):
	"""
	Compute ion pressure anisotropy factor: (Pi_para - Pi_perp) * mu0 / B^2

	Parameters :
		- Pi                [xarray]                Pressure tensor
		- B                 [xarray]                Magnetic field

	Returns :
		- out               [xarray]                Pressure anisotropy

	"""

	if Pi is None or B is None:
		raise ValueError("pres_anis requires at least 2 arguments")

	if not isinstance(Pi,xr.Datarray):
		raise TypeError("Pi must be a DataArray")

	if not isinstance(B,xr.Datarray):
		raise TypeError("B must be a DataArray")

	Bres    = resample(B,Pi)
	Pi_para = Pi[:,0,0]
	Pi_perp = (Pi[:,1,1]+Pi[:,2,2])/2
	res     = 4*3.14159e2*(Pi_para - Pi_perp)/np.linalg.norm(Bres)**2
	out     = ts_scalar(res.time.data,res.data)

	return out