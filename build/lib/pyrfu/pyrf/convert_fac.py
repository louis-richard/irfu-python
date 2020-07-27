import numpy as np
import xarray as xr

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def convert_fac(inp=None, Bbgd=None, r=np.array([1,0,0])):
	"""
	Transforms to a field-aligned coordinate (FAC) system defined as:
	R_parallel_z aligned with the background magnetic field
	R_perp_y defined by R_parallel cross the position vector of the spacecraft (nominally eastward at the equator)
	R_perp_x defined by R_perp_y cross R_par
	If inp is one vector along r direction, out is inp[perp, para] projection
	
	Parameters :
		inp : DataArray
			Time series of the input field

		Bbgd : DataArray
			Time series of the background magnetic field

		r : DataArray/ndarray/list
			Position vector of spacecraft

	Returns :
		out : DataArray
			Time series of the input field in field aligned coordinates system

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field (FGM) and electric field (EDP)
		>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
		>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
		>>> # Convert to field aligned coordinates
		>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
	
	Note : 
		all input parameters must be in the same coordinate system
	
	"""
	
	if (inp is None) or (Bbgd is None):
		raise ValueError("convert_fac requires at least 2 arguments")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if not isinstance(Bbgd,xr.DataArray):
		raise TypeError("Bbgd must be a DataArray")

	if len(inp) != len(Bbgd):
		Bbgd = resample(Bbgd,inp)
	
	t           = inp.time
	inp_data    = inp.data
	Bbgd        = Bbgd
	Bn          = Bbgd/np.linalg.norm(Bbgd,axis=1,keepdims=True)

	if isinstance(r,(list,np.ndarray)) and len(r) == 3:
		r = np.tile(r,(len(Bbgd),1))
	elif isinstance(r, xr.DataArray):
		r = resample(r,Bbgd)
	
	Rpar    = Bn
	Rperpy  = np.cross(Rpar,r,axis=1)
	Rperpy  = Rperpy/np.linalg.norm(Rperpy,axis=1,keepdims=True)
	Rperpx  = np.cross(Rperpy,Bbgd,axis=1)
	Rperpx  = Rperpx/np.linalg.norm(Rperpx,axis=1,keepdims=True)
	
	(ndata, ndim) = inp_data.shape
	
	if ndim == 3:
		outdata         = np.zeros(inp.shape)
		outdata[:,2]    = Rpar[:,0]*inp_data[:,0]+  Rpar[:,1]*inp_data[:,1]+  Rpar[:,2]*inp_data[:,2]
		outdata[:,0]    = Rperpx[:,0]*inp_data[:,0]+Rperpx[:,1]*inp_data[:,1]+Rperpx[:,2]*inp_data[:,2]
		outdata[:,1]    = Rperpy[:,0]*inp_data[:,0]+Rperpy[:,1]*inp_data[:,1]+Rperpy[:,2]*inp_data[:,2]
		out             = xr.DataArray(outdata,coords=[inp.time.data,inp.comp],dims=["time","comp"])

	elif ndim ==1:
		outdata         = np.zeros((2,ndata))
		outdata[:,0]    = inp[:,0]*(Rperpx[:,0]*r[:,0] + Rperpx[:,1]*r[:,1] + Rperpx[:,2]*r[:,2])
		outdata[:,1]    = inp[:,0]*(Rpar[:,0]*r[:,0]    + Rpar[:,1]*r[:,1]   + Rpar[:,2]*r[:,2]) 
		out             = ts_vec_xy(inp.time.data,outdata,attrs=inp.attrs)

	return out