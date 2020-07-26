import xarray as xr
import numpy as np
from scipy import signal




def medfilt(inp=None,npts=11):
	"""
	Applies a median filter over npts points to inp

	Parameters :
		inp : DataArray
			Time series of the input variable

		npts : float/int
			Number of points of median filter

	Returns :
		out : DataArry
			Time series of the median filtered input variable
	
	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> # Load magnetic field and electric field
		>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,1) for i in ic]
		>>> Rxyz = [mms.get_data("R_gse",Tint,1) for i in ic]
		>>> # Compute current density, etc
		>>> J, divB, Bavg, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
		>>> # Get J sampling frequency
		>>> fs = pyrf.calc_fs(J)
		>>> # Median filter over 1s
		>>> J = pyrf.medfilt(J,fs)


	"""

	if inp is None:
		raise ValueError("medfilt requires at least 1 argument")
	
	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")
	
	if not isinstance(npts,(float,int)):
		raise TypeError("npts must be a float or int")

	if isinstance(npts,float):
		npts = np.floor(npts).astype(int)

	if npts%2 == 0:
		npts +=1

	nt = len(inp)  

	if inp.ndim == 3:
		inpdata = np.reshape(inp.data,[nt,9])
	else :
		inpdata = inp.data

	try :
		ncomp = inpdata.shape[1]
	except IndexError:
		ncomp   = 1
		inpdata = inpdata[...,None]

	outdata = np.zeros(inpdata.shape)

	if not npts%2:
		npts += 1

	for i in range(ncomp):
		outdata[:,i] = signal.medfilt(inpdata[:,i],npts)

	if ncomp == 9:
		outdata = np.reshape(outdata,[nt,3,3])

	if outdata.shape[1] == 1:
		outdata = outdata[:,0]
	out = xr.DataArray(outdata,coords=inp.coords,dims=inp.dims)
	out.attrs = inp.attrs
	
	return out