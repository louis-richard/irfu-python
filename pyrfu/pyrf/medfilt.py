import xarray as xr
import numpy as np
from scipy import signal




def medfilt(inp=None,npts=11):
	"""
	Applies a median filter over npts points to inp

	Parameters :
		- inp               [xarray]                Input time serie
		- npts              [int]                   Number of points of median filter

	Returns :
		- out               [xarray]                Filtered time serie

	"""

	if inp is None:
		raise ValueError("medfilt requires at least 1 argument")
	
	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")
	
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