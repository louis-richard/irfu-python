import xarray as xr
import numpy as np



def movmean(inp=None, N=100):
	"""
	Computes running average of the inp over N points.
	
	Parameters :
		- inp 				[xarray] 				Input time serie/distribution
		- N 				[int] 					Number of points to average

	Returns :
		- out 				[xarray] 				Input time serie/distribution averaged over N points

	"""

	if inp is None:
		raise ValueError("movmean requires at least one argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if not isinstance(N,int): 
		N = int(N)

	if N%2: N -=1

	# Computes moving average
	cumsum 	= np.cumsum(inp.data,axis=0)
	outdata = (cumsum[N:]-cumsum[:-N])/N

	for k in keys: 
		if k == "time": 
			coords.append(inp.coords[k][int(N/2):-int(N/2)]) 
		else: 
			coords.append(inp.coords[k]) 

	# Output in DataArray type
	out = xr.DataArray(outdata,coords=coords,dims=inp.dims,attrs= inp.attrs)

	return out