import xarray as xr
import numpy as np
from dateutil import parser
from astropy.time import Time
import bisect



def tlim(inp=None, tint=None):
	"""
	Time clip the input (if time interval is TSeries clip between start and stop)

	Parameters :
		- inp               [xarray]                Quantity to clip
		- tint              [xarray/ndarray/list]   Time interval can be a time series, a array of datetime64 or a list

	Returns : 
		- out               [xarray]                Normalized TSeries

	"""

	if type(inp) != xr.DataArray: raise TypeError('Input must be a TSeries')
	
	if type(tint) == xr.DataArray:
		tstart  = tint.time.data[0]
		tstop   = tint.time.data[-1]
	elif type(tint) == np.ndarray:
		if type(tint[0]) == datetime.datetime and type(tint[-1]) == datetime.datetime:
			tstart  = tint.time[0]
			tstop   = tint.time[-1]
		else :
			raise TypeError('Values must be in Datetime64')
	elif type(tint) == list:
		tstart  = parser.parse(tint[0])
		tstop   = parser.parse(tint[-1])
	
	

	idxmin = bisect.bisect_left(inp.time.data,Time(tstart,format="datetime").datetime64)
	idxmax = bisect.bisect_right(inp.time.data,Time(tstop,format="datetime").datetime64)

	coords = [inp.time.data[idxmin:idxmax]]
	if len(inp.coords) > 1:
		for k in inp.dims[1:]:
			coords.append(inp.coords[k])

	out = xr.DataArray(inp.data[idxmin:idxmax,...],coords=coords,dims=inp.dims,attrs=inp.attrs)
	out.time.attrs = inp.time.attrs
	return out