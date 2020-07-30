import xarray as xr
import numpy as np



def movmean(inp=None, npts=100):
	"""
	Computes running average of the inp over npts points.
	
	Parameters :
		inp : DataArray
			Time series of the input variable

		npts : int
			Number of points to average over

	Returns :
		out : DataArray
			Time series of the input variable averaged over npts points

	Notes :
		Works also with 3D skymap distribution

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load ion pressure tensor
		>>> Pixyz = mms.get_data("Pi_gse_fpi_brst_l2",Tint,ic)
		>>> # Running average the pressure tensor over 10s
		>>> fs = pyrf.calc_fs(Pixyz)
		>>>> Pixyz = pyrf.movmean(Pixyz,10*fs)

	"""

	if inp is None:
		raise ValueError("movmean requires at least one argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if not isinstance(npts,int): 
		npts = np.floor(npts).astype(int)

	if npts%2: npts -=1

	# Computes moving average
	cumsum 	= np.cumsum(inp.data,axis=0)
	outdata = (cumsum[npts:,...]-cumsum[:-npts,...])/npts

	for k in keys: 
		if k == "time": 
			coords.append(inp.coords[k][int(npts/2):-int(npts/2)]) 
		else: 
			coords.append(inp.coords[k]) 

	# Output in DataArray type
	out = xr.DataArray(outdata,coords=coords,dims=inp.dims,attrs= inp.attrs)

	return out