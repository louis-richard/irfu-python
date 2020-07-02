import numpy as np
import xarray as xr


def mean_bins(x=None,y=None,nbins=10):
	"""
	Computes mean of values of y corresponding to bins of x
	
	Parameters :
		- x                 [xarray]                Time serie of the quantity of bins
		- y                 [xarray]                Time serie of the quantity to the mean
		- nbins             [int]                   Number of bins   
		
	Returns :
		- b                 [ndarray]               Array of bins values
		- m                 [ndarray]               Mean of y for each bin
		
	"""
	
	if x is None:
		raise ValueError("mean_bins requires at least 1 argument")
		
	if y is None:
		y = x
	
	if isinstance(x,xr.DataArray):
		x = x.data
		
	if isinstance(y,xr.DataArray):
		y = y.data
	
	xs      = np.sort(x)
	xedges  = np.linspace(xs[0],xs[-1],nbins+1)
	m       = np.zeros(nbins)

	for i in range(nbins):
		idxl    = x > xedges[i]
		idxr    = x < xedges[i+1]
		yb      = np.abs(y[idxl*idxr])
		m[i]    = np.mean(yb)
		
	bins = xedges[:-1]+np.median(np.diff(xedges))/2
	return (bins,m)
