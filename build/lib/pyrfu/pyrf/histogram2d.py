import numpy as np
import xarray as xr





def histogram2d(inp1=None,inp2=None,nbins=100):
	"""
	Computes 2d histogram of inp2 vs inp1 with nbins number of bins

	Parameters :
		- inp1              [xarray]                Time serie of the x values
		- inp2              [xarray]                Time serie of the y values
		- nbins             [int]                   Number of bins

	Returns :
		- out               [xarray]                Map of the density of inp2 vs inp1

	"""
	if inp1 is None or inp2 is None:
		raise ValueError("histogram2d requiers at least 2 arguments")

	if not isinstance(inp1,xr.DataArray) or not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")


	H, xedges, yedges = np.histogram2d(inp1.data,inp2.data,bins=nbins)

	x = xedges[:-1]+np.median(np.diff(xedges))/2
	y = yedges[:-1]+np.median(np.diff(yedges))/2

	out = xr.DataArray(H,coords=[x,y],dims=["xbins","ybins"])
	return out