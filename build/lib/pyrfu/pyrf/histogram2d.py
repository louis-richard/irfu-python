import numpy as np
import xarray as xr

from .resample import resample


def histogram2d(inp1=None, inp2=None, nbins=100):
	"""
	Computes 2d histogram of inp2 vs inp1 with nbins number of bins

	Parameters :
		inp1 : DataArray
			Time series of the x values

		inp2 : DataArray
			Time series of the y values
		
		nbins : int
			Number of bins

	Returns :
		out : DataArray
			2D map of the density of inp2 vs inp1

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
		>>> # Compute magnitude of B and J
		>>> Bmag = pyrf.norm(Bavg)
		>>> Jmag = pyrf.norm(J)
		>>> # Histogram of |J| vs |B|
		>>> HBJ = pyrf.histogram2d(Bmag,Jmag)

	"""

	if inp1 is None or inp2 is None:
		raise ValueError("histogram2d requiers at least 2 arguments")

	if not isinstance(inp1,xr.DataArray) or not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	# resample inp2 with respect to inp1
	if len(inp2) != len(inp1):
		inp2 = resample(inp2,inp1)

	H, xedges, yedges = np.histogram2d(inp1.data,inp2.data,bins=nbins)

	x = xedges[:-1]+np.median(np.diff(xedges))/2
	y = yedges[:-1]+np.median(np.diff(yedges))/2

	out = xr.DataArray(H,coords=[x,y],dims=["xbins","ybins"])
	return out