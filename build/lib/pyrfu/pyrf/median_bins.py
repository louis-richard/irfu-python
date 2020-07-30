import numpy as np
import xarray as xr


def median_bins(x=None,y=None,nbins=10):
	"""
	Computes median of values of y corresponding to bins of x
	
	Parameters :
		x : DataArray
			Time series of the quantity of bins

		y : DataArray
			Time series of the quantity to the median

		nbins : int
			Number of bins   
		
	Returns :
		out : Dataset
			Dataset with :
				* bins : DataArray
					bin values of the x variable

				* data : DataArray
					Median values of y corresponding to each bin of x
					
				* sigma : DataArray
					Standard deviation

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
		>>> # Median value of |J| for 10 bins of |B|
		>>> MedBJ = pyrf.mean_bins(Bmag,Jmag)
		
	"""
	
	if x is None:
		raise ValueError("median_bins requires at least 1 argument")
		
	if y is None:
		y = x
	
	if isinstance(x,xr.DataArray):
		x = x.data
		
	if isinstance(y,xr.DataArray):
		y = y.data
	
	xs      = np.sort(x)
	xedges  = np.linspace(xs[0],xs[-1],nbins+1)
	m       = np.zeros(nbins)
	s       = np.zeros(nbins)

	for i in range(nbins):
		idxl    = x > xedges[i]
		idxr    = x < xedges[i+1]
		yb      = np.abs(y[idxl*idxr])
		m[i]    = np.median(yb)
		s[i]    = np.std(yb)
		
	bins = xedges[:-1]+np.median(np.diff(xedges))/2
	
	outdict = {"data": (["bins"], m), "sigma": (["bins"], s),"bins":bins}

	out = xr.Dataset(outdict)
	return out
