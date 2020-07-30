import xarray as xr
import numpy as np

from .resample import resample
from .ts_scalar import ts_scalar



def dot(inp1=None, inp2=None):
	"""
	Computes dot product of two fields

	Parameters : 
		inp1 : DataArray
			Time series of the first field X

		inp2 : DataArray
			Time series of the second field Y

	Returns :
		out : DataArray
			Time series of the dot product Z = X.Y

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> # Load magnetic field, electric field and spacecraft position
		>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
		>>> Exyz = [mms.get_data("E_gse_edp_fast_l2",Tint,i) for i in ic]
		>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
		>>> Jxyz, divB, B, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
		>>> # Compute the electric at the center of mass of the tetrahedron
		>>> Exyzavg = pyrf.avg_4sc(Exyz)
		>>> # Compute J.E dissipation
		>>> JE = pyrf.dot(Jxyz,Exyz)

	"""

	if (inp1 is None) or (inp2 is None):
		raise ValueError("dot requires 2 arguments")

	if not isinstance(inp1,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")

	inp2 = resample(inp2,inp1)

	outdata = np.sum(inp1.data*inp2.data,axis=1)

	out = ts_scalar(inp1.time.data,outdata)
	
	return out