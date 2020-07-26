import numpy as np
import xarray as xr
from astropy.time import Time




def integrate(inp=None,time_step=None):
	"""
	Integrate time series

	Parameters :
		inp : DataArray
			Time series of the variable to integrate
	
	Options :
		time_step : float
			Time steps threshold. All time_steps larger than 3*time_step are assumed data gaps, default is that 
			time_step is the smallest value of all time_steps of the time series

	Returns :
		out : DataArray
			Time series of the time integrated input

	Example :
		>>> # Time interval
		>>> Tint = ["2015-12-14T01:17:40.200","2015-12-14T01:17:41.500"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field and electric field
		>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
		>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
		>>> # Convert electric field to field aligned coordinates
		>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
		

	"""

	if inp is None:
		raise ValueError("integrate requires at least one argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")


	ttemp = Time(inp.time.data,format="datetime64").unix
	datatemp = inp.data
	unitsTmp = inp.attrs["UNITS"]

	x = np.hstack([ttemp,data])

	dt = np.hstack([0,np.diff(x[:,0])])

	if time_step is None:
		time_steps  = np.diff(x[:,0])
		ind_min     = np.argmin(time_steps)
		time_step   = np.delete(time_steps,ind_min) # remove the smallest time step in case some problems
		time_step   = np.min(time_steps)


	dt[dt>3*time_step] = 0

	xint = x
	for j in range(1,xint.shape[1]):
		j_ok            = ~np.isnan(xint[:,j])
		xint[j_ok,j]    = np.cumsum(x[j_ok,j]*dt[j_ok])

	out = xr.DataArray(x[:,1:],coords=inp.coords,dims=inp.dims)
	out.attrs["UNITS"] = unitsTmp+"*s"

	return out