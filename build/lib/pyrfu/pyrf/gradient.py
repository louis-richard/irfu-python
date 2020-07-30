import numpy as np
import xarray as xr



def gradient(inp=None):
	"""
	Computes time derivative of the input variable

	Parameters :
		inp : DataArray
			Time series of the input variable

	Returns :
		out : DataArray
			Time series of the time derivative of the input variable

	Example :
		>>> # Time interval
		>>> Tint = ["2017-07-18T13:03:34.000","2017-07-18T13:07:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field
		>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
		>>> # Time derivative of the magnetic field
		>>> dtB = pyrf.gradient(Bxyz)

	"""


	if inp is None:
		raise ValueError("gradient requires at least 1 argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")


	# guess time step
	dt = np.median(np.diff(inp.time.data)).astype(float)*1e-9

	dinpdt = np.gradient(inp.data,axis=0)/dt

	out = xr.DataArray(dinpdt,coords=inp.coords,dims=inp.dims, attrs=inp.attrs)
	

	if "UNITS" in out.attrs:
		out.attrs["UNITS"] += "/s"

	return out