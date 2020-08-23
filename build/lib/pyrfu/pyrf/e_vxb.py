import numpy as np
import xarray as xr


from .resample import resample
from .ts_vec_xyz import ts_vec_xyz



def e_vxb(v=None, b=None, flag="vxb"):
	"""
	Computes the convection electric field VxB (default) or the ExB drift velocity ExB/B^2 (flag="exb")

	Parameters :
		v : DataArray
			Time series of the velocity/electric field

		b : DataArray
			Time series of the magnetic field

		flag : str
			Method flag : 
				"vxb" -> computes convection electric field (default)
				"exb" -> computes ExB drift velocity
	
	Returns :
		out : DataArray
			Time series of the convection electric field/ExB drift velocity

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field and electric field
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
		>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,1)
		>>> # Compute ExB drift velocity
		>>> ExB = pyrf.e_vxb(Exyz,Bxyz,"ExB")

	"""

	if (v is None) or (b is None):
		raise ValueError("e_vxb requires at least two arguments")

	if not isinstance(v,xr.DataArray):
		raise TypeError("v must be a DataArray")

	if not isinstance(b,xr.DataArray):
		raise TypeError("b must be a DataArray")

	
	attrs = {}
	estimateExB    = False
	estimateVxB    = True
	inputTSeries   = False
	inputNumeric   = False
	inputVConstant = False

	if flag.lower() == "exb":
		estimateExB    = True
		estimateVxB    = False

	if v.size == 3:
		inputVConstant = True

	if estimateExB:
		e = v
		if len(e) != len(b):
			b = resample(b,e)

		res = np.cross(e.data,b.data,axis=1)
		res = res/np.linalg.norm(b.data,axis=1)[:,None]**2*1e3

		attrs["UNITS"]      = "km/s"
		attrs["FIELDNAM"]   = "Velocity"
		attrs["LABLAXIS"]   = "V"

	elif estimateVxB:
		if inputVConstant :
			res = np.cross(np.tile(v,(len(b),1)),b.data)*(-1)*1e-3
			v = []

		else :
			if len(v) != len(b):
				b = resample(b,v)

			res = np.cross(v.data,b.data)*(-1)*1e-3;

		attrs["UNITS"]      = "mV/s"
		attrs["FIELDNAM"]   = "Electric field"
		attrs["LABLAXIS"]   = "E"

	out = ts_vec_xyz(b.time.data,res,attrs)
	return out