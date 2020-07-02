import numpy as np

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz



def e_vxb(v=None, b=None, flag=0):
	"""
	Compute VxB and ExB/B^2

	Parameters :
		- v                 [xarray]                Velocity/Electric field
		- b                 [xarray]                Magnetic field
		- flag              [int]                   Flag. If vxb (default) computes electric field. If exb compute drift
													velocity
	
	Returns :
		- out               [xarray]                Electric/velocity field time serie

	"""
	if v is None:
		raise ValueError("e_vxb requires at least two arguments")

	if b is None:
		raise ValueError("e_vxb requires at least two arguments")
	
	attrs = {}
	estimateExB    = False
	estimateVxB    = True
	inputTSeries   = False
	inputNumeric   = False
	inputVConstant = False

	if flag == -1:
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
			res = np.cross(v.data,b.data)*(-1)*1e-3;

		attrs["UNITS"]      = "mV/s"
		attrs["FIELDNAM"]   = "Electric field"
		attrs["LABLAXIS"]   = "E"

	out = ts_vec_xyz(b.time.data,res,attrs)
	return out