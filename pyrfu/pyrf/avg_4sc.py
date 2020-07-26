import numpy as np
import xarray as xr

from .resample import resample


def avg_4sc(B=None):
	"""
	Computes the input quantity at the center of mass of the MMS tetrahedron

	Parameters :
		B : list of DataArray
			List of the time series of the quantity for each spacecraft

	Returns :
		Bavg : DataArray
			Time series of the input quantity a the enter of mass of the MMS tetrahedron

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
		>>> Bxyzavg = pyrf.avg_4sc(Bxyz)

	"""

	if B is None:
		raise ValueError("avg_4sc requires 1 argument")

	if not isinstance(B,list):
		raise TypeError("B must be a list of the 4 spacecraft data")

	for i in range(4):
		if not isinstance(B[i],xr.DataArray):
			raise TypeError("B[{:d}] must be a DataArray".format(i))

	B = [resample(b,B[0]) for b in B]

	Bavg = (B[0]+B[1]+B[2]+B[3])/4

	return Bavg