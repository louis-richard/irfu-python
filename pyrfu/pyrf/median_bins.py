import numpy as np
import xarray as xr


def median_bins(x=None, y=None, nbins=10):
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
		>>> import numpy as np
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> mms_list = np.arange(1,5)
		>>> # Load magnetic field and electric field
		>>> b_mms = [mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id) for mms_id in mms_list]
		>>> r_mms = [mms.get_data("R_gse", tint, mms_id) for mms_id in mms_list]
		>>> # Compute current density, etc
		>>> j_xyz, div_b, b_xyz, jxb, div_t_shear, div_pb = pyrf.c_4_j(r_mms, b_mms)
		>>> # Compute magnitude of B and J
		>>> b_mag = pyrf.norm(b_xyz)
		>>> j_mag = pyrf.norm(j_xyz)
		>>> # Median value of |J| for 10 bins of |B|
		>>> med_b_j = pyrf.mean_bins(b_mag, j_mag)
		
	"""
	
	if x is None:
		raise ValueError("median_bins requires at least 1 argument")
		
	if y is None:
		y = x
	
	if isinstance(x, xr.DataArray):
		x = x.data
		
	if isinstance(y, xr.DataArray):
		y = y.data
	
	x_sort = np.sort(x)
	x_edge = np.linspace(x_sort[0], x_sort[-1], nbins + 1)

	m, s = [np.zeros(nbins), np.zeros(nbins)]

	for i in range(nbins):
		idx_l = x > x_edge[i]
		idx_r = x < x_edge[i+1]

		y_bins = np.abs(y[idx_l * idx_r])

		m[i], s[i] = [np.median(y_bins), np.std(y_bins)]
		
	bins = x_edge[:-1] + np.median(np.diff(x_edge)) / 2
	
	outdict = {"data": (["bins"], m), "sigma": (["bins"], s), "bins": bins}

	out = xr.Dataset(outdict)

	return out
