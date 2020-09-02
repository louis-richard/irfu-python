import xarray as xr
import numpy as np


def movmean(inp=None, n_pts=100):
	"""
	Computes running average of the inp over npts points.
	
	Parameters :
		inp : DataArray
			Time series of the input variable

		n_pts : int
			Number of points to average over

	Returns :
		out : DataArray
			Time series of the input variable averaged over npts points

	Notes :
		Works also with 3D skymap distribution

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load ion pressure tensor
		>>> p_xyz_i = mms.get_data("Pi_gse_fpi_brst_l2", tint, mms_id)
		>>> # Running average the pressure tensor over 10s
		>>> fs = pyrf.calc_fs(p_xyz_i)
		>>>> p_xyz_i = pyrf.movmean(p_xyz_i,10 * fs)

	"""

	if inp is None:
		raise ValueError("movmean requires at least one argument")

	if not isinstance(inp, xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if not isinstance(n_pts, int):
		n_pts = np.floor(n_pts).astype(int)

	if n_pts % 2:
		n_pts -= 1

	# Computes moving average
	cum_sum = np.cumsum(inp.data, axis=0)
	out_dat = (cum_sum[n_pts:, ...] - cum_sum[:-n_pts, ...]) / n_pts

	coords = []

	for k in inp.dims:
		if k == "time": 
			coords.append(inp.coords[k][int(n_pts / 2):-int(n_pts / 2)])
		else: 
			coords.append(inp.coords[k].data)

	# Output in DataArray type
	out = xr.DataArray(out_dat, coords=coords, dims=inp.dims, attrs=inp.attrs)

	return out
