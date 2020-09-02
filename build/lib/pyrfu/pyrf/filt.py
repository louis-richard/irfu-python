import xarray as xr
import numpy as np
from scipy import signal


def filt(inp=None, fmin=0, fmax=1, n=-1):
	"""
	Filters input quantity

	Parameters :
		inp : DataArray
			Time series of the variable to filter

		fmin : float
			Lower limit of the frequency range

		fmax : float
			Upper limit of the frequency range

		n : int
			Order of the elliptic filter

	Returns : 
		out : DataArray
			Time series of the filtered signal

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> tint = ["2017-07-18T13:03:34.000", "2017-07-18T13:07:00.000"]
		>>> # Spacecraft index
		>>> mms_id = 1
		>>> # Load magnetic and electric fields
		>>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
		>>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)
		>>> # Convert E to field aligned coordinates
		>>> e_xyzfac = pyrf.convert_fac(e_xyz, b_xyz, [1,0,0])
		>>> # Bandpass filter E waveform
		>>> fmin = 4
		>>> e_xyzfac_hf = pyrf.filt(e_xyzfac, fmin, 0, 3)
		>>> e_xyzfac_lf = pyrf.filt(e_xyzfac, 0, fmin, 3)

	"""

	if type(inp) != xr.DataArray:
		raise TypeError('Input must be a DataArray')

	fs = 1 / (np.median(np.diff(inp.time)).astype(int) * 1e-9)

	# Data of the input
	inpdata = inp.data

	fmin, fmax = [fmin / (fs / 2), fmax / (fs / 2)]

	if fmax > 1:
		fmax = 1

	# Parameters of the elliptic filter. fact defines the width between stopband and passband
	r_p, r_s, fact = [0.5, 60, 1.1]

	if fmin == 0:
		b1, a1 = [None] * 2
		b2, a2 = [None] * 2

		if n == -1:
			n, fmax = signal.ellipord(fmax, np.min([fmax * fact, 0.9999]), r_p, r_s)

		b, a = signal.ellip(n, r_p, r_s, fmax, btype="lowpass")
	elif fmax == 0:
		b1, a1 = [None] * 2
		b2, a2 = [None] * 2

		if n == -1:
			n, fmin = signal.ellipord(fmin, np.min([fmin * fact, 0.9999]), r_p, r_s)

		b, a = signal.ellip(n, r_p, r_s, fmin, btype="highpass")
	else:
		b, a = [None] * 2

		if n == -1:
			n, fmax = signal.ellipord(fmax, np.min([fmax * 1.3, 0.9999]), r_p, r_s)

		b1, a1 = signal.ellip(n, r_p, r_s, fmax)

		if n == -1:
			n, fmin = signal.ellipord(fmin, fmin * .75, r_p, r_s)

		b2, a2 = signal.ellip(n, r_p, r_s, fmin)

	try:
		n_c = inpdata.shape[1]
	except IndexError:
		n_c = 1
		inpdata = inpdata[:, np.newaxis]

	outdata = np.zeros(inpdata.shape)

	if fmin != 0 and fmax != 0:
		for i_col in range(n_c):
			outdata[:, i_col] = signal.filtfilt(b1, a1, inpdata[:, i_col])
			outdata[:, i_col] = signal.filtfilt(b2, a2, outdata[:, i_col])
	else:
		for i_col in range(n_c):
			outdata[:, i_col] = signal.filtfilt(b, a, inpdata[:, i_col])

	if n_c == 1:
		outdata = outdata[:, 0]

	out = xr.DataArray(outdata, coords=inp.coords, dims=inp.dims, attrs=inp.attrs)
	
	return out
