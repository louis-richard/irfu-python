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
		>>> # Time interval
		>>> Tint = ["2017-07-18T13:03:34.000","2017-07-18T13:07:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic and electric fields
		>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
		>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
		>>> # Convert E to field aligned coordinates
		>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
		>>> # Bandpass filter E waveform
		>>> fmin = 4
		>>> Exyzfachf = pyrf.filt(Exyzfac,fmin,0,3)
		>>> Exyzfaclf = pyrf.filt(Exyzfac,0,fmin,3)

	"""

	if type(inp) != xr.DataArray: raise TypeError('Input must be a DataArray')

	Fs      = 1/(np.median(np.diff(inp.time)).astype(int)*1e-9)
	inpdata = inp.data
	
	

	fmin    = fmin/(Fs/2)
	fmax    = fmax/(Fs/2)

	if fmax > 1: fmax = 1

	Rp = 0.5
	Rs = 60
	fact = 1.1    # fact defines the width between stopband and passband

	if fmin == 0:
		if n == -1:
			n, fmax = signal.ellipord(fmax,np.min([fmax*fact,0.9999]),Rp,Rs)
		B, A = signal.ellip(n,Rp,Rs,fmax,btype="lowpass")
	elif fmax == 0:
		if n == -1:
			n, fmin = signal.ellipord(fmin,np.min([fmin*fact,0.9999]),Rp,Rs);
		B, A = signal.ellip(n,Rp,Rs,fmin,btype="highpass")
	else :
		if n == -1:
			n, fmax = signal.ellipord(fmax,np.min([fmax*1.3,0.9999]),Rp,Rs)
		B1, A1 = signal.ellip(n,Rp,Rs,fmax)
		if n == -1:
			n, fmin = signal.ellipord(fmin,fmin*.75,Rp,Rs)
		B2, A2 = signal.ellip(n,Rp,Rs,fmin)

	try:
		nColumnsToFilter=inpdata.shape[1]
	except IndexError:
		nColumnsToFilter = 1
		inpdata = inpdata[:,np.newaxis]

	outdata = np.zeros(inpdata.shape)

	if fmin != 0 and fmax != 0:
		for iCol in range(nColumnsToFilter):
			outdata[:,iCol] = signal.filtfilt(B1,A1,inpdata[:,iCol])
			outdata[:,iCol] = signal.filtfilt(B2,A2,outdata[:,iCol])
	else:
		for iCol in range(nColumnsToFilter):
			outdata[:,iCol] = signal.filtfilt(B,A,inpdata[:,iCol])

	if nColumnsToFilter == 1:
		outdata = outdata[:,0]
	out = xr.DataArray(outdata,coords=inp.coords,dims=inp.dims,attrs=inp.attrs)
	
	return out