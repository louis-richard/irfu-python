import xarray as xr
import numpy as np
from scipy import signal
 


def filt(inp=None, fmin=0, fmax=1, n=-1):
	"""
	Filter input quantity

	Parameters :
		- inp               [xarray]                Quantity to filter
		- fmin              [float]                 Lower limit of the frequency range
		- fmax              [float]                 Upper limit of the frequency range
		- n                 [int]                   Order rof the elliptic filter

	Returns : 
		- out               [xarray]                Filtered signal

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