import numpy as np
import bisect
import xarray as xr

from ..pyrf.resample import resample
from ..pyrf.ts_vec_xyz import ts_vec_xyz
from ..pyrf.tlim import tlim
from ..pyrf.wavelet import wavelet

def fk_powerspec4SC(E=None ,R=None,B=None, Tints=None, cav=8, numk=500, numf=200, df=None, wwidth=1, frange=None):
	"""
	Calculates the frequency-wave number power spectrum using the four MMS spacecraft. Uses a generalization of 
	mms.fk_powerspectrum. Wavelet based cross-spectral analysis is used to calculate the phase difference each 
	spacecraft pair and determine 3D wave vector. A generalization of the method used in mms.fk_powerspectrum 
	to four point measurements. 

	Parameters :
		E : list of DataArray
			Fields to apply 4SC cross-spectral analysis to. E.g., E or B fields 
			(if multiple components only the first is used).

		R : list of DataArray
			Positions of the four spacecraft

		B : list of DataArray
			Background magnetic field in the same coordinates as R. Used to determine the parallel and perpendicular 
			wave numbers using 4SC average.

		Tints : list of str
			Time interval over which the power spectrum is calculated. To avoid boundary effects use a longer time 
			interval for E and B. 

	Options :
		cav : int
			Number of points in time series used to estimate phase. (default cav = 8)

		numk : int
			Number of wave numbers used in spectrogram. (default numk = 500)

		df : float
			Linear spacing of frequencies (default log spacing).

		numf : int
			Number of frequencies used in spectrogram. (default numf = 200)

		wwidth : float
			Multiplier for Morlet wavelet width. (default wwidth = 1)

		frange : list of float
			Frequency range for k-k plots. [minf maxf]

	Returns :
		out : Dataset
			Dataset of array of powers as a function of frequency and wavenumber. Power is normalized to the maximum 
			value.

	Notes: 
		Wavelength must be larger than twice the spacecraft separations, otherwise spatial aliasing will occur. 

	Example: 
		>>> Power = mms.fk_powerspec4SC(Epar,Rxyz,Bxyz,Tints)
		>>> Power = mms.fk_powerspec4SC(Bscmfacx,Rxyz,Bxyz,Tints,linear=10,numk=500,cav=4,wwidth=2)

	Example to plot:
		>>> fig, ax = plt.subplots(1)
		>>> ax, cax = pltrf.plot_spectr(ax,Power.kmagf,cscale="log",cmap="viridis")
		>>> ax.set_xlabel("$|k|$ [m$^{-1}$]")
		>>> ax.set_ylabel("$f$ [Hz]")

	"""

	if (E is None) or (R is None) or (B is None) or (Tints is None):
		raise ValueError("fk_powerspec4SC requires at least 4 arguments")


	ic = np.arange(1,5)

	#pdb.set_trace()

	E1, E2, E3, E4 = [resample(E[i-1],E[0]) for i in ic]
	R1, R2, R3, R4 = [resample(R[i-1],E[0]) for i in ic]
	B1, B2, B3, B4 = [resample(B[i-1],B[0]) for i in ic]


	Bav = ts_vec_xyz(B1.time.data,(B1.data+B2.data+B3.data+B4.data)/4)


	times = E1.time
	uselinear = not(df is None)

	idx = tlim(E1.time,Tints)

	# If odd, remove last data point (as is done in irf_wavelet)
	if len(idx)%2:
		idx = idx[:-1]

	if uselinear:
		W = [wavelet(e,linear=df,returnpower=False,wavelet_width=5.36*wwidth,plot=False) for e in [E1,E2,E3,E4]]
	else :
		W = [wavelet(e,nf=numf,returnpower=False,wavelet_width=5.36*wwidth,plot=False) for e in [E1,E2,E3,E4]]

	numf = len(W[0].frequency)

	times 	= tlim(times,Tints)
	L 		= len(times)

	W = [tlim(W[i],Tints) for i in range(4)]

	fkPower = 0
	for i in range(4):
		fkPower += W[i].data*np.conj(W[i].data)/4

	N 		= int(np.floor(L/cav)-1)
	posav 	= cav/2 + np.arange(N)*cav
	avtimes = times[posav.astype(int)]

	Bav 			= resample(Bav,avtimes)
	R1, R2, R3, R4 	= [resample(r,avtimes) for r in [R1, R2, R3, R4]]

	cx12 	= np.zeros((N,numf),dtype="complex128")
	cx13 	= np.zeros((N,numf),dtype="complex128")
	cx14 	= np.zeros((N,numf),dtype="complex128")
	cx23 	= np.zeros((N,numf),dtype="complex128")
	cx24 	= np.zeros((N,numf),dtype="complex128")
	cx34 	= np.zeros((N,numf),dtype="complex128")
	Powerav = np.zeros((N,numf),dtype="complex128")

	for m, posavm in enumerate(posav):
		lb, ub 			= [int(posavm-cav/2+1),int(posavm+cav/2)]
		cx12[m,:] 		= np.nanmean(W[0].data[lb:ub,:]*np.conj(W[1].data[lb:ub,:]),axis=0)
		cx13[m,:] 		= np.nanmean(W[0].data[lb:ub,:]*np.conj(W[2].data[lb:ub,:]),axis=0)
		cx14[m,:] 		= np.nanmean(W[0].data[lb:ub,:]*np.conj(W[3].data[lb:ub,:]),axis=0)
		cx23[m,:] 		= np.nanmean(W[1].data[lb:ub,:]*np.conj(W[2].data[lb:ub,:]),axis=0)
		cx24[m,:] 		= np.nanmean(W[1].data[lb:ub,:]*np.conj(W[3].data[lb:ub,:]),axis=0)
		cx34[m,:] 		= np.nanmean(W[2].data[lb:ub,:]*np.conj(W[3].data[lb:ub,:]),axis=0)
		Powerav[m,:] 	= np.nanmean(fkPower[lb:ub,:],axis=0)

	# Compute phase differences between each spacecraft pair
	th12 = np.arctan2(np.imag(cx12),np.real(cx12))
	th13 = np.arctan2(np.imag(cx13),np.real(cx13))
	th14 = np.arctan2(np.imag(cx14),np.real(cx14))
	th23 = np.arctan2(np.imag(cx23),np.real(cx23))
	th24 = np.arctan2(np.imag(cx24),np.real(cx24))
	th34 = np.arctan2(np.imag(cx34),np.real(cx34))


	wmat = 2*np.pi*np.tile(W[0].frequency.data,(N,1))

	# Convert phase difference to time delay
	dt12 = th12/wmat
	dt13 = th13/wmat
	dt14 = th14/wmat
	dt23 = th23/wmat
	dt24 = th24/wmat
	dt34 = th34/wmat

	# Weighted averaged time delay using all spacecraft pairs
	dt2 = 0.5*dt12 + 0.2*(dt13-dt23) + 0.2*(dt14-dt24) + 0.1*(dt14-dt34-dt23)
	dt3 = 0.5*dt13 + 0.2*(dt12+dt23) + 0.2*(dt14-dt34) + 0.1*(dt12+dt24-dt34)
	dt4 = 0.5*dt14 + 0.2*(dt12+dt24) + 0.2*(dt13+dt34) + 0.1*(dt12+dt23+dt34)
	#dt2 = dt12
	#dt3 = dt13
	#dt4 = dt14

	# Compute phase speeds
	R1 = R1.data
	R2 = R2.data
	R3 = R3.data
	R4 = R4.data

	kx = np.zeros((N,numf))
	ky = np.zeros((N,numf))
	kz = np.zeros((N,numf))

	# Volumetric tensor with SC1 as center.
	dR = np.reshape(np.hstack([R2,R3,R4]),(N,3,3))-np.reshape(np.tile(R1,(1,3)),(N,3,3))
	dR = np.transpose(dR,[0,2,1])
	# Delay tensor with SC1 as center.
	#dT = np.reshape(np.hstack([dt2,dt3,dt4]),(N,numf,3))
	dT = np.dstack([dt2,dt3,dt4])

	for ii in range(numf):
		m = np.linalg.solve(dR,np.squeeze(dT[:,ii,:]))

		kx[:,ii] = 2*np.pi*W[0].frequency[ii].data*m[:,0]
		ky[:,ii] = 2*np.pi*W[0].frequency[ii].data*m[:,1]
		kz[:,ii] = 2*np.pi*W[0].frequency[ii].data*m[:,2]

	kx /= 1e3
	ky /= 1e3
	kz /= 1e3
	kmag = np.linalg.norm(np.array([kx,ky,kz]),axis=0)


	Bavxmat = np.tile(Bav.data[:,0],(numf,1)).T
	Bavymat = np.tile(Bav.data[:,1],(numf,1)).T
	Bavzmat = np.tile(Bav.data[:,2],(numf,1)).T
	Bavabs = np.linalg.norm(Bav,axis=1)
	Bavabsmat = np.tile(Bavabs,(numf,1)).T

	kpar 	= (kx*Bavxmat + ky*Bavymat + kz*Bavzmat)/Bavabsmat
	kperp 	= np.sqrt(kmag**2 - kpar**2)

	kmax 	= np.max(kmag)*1.1
	kmin 	= -kmax
	kvec 	= np.linspace(-kmax,kmax,numk)
	kmagvec = np.linspace(0,kmax,numk);

	dkmag 	= kmax/numk
	dk 		= 2*kmax/numk


	# Sort power into frequency and wave vector
	print("notice : Computing power versus kx,f; ky,f, kz,f")
	powerkxf 	= np.zeros((numf,numk))
	powerkyf 	= np.zeros((numf,numk))
	powerkzf	= np.zeros((numf,numk))
	powerkmagf 	= np.zeros((numf,numk))

	for nn in range(numf):
		kxnumber 	= np.floor((kx[:,nn]-kmin)/dk).astype(int)
		kynumber 	= np.floor((ky[:,nn]-kmin)/dk).astype(int)
		kznumber 	= np.floor((kz[:,nn]-kmin)/dk).astype(int)
		knumber 	= np.floor((kmag[:,nn])/dkmag).astype(int)

		powerkxf[nn,kxnumber] 	+= np.real(Powerav[:,nn])
		powerkyf[nn,kynumber] 	+= np.real(Powerav[:,nn])
		powerkzf[nn,kznumber] 	+= np.real(Powerav[:,nn])
		powerkmagf[nn,knumber] 	+= np.real(Powerav[:,nn])

	#powerkxf[powerkxf == 0] 	= np.nan
	#powerkyf[powerkyf == 0] 	= np.nan
	#powerkzf[powerkzf == 0] 	= np.nan
	#powerkmagf[powerkmagf == 0] = np.nan

	powerkxf 	/= np.max(powerkxf)
	powerkyf 	/= np.max(powerkyf)
	powerkzf 	/= np.max(powerkzf)
	powerkmagf 	/= np.max(powerkmagf)

	#powerkxf[powerkxf < 1.0e-6] 		= 1e-6
	#powerkyf[powerkyf < 1.0e-6] 		= 1e-6
	#powerkzf[powerkzf < 1.0e-6] 		= 1e-6
	#powerkmagf[powerkmagf < 1.0e-6] 	= 1e-6

	freqs 	= W[0].frequency.data
	idxf 	= np.arange(numf)

	if not frange is None:
		idx_minfreq = bisect.bisect_left(np.min(frange))
		idx_maxfreq = bisect.bisect_left(np.max(frange))
		idxf 		= idxf[idx_minfreq:idx_maxfreq]

	print("notice : Computing power versus kx,ky; kx,kz; ky,kz\n")
	powerkxky 		= np.zeros((numk,numk))
	powerkxkz 		= np.zeros((numk,numk))
	powerkykz 		= np.zeros((numk,numk))
	powerkperpkpar 	= np.zeros((numk,numk))

	for nn in idxf:
		kxnumber 	= np.floor((kx[:,nn]-kmin)/dk).astype(int)
		kynumber 	= np.floor((ky[:,nn]-kmin)/dk).astype(int)
		kznumber 	= np.floor((kz[:,nn]-kmin)/dk).astype(int)
		kparnumber 	= np.floor((kpar[:,nn]-kmin)/dk).astype(int)
		kperpnumber = np.floor((kperp[:,nn])/dkmag).astype(int)

		powerkxky[kynumber,kxnumber] 			+= np.real(Powerav[:,nn])
		powerkxkz[kznumber,kxnumber] 			+= np.real(Powerav[:,nn])
		powerkykz[kznumber,kynumber] 			+= np.real(Powerav[:,nn])
		powerkperpkpar[kparnumber,kperpnumber] 	+= np.real(Powerav[:,nn])


	#powerkxky[powerkxky == 0] 			= np.nan
	#powerkxkz[powerkxkz == 0] 			= np.nan
	#powerkykz[powerkykz == 0] 			= np.nan
	#powerkperpkpar[powerkperpkpar == 0] = np.nan

	powerkxky 		/= np.max(powerkxky)
	powerkxkz 		/= np.max(powerkxkz)
	powerkykz 		/= np.max(powerkykz)
	powerkperpkpar 	/= np.max(powerkperpkpar)

	#powerkxky(powerkxky < 1.0e-6) 				= 1e-6
	#powerkxkz(powerkxkz < 1.0e-6) 				= 1e-6
	#powerkykz(powerkykz < 1.0e-6) 				= 1e-6
	#powerkperpkpar[powerkperpkpar < 1.0e-6] 	= 1e-6

	outdict = {}
	outdict["kxf"] 			= (["kx","f"], powerkxf.T)
	outdict["kyf"] 			= (["kx","f"], powerkyf.T)
	outdict["kzf"] 			= (["kx","f"], powerkzf.T)
	outdict["kmagf"] 		= (["kmag","f"], powerkmagf.T)

	outdict["kxky"] 		= (["kx","ky"], powerkxky.T)
	outdict["kxkz"] 		= (["kx","kz"], powerkxkz.T)
	outdict["kykz"] 		= (["ky","kz"], powerkykz.T)
	outdict["kperpkpar"] 	= (["kperp","kpar"], powerkperpkpar.T)

	outdict["kx"] 			= kvec
	outdict["ky"] 			= kvec
	outdict["kz"] 			= kvec
	outdict["kmag"] 		= kmagvec
	outdict["kperp"] 		= kmagvec
	outdict["kpar"] 		= kvec
	outdict["f"] 			= freqs

	out = xr.Dataset(outdict)


	return out