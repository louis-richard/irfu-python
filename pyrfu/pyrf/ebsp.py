import numpy as np
import xarray as xr
import pyfftw
from astropy.time import Time
import warnings
from tqdm import tqdm
import multiprocessing as np
import sfs

from .ts_vec_xyz import ts_vec_xyz
from .resample import resample
from .iso2unix import iso2unix
from .start import start
from .end import end
from .calc_fs import calc_fs
from .convert_fac import convert_fac


# TODO parrallelized AverageData

def AverageData(data=None,x=None,y=None,avWindow=None):
	# average data with time x to time y using window
	
	dtx = np.median(np.diff(x))
	dty = np.median(np.diff(y))

	if avWindow is None :
		avWindow = dty

	
	dt2         = avWindow/2
	ndataOut    = len(y)
	
	# Pad data with NaNs from each side
	nPointToAdd = int(np.ceil(dt2/dtx))
	padNan      = np.zeros((nPointToAdd,data.shape[1]))*np.nan
	data        = np.vstack([padNan,data,padNan])
	padTime     = dtx*np.arange(nPointToAdd)
	x           = np.hstack([x[0]-np.flip(padTime), x, x[-1]+padTime])

	out = np.zeros((ndataOut,data.shape[1]))
	
	for i, iy in enumerate(y):
		il = bisect.bisect_left(x,iy-dt2)
		ir = bisect.bisect_left(x,iy+dt2)
		out[i,:] = np.nanmean(data[il:ir,:],axis=0)
	
	return out


#---------------------------------------------------------------------------------------------------------------------
def ebsp(e=None,dB=None,fullB=None,B0=None,xyz=None,freq_int=None,**kwargs):
	"""
	Calculates wavelet spectra of E&B and Poynting flux using wavelets (Morlet wavelet). Also computes polarization 
	parameters of B using SVD. SVD is performed on spectral matrices computed from the time series of B using wavelets
	and then averaged over a number of wave periods.
	
	Parameters :
		- e 				[xarray] 				Wave electric field time series
		- dB 				[xarray]	 			Wave magnetic field time series
		- fullB 			[xarray] 				High resolution background magnetic field time series used for 
													E*B=0
		- B0 				[xarray] 				Background magnetic field time series used for field aligned 
													coordinates
		- xyz 				[xarray] 				Position time series of spacecraft used for field aligned 
													coordinates
		- freq_int 			[str/list/ndarray] 		Frequency interval : either "pc12", "pc35" or arbitrary interval 
													[fmin,fmax]

	Options : 
		- polarization 		[bool]					Compute polarization parameters 
													(default False)
		- noresamp 			[bool] 					No resampling, E and dB are given at the same timeline 
													(default False)
		- fac 				[bool] 					Use FAC coordinate system (defined by B0 and optionally xyz), 
													otherwise no coordinate system transformation is performed 
													(default False)

		- dEdotB_0     		[bool] 					Compute dEz from dB dot B = 0, uses fullB 
													(default False)
		- fullB_dB     		[bool]					dB contains DC field 
													(default False)
		- nAv           	[int] 					Number of wave periods to average 
													(default 8)
		- facMatrix 		[ndarray] 				Specify rotation matrix to FAC system 
													(default None)
		- mwidthcoef 		[float] 				Specify coefficient to multiple Morlet wavelet width by.
													(default 1)

	Returns : 
		- t           		[xarray] 				Time
		- f           		[xarray]				Frequency
		- bb          		[xarray]				B power spectrum (xx, yy, zz)
		- ee_ss 			[xarray] 				E power spectrum (xx+yy spacecraft coords, e.g. ISR2)
		- ee 				[xarray] 				E power spectrum (xx, yy, zz)
		- pf_xyz 			[xarray]				Poynting flux (xyz)
		- pf_rtp     		[xarray]				Poynting flux (r, theta, phi) [angles in degrees]
		- dop         		[xarray]				3D degree of polarization
		- dop2d       		[xarray]				2D degree of polarization in the polarization plane
		- planarity   		[xarray]				Planarity of polarization
		- ellipticity 		[xarray]				Ellipticity of polarization ellipse
		- k           		[xarray]				k-vector (theta, phi FAC) [angles in degrees]

	
	Examples :
		>>> res = pyrf.ebsp(e,b,B,B0,xyz,"pc12")
		>>> res = pyrf.ebsp(e,b,None,B0,xyz,"pc35",polarization=True,fullB_dB=True)
		>>> res = pyrf.ebsp(e,b,None,B0,xyz,"pc12",fullB_dB=True,dEdotB_0=True)

	See also: PL_EBSP, CONVERT_FAC

	This software was developed as part of the MAARBLE (Monitoring,
	Analyzing and Assessing Radiation Belt Energization and Loss)
	collaborative research project which has received funding from the
	European Community's Seventh Framework Programme (FP7-SPACE-2011-1)
	under grant agreement n. 284520.
	"""


	if not isinstance(dB, xr.DataArray):
		raise TypeError("dB must be a DataArray")

	if not isinstance(fullB, xr.DataArray):
		raise TypeError("fullB must be a DataArray")

	if not isinstance(B0, xr.DataArray):
		raise TypeError("B0 must be a DataArray")

	if not isinstance(xyz, xr.DataArray):
		raise TypeError("xyz must be a DataArray")

	# Check the input
	nWavePeriodToAverage    = 8         # Number of wave periods to average
	angleBElevationMax      = 15        # Below which we cannot apply E*B=0
	facMatrix               = None      # matrix for totation to FAC
	mwidthcoef              = 1
	wantPolarization        = False

	if e is None:
		wantEE = 0
	else :
		wantEE = 1

	res   = {"t"            : None, "f"           : None,           "flagFac"     : 0     ,\
			  "bb_xxyyzzss" : None, "ee_xxyyzzss" : None,           "ee_ss"       : None  ,\
			  "pf_xyz"      : None, "pf_rtp"      : None,           "dop"         : None  ,\
			  "dop2d"       : None, "planarity"   : None,           "ellipticity" : None  ,\
			  "k_tp"        : None, "fullB"       : fullB,          "B0"          : B0    ,\
			  "r"           : xyz}


	flag_no_resamp  = False
	flag_want_fac   = False
	flag_dEdotB0    = False
	flag_fullB_dB   = False


	if "polarization" in kwargs:
		wantPolarization = True

	if "mwidthcoef" in kwargs:
		if not kwargs["mwidthcoef"] is None  and isinstance(kwargs["mwidthcoef"],float):
			mwidthcoef = kwargs["mwidthcoef"]
		else :
			raise ValueError("parameter ''mwidthcoef'' without parameter value")

	if "noresamp" in kwargs:
		flag_no_resamp = True

	if "fac" in kwargs:
		flag_want_fac = True

	if "dedotb_0" in kwargs:
		flag_dEdotB0 = True

	if "fullb_db" in kwargs:
		flag_fullB_dB = True

	if "nav" in kwargs:
		if kwargs["nav"] is None or not isinstance(kwargs["nav"],int):
			raise TypeError("NAV requires  must be an integer")
			  
		nWavePeriodToAverage = kwargs["nav"]

	if "facmatrix" in kwargs:
		if kwargs["facmatrix"] is None or not isinstance(kwargs["facmatrix"],xr.DataArray) or not "time" in kwargs["facmatrix"].coords:
			raise ValueError("FACMATRIX requires a second argument struct(t,rotMatrix)")
			  
		facMatrix = kwargs["facmatrix"]


	if flag_want_fac and facMatrix is None:
		if B0 is None:
			raise ValueError("ebsp(): at least B0 should be given for option FAC")
		
		if xyz is None:
			print("fcal : assuming s/c position [1 0 0] for estimating FAC")
			xyz = [1,0,0]
			xyz = ts_vec_xyz(dB.time.data,np.tile(xyz,(len(dB,1))))

		xyz = resample(xyz,dB)

	B0 = resample(B0,dB)

	if flag_fullB_dB:
		fullB           = dB
		res["fullB"]    = fullB
		dB              = dB - B0

	if flag_dEdotB0 and fullB is None:
		raise ValueError("fullB must be given for option dEdotB=0")


	pc12_range  = 0
	pc35_range  = 0
	other_range = 0

	if isinstance(freq_int,str):
		if freq_int.lower() == "pc12":
			pc12_range  = 1
			freq_int    = [.1,5]
			deltaT      = 1
			tint        = list(Time(np.round([start(dB),end(dB)]),format="unix").iso)

		elif freq_int.lower() == "pc35":
			pc35_range  = 1
			freq_int    = [.002,.1]
			deltaT      = 60
			tint        = list(Time(np.round([start(dB),end(dB)]/60)*60,format="unix").iso)

		outSampling = 1/deltaT
		nt          = np.round((iso2unix(tint[1])-iso2unix(tint[0]))/deltaT).astype(int)
		outTime     = np.linspace(iso2unix(tint[0]),iso2unix(tint[1]),nt) + deltaT/2
		outTime     = outTime[:-1]
	else : 
		if freq_int[1] >= freq_int[0]:
			other_range   = True
			outSampling   = freq_int[1]/5
			deltaT        = 1/outSampling
			nt            = np.round((end(dB)-start(dB))/deltaT).astype(int)
			outTime       = np.linspace(start(dB),end(dB),nt) + deltaT/2
			outTime       = outTime[:-1]
		else :
			raise ValueError("FREQ_INT must be [f_min f_max], f_min<f_max")

	if wantEE :# Check the sampling rate
		if e is None:
			raise ValueError("E cannot be empty for the chosen output parameters")

		sampl_e = calc_fs(e)
		sampl_b = calc_fs(dB)
		if flag_no_resamp:
			if sampl_e != sampl_b:
				raise IndexError("E and B must have the same sampling for NORESAMP")
			elif len(e) != len(dB):
				raise IndexError("E and B must have the same number of points for NORESAMP")

			inSampling = sampl_e
		else :
			if sampl_b > 1.5*sampl_e:
				e   = resample(e,dB)
				B0  = resample(B0,dB)
				
				inSampling = sampl_b
				warnings.warn("Interpolating e to b",UserWarning)
			elif sampl_e > 1.5*sampl_b:
				dB = resample(dB,e)
				B0 = resample(B0,e)
				
				inSampling = sampl_e
				warnings.warn("Interpolating b to e",UserWarning)
			elif sampl_e == sampl_b and len(e) == len(dB):
				inSampling = sampl_e
			else :
				inSampling = 2*sampl_e
				
				nt  = (np.min([end(e),end(dB)])-np.max([start(e),start(dB)]))/(1/inSampling)
				t   = np.linspace(np.max([start(e),start(dB)]),np.min([end(e),end(dB)]),int(nt))
				
				t   = ts_time(t) 
				
				e       = resample(e,t)
				dB      = resample(dB,t)
				B0      = resample(B0,t)
				fullB   = resample(fullB,t)
				warnings.warn("Interpolating b and e to 2x e sampling",UserWarning)
				
		
		print("Fs = {:4.2f}, Fs_e = {:4.2f}, Fs_b = {:4.2f}".format(inSampling,sampl_e,sampl_b))
		
	  
	else :
		inSampling = calc_fs(dB)
		e = None

	if inSampling/2<freq_int[1]:
	  raise ValueError("F_MAX must be lower than the Nyquist frequecy")

	if wantEE and e.shape[1] < 3 and not flag_dEdotB0:
	  raise ValueError("E must have all 3 components or flag ''dEdotdB=0'' must be given")


	if len(dB)%2 :
		dB = dB[:-1,:]
		B0 = B0[:-1,:]

		if facMatrix is None:
			xyz = xyz[:-1,:]
		else :
			facMatrix["t"]          = facMatrix["t"][:-1,:]
			facMatrix["rotMatrix"]  = facMatrix["rotMatrix"][:-1,:,:]

		if wantEE:
			e = e[:-1,:]

	inTime = dB.time.data.view("i8")*1e-9

	Bx = None
	By = None
	Bz = None
	idxBparSpinPlane = None

	if flag_dEdotB0:
		Bx = fullB[:,0].data  # Needed for parfor
		By = fullB[:,1].data
		Bz = fullB[:,2].data

		# Remove the last sample if the total number of samples is odd
		if len(fullB)%2 :
			Bx = Bx[:-1,:]
			By = By[:-1,:]
			Bz = Bz[:-1,:]

		angleBElevation     = np.arctan(Bz/np.sqrt(Bx**2+By**2))*180/np.pi
		idxBparSpinPlane    = np.abs(angleBElevation)<angleBElevationMax


	# If E has all three components, transform E and B waveforms to a magnetic field aligned coordinate (FAC)
	# and save eISR for computation of ESUM. Ohterwise we compute Ez within the main loop and do the
	# transformation to FAC there.

	timeB0 = 0
	if flag_want_fac:
		res["flagFac"] = True
		timeB0 = B0.time.data.view("i8")*1e-9
		if wantEE:
			if  not flag_dEdotB0:
				eISR2 = e[:,:2]
				if e.shape[1] < 3:
					raise TypeError("E must be a 3D vector to be rotated to FAC")

				if facMatrix is None:
					e = convert_fac(e,B0,xyz)
				else:
					e = convert_fac(e,facMatrix)
		
		if facMatrix is None:
			dB = convert_fac(dB,B0,xyz)
		else:
			dB = convert_fac(dB,facMatrix)


	# Find the frequencies for an FFT of all data and set important parameters
	nd2     = len(inTime)/2
	nyq     = 1/2
	freq    = inSampling*np.arange(nd2)/(nd2)*nyq
	w       = np.hstack([0,freq,-np.flip(freq[:-1])])    # The frequencies corresponding to FFT

	Morlet_width    = 5.36*mwidthcoef
	freq_number     = np.ceil((np.log10(freq_int[1]) - np.log10(freq_int[0]))*12*mwidthcoef) #to get proper overlap for Morlet
	amin            = np.log10(0.5*inSampling/freq_int[1])
	amax            = np.log10(0.5*inSampling/freq_int[0])
	anumber         = freq_number
	# amin           = 0.01      # The highest frequency to consider is 0.5*sampl/10^amin
	# amax           = 2         # The lowest frequency to consider is 0.5*sampl/10^amax
	# anumber        = 400    # The number of frequencies

	a = np.logspace(amin,amax,int(anumber))
	# a = np.logspace(0.01,2.4,100)

	w0      = inSampling/2      # The maximum frequency
	# sigma  = 5.36/w0           # The width of the Morlet wavelet
	sigma   = Morlet_width/w0   # The width of the Morlet wavelet
	
	# Make the FFT of all data
	idxNanB             = np.isnan(dB.data)
	dB.data[idxNanB]    = 0
	Swb                 = pyfftw.interfaces.numpy_fft.fft(dB,axis=0,threads=mp.cpu_count())

	Swe         = None
	idxNanE     = None
	SweISR2     = None
	idxNanEISR2 = None # Needed for parfor

	if wantEE:
		print("ebsp ... calculate E and B wavelet transform ... ")
		idxNanE             = np.isnan(e.data)
		e.data[idxNanE]     = 0
		Swe                 = pyfftw.interfaces.numpy_fft.fft(e,axis=0,threads=mp.cpu_count())

		if flag_want_fac and not flag_dEdotB0:
			idxNanEISR2                 = np.isnan(eISR2.data)
			eISR2.data[idxNanEISR2]     = 0
			SweISR2                     = pyfftw.interfaces.numpy_fft.fft(eISR2,axis=0,threads=mp.cpu_count())
	else :
		print("ebsp ... calculate B wavelet transform ....")

	# Loop through all frequencies
	ndata       = len(inTime)
	nfreq       = len(a)
	ndataOut    = len(outTime)

	powerEx_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	powerEy_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	powerEz_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	power2E_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	power2E_ISR2_plot       = np.zeros((ndata,nfreq),dtype="complex128")
	power2B_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	powerBx_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	powerBy_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	powerBz_plot            = np.zeros((ndata,nfreq),dtype="complex128")
	S_plot_x                = np.zeros((ndata,nfreq))
	S_plot_y                = np.zeros((ndata,nfreq))
	S_plot_z                = np.zeros((ndata,nfreq))
	planarity               = np.zeros((ndataOut,nfreq))
	ellipticity             = np.zeros((ndataOut,nfreq))
	degreeOfPolarization3D  = np.zeros((ndataOut,nfreq),dtype="complex128")
	degreeOfPolarization2D  = np.zeros((ndataOut,nfreq),dtype="complex128")
	thetaSVD_fac            = np.zeros((ndataOut,nfreq))
	phiSVD_fac              = np.zeros((ndataOut,nfreq))

	# Get the correct frequencies for the wavelet transform
	frequencyVec    = w0/a
	censur          = np.floor(2*a*outSampling/inSampling*nWavePeriodToAverage)
	
	#---------------------------------------------------------------------------------------------------------------------
	#---------------------------------------------------------------------------------------------------------------------
	
	"""
	pdb.set_trace()
	mpi_args = (frequencyVec,nWavePeriodToAverage,outSampling,sigma,a,w,w0,Swb,Swe,flag_want_fac,flag_dEdotB0,\
			Bx,By,Bz,B0,xyz,facMatrix,inTime,outTime,idxNanB,idxNanE,idxNanEISR2,idxBparSpinPlane,wantEE,wantPolarization,SweISR2)
	pool = mp.Pool(mp.cpu_count())
	pool.starmap(my_func, [(*mpi_args,ind_a) for ind_a in range(2)])
	"""
	# begin for
	# inputs (frequencyVec,nWavePeriodToAverage,outSampling,sigma,w,w0,Swb,)

	for ind_a in tqdm(range(len(a))):
	#for ind_a in range(len(a)): # Main loop over frequencies
		#disp([num2str(ind_a) '. frequency, ' num2str(newfreq(ind_a)) ' Hz.']);

		## resample to 1 second sampling for Pc1-2 or 1 minute sampling for Pc3-5
		# average top frequencies to 1 second/1 minute
		# below will be an average over 8 wave periods. first find where one
		# sample is less than eight wave periods
		if frequencyVec[ind_a]/nWavePeriodToAverage > outSampling :
			avWindow = 1/outSampling
		else :
			avWindow = nWavePeriodToAverage/frequencyVec[ind_a]


		# Get the wavelet transform by IFFT of the FFT
		mWexp     = np.exp(-sigma*sigma*((a[ind_a]*w-w0)**2)/2)
		mWexp2    = np.tile(mWexp,(2,1)).T
		mWexp     = np.tile(mWexp,(3,1)).T
		
		Wb          = pyfftw.interfaces.numpy_fft.ifft(np.sqrt(1)*Swb*mWexp,axis=0,threads=mp.cpu_count())
		Wb[idxNanB] = np.nan

		We          = None
		WeISR2      = None
		if wantEE :
			if Swe.shape[1] == 2 :
				We = pyfftw.interfaces.numpy_fft.ifft(np.sqrt(1)*Swe*mWexp2,axis=0,threads=mp.cpu_count())
			else :
				We = pyfftw.interfaces.numpy_fft.ifft(np.sqrt(1)*Swe*mWexp,axis=0,threads=mp.cpu_count())

			We[idxNanE] = np.nan

			if flag_want_fac and not flag_dEdotB0 :
				WeISR2              = pyfftw.interfaces.numpy_fft.ifft(np.sqrt(1)*SweISR2*mWexp2,axis=0,threads=mp.cpu_count())
				WeISR2[idxNanEISR2] = np.nan

		newfreqmat=w0/a[ind_a]
		## Power spectrum of E and Poynting flux
		if wantEE : 
			# Power spectrum of E, power = (2*pi)*conj(W).*W./newfreqmat
			if flag_want_fac and not flag_dEdotB0 :
				SUMpowerEISR2 = np.sum(2*np.pi*(WeISR2*np.conj(WeISR2))/newfreqmat,axis=1)
			else :
				SUMpowerEISR2 = np.sum(2*np.pi*(We*np.conj(We))/newfreqmat,axis=1)
			end

			power2E_ISR2_plot[:,ind_a] = SUMpowerEISR2

			if flag_dEdotB0 :# Compute Ez from dE * B = 0
				rWe = np.real(We)
				iWe = np.imag(We)
				wEz = -(rWe[:,0]*Bx+rWe[:,1]*By)/Bz-1j*(iWe[:,0]*Bx+iWe[:,1]*By)/Bz
				wEz[idxBparSpinPlane] = np.nan
				if flag_want_fac :
					if facMatrix is None :
						We = convert_fac(ts_vec_xyz(timesB0,np.hstack([We[:,:2],wEz])),B0,xyz)
					else :
						We = convert_fac(ts_vec_xyz(timesB0,np.hstack([We[:,:2],wEz])),facMatrix)
					
					We = We[:,1:]
				else :
					We = np.hstack([We[:,:2],wEz])
			
			powerE = 2*np.pi*(We*np.conj(We))/newfreqmat
			powerE = np.vstack([powerE.T,np.sum(powerE,axis=1)]).T

			powerEx_plot[:,ind_a] = powerE[:,0]
			powerEy_plot[:,ind_a] = powerE[:,1]
			powerEz_plot[:,ind_a] = powerE[:,2]
			power2E_plot[:,ind_a] = powerE[:,3]

			# Poynting flux calculations, assume E and b units mV/m and nT, get  S in uW/m^2
			coef_poynt = 10/4/np.pi*(1/4)*(4*np.pi); # 4pi from wavelets, see A. Tjulins power estimates a few lines above
			
			S = np.zeros((ndata,3))

			Wex = We[:,0]
			Wey = We[:,1]
			Wez = We[:,2]

			Wbx = Wb[:,0]
			Wby = Wb[:,1]
			Wbz = Wb[:,2]

			S[:,0] = coef_poynt*np.real(Wey*np.conj(Wbz)+np.conj(Wey)*Wbz-Wez*np.conj(Wby)-np.conj(Wez)*Wby)/newfreqmat
			S[:,1] = coef_poynt*np.real(Wez*np.conj(Wbx)+np.conj(Wez)*Wbx-Wex*np.conj(Wbz)-np.conj(Wex)*Wbz)/newfreqmat
			S[:,2] = coef_poynt*np.real(Wex*np.conj(Wby)+np.conj(Wex)*Wby-Wey*np.conj(Wbx)-np.conj(Wey)*Wbx)/newfreqmat

			S_plot_x[:,ind_a] = S[:,0]
			S_plot_y[:,ind_a] = S[:,1]
			S_plot_z[:,ind_a] = S[:,2]
	  
	  #---------------------------------------------------------------------------------------------------------------------
	  
		## Power spectrum of B
		powerB = 2*np.pi*(Wb*np.conj(Wb))/newfreqmat
		powerB = np.vstack([powerB.T,np.sum(powerB,axis=1)]).T

		powerBx_plot[:,ind_a] = powerB[:,0]
		powerBy_plot[:,ind_a] = powerB[:,1]
		powerBz_plot[:,ind_a] = powerB[:,2]
		power2B_plot[:,ind_a] = powerB[:,3]
		
		if wantPolarization : # Polarization parameters
			## Construct spectral matrix and average it
			SM          = np.zeros((3,3,ndata),dtype="complex128")
			SM[0,0,:]   = 2*np.pi*(Wb[:,0]*np.conj(Wb[:,0]))/newfreqmat
			SM[0,1,:]   = 2*np.pi*(Wb[:,0]*np.conj(Wb[:,1]))/newfreqmat
			SM[0,2,:]   = 2*np.pi*(Wb[:,0]*np.conj(Wb[:,2]))/newfreqmat
			SM[1,0,:]   = 2*np.pi*(Wb[:,1]*np.conj(Wb[:,0]))/newfreqmat
			SM[1,1,:]   = 2*np.pi*(Wb[:,1]*np.conj(Wb[:,1]))/newfreqmat
			SM[1,2,:]   = 2*np.pi*(Wb[:,1]*np.conj(Wb[:,2]))/newfreqmat
			SM[2,0,:]   = 2*np.pi*(Wb[:,2]*np.conj(Wb[:,0]))/newfreqmat
			SM[2,1,:]   = 2*np.pi*(Wb[:,2]*np.conj(Wb[:,1]))/newfreqmat
			SM[2,2,:]   = 2*np.pi*(Wb[:,2]*np.conj(Wb[:,2]))/newfreqmat
			SM          = np.transpose(SM,[2,0,1])

			avSM = np.zeros((ndataOut,3,3),dtype="complex128") # Averaged SM
			
			for comp in range(3):
				avSM[...,comp] = AverageData(SM[...,comp],inTime,outTime,avWindow)
			#---------------------------------------------------------------------------------------------------------------------    
			# Remove data possibly influenced by edge effects
			
			censurIdx           = np.hstack([np.arange(np.min([censur[ind_a],len(outTime)])),\
												np.arange(np.max([0,len(outTime)-censur[ind_a]-1]),len(outTime))])
			censurIdx           = censurIdx.astype(int)
			avSM[censurIdx,...] = np.nan

			## compute singular value decomposition
			A = np.zeros((6,3,ndataOut)) #real matrix which is superposition of real part of spectral matrix over imaginary part
			U = np.zeros((6,3,ndataOut))
			W = np.zeros((3,3,ndataOut))
			V = np.zeros((3,3,ndataOut))
			#wSingularValues = zeros(3,ndata2);
			#R = zeros(3,3,ndata2); #spectral matrix in coordinate defined by V axes
			A[:3,...]   = np.real(np.transpose(avSM,[1,2,0]))
			A[3:6,...]  = -np.imag(np.transpose(avSM,[1,2,0]))

			for i in range(ndataOut):
				
				if np.isnan(A[...,i]).any() :
					U[...,i] = np.nan
					W[...,i] = np.nan
					V[...,i] = np.nan
				else : 
					
					[U[...,i],W[...,i],V[...,i]] = np.linalg.svd(A[...,i],full_matrices=False)
			
			# compute direction of propogation
			signKz      = np.sign(V[2,2,:])
			V[2,2,:]    = V[2,2,:]*signKz
			V[1,2,:]    = V[1,2,:]*signKz
			V[0,2,:]    = V[0,2,:]*signKz

			thetaSVD_fac[:,ind_a]   = np.abs(np.squeeze(np.arctan(np.sqrt(V[0,2,:]**2+V[1,2,:]**2)/V[2,2,:])*180/np.pi))    #ok<PFOUS>
			phiSVD_fac[:,ind_a]     = np.squeeze(np.arctan2(V[1,2,:],V[0,2,:])*180/np.pi)                                   #ok<PFOUS>

			## Calculate polarization parameters
			planarityLocal              = np.squeeze(1-np.sqrt(W[2,2,:]/W[0,0,:]))
			planarityLocal[censurIdx]   = np.nan
			planarity[:,ind_a]          = planarityLocal

			#ellipticity: ratio of axes of polarization ellipse axes*sign of polarization
			ellipticityLocal            = np.squeeze(W[1,1,:]/W[0,0,:])*np.sign(np.imag(avSM[:,0,1]))
			ellipticityLocal[censurIdx] = np.nan
			ellipticity[:,ind_a]        = ellipticityLocal
			#---------------------------------------------------------------------------------------------------------------------    
			# DOP = sqrt[(3/2.*trace(SM^2)./(trace(SM))^2 - 1/2)]; Samson, 1973, JGR
			dop = np.sqrt((3/2)*(np.trace(np.matmul(avSM,avSM),axis1=1,axis2=2)/np.trace(avSM,axis1=1,axis2=2)**2)-1/2)

			dop[censurIdx]                  = np.nan
			degreeOfPolarization3D[:,ind_a] = dop



			# DOP in 2D = sqrt[2*trace(rA^2)/trace(rA)^2 - 1)]; Ulrich
			Vnew = np.transpose(V,[2,0,1])

			avSM2dim    = np.matmul(Vnew,np.matmul(avSM,np.transpose(Vnew,[0,2,1])))
			avSM2dim    = avSM2dim[:,:2,:2]
			avSM        = avSM2dim;
			dop2dim     = np.sqrt(2*(np.trace(np.matmul(avSM,avSM),axis1=1,axis2=2)/np.trace(avSM,axis1=1,axis2=2)**2)-1)

			dop2dim[censurIdx]              = np.nan
			degreeOfPolarization2D[:,ind_a] = dop
	#---------------------------------------------------------------------------------------------------------------------
	#---------------------------------------------------------------------------------------------------------------------
	
	

	# set data gaps to NaN and remove edge effects
	censur = np.floor(2*a)
	for ind_a in range(len(a)):
		censurIdx = np.hstack([np.arange(np.min([censur[ind_a],len(inTime)])),\
								np.arange(np.max([1,len(inTime)-censur[ind_a]]),len(inTime))])
		
		
		censurIdx = censurIdx.astype(int)
		powerBx_plot[censurIdx,ind_a] = np.nan
		powerBy_plot[censurIdx,ind_a] = np.nan
		powerBz_plot[censurIdx,ind_a] = np.nan
		power2B_plot[censurIdx,ind_a] = np.nan
		if wantEE:
			powerEx_plot[censurIdx,ind_a]       = np.nan
			powerEy_plot[censurIdx,ind_a]       = np.nan
			powerEz_plot[censurIdx,ind_a]       = np.nan
			power2E_plot[censurIdx,ind_a]       = np.nan
			power2E_ISR2_plot[censurIdx,ind_a]  = np.nan
			S_plot_x[censurIdx,ind_a]           = np.nan
			S_plot_y[censurIdx,ind_a]           = np.nan
			S_plot_z[censurIdx,ind_a]           = np.nan


	# remove edge effects from data gaps
	idxNanE     = np.sum(idxNanE,axis=1)>0
	idxNanB     = np.sum(idxNanB,axis=1)>0
	idxNanEISR2 = np.sum(idxNanEISR2,axis=1)>0

	ndata2 = len(power2B_plot)
	if pc12_range or other_range:
		censur3 = np.floor(1.8*a)

	if pc35_range:
		censur3 = np.floor(.4*a)

	for i in range(len(idxNanB)-1):
		if idxNanB[i] < idxNanB[i+1]:
			for j in range(len(a)):
				censur_index_front = np.arange(np.max([i-censur3[j],0]),i)

				powerBx_plot[censur_index_front,j]  = np.nan
				powerBy_plot[censur_index_front,j]  = np.nan
				powerBz_plot[censur_index_front,j]  = np.nan
				power2B_plot[censur_index_front,j]  = np.nan
				S_plot_x[censur_index_front,j]      = np.nan
				S_plot_y[censur_index_front,j]      = np.nan
				S_plot_z[censur_index_front,j]      = np.nan

		if idxNanB[i] > idxNanB[i+1]:
			for j in range(len(a)):
				censur_index_back = np.arange(i,np.min([i+censur3[j],ndata2]))

				powerBx_plot[censur_index_back,j]   = np.nan
				powerBy_plot[censur_index_back,j]   = np.nan
				powerBz_plot[censur_index_back,j]   = np.nan
				power2B_plot[censur_index_back,j]   = np.nan
				S_plot_x[censur_index_back,j]       = np.nan
				S_plot_y[censur_index_back,j]       = np.nan
				S_plot_z[censur_index_back,j]       = np.nan


	ndata3 = len(power2E_plot)

	for i in range(len(idxNanE)-1):
		if idxNanE[i] < idxNanE[i+1]:
			for j in range(len(a)):
				censur_index_front = np.arange(np.max([i-censur3[j],1]),i)

				powerEx_plot[censur_index_front,j]      = np.nan
				powerEy_plot[censur_index_front,j]      = np.nan
				powerEz_plot[censur_index_front,j]      = np.nan
				power2E_plot[censur_index_front,j]      = np.nan
				power2E_ISR2_plot[censur_index_front,j] = np.nan
				S_plot_x[censur_index_front,j]          = np.nan
				S_plot_y[censur_index_front,j]          = np.nan
				S_plot_z[censur_index_front,j]          = np.nan

		if idxNanE[i] > idxNanE[i+1]:
			for j in range(len(a)):
				censur_index_back = np.arange(i,np.min([i+censur3[j],ndata3]))

				powerEx_plot[censur_index_back,j]       = np.nan
				powerEy_plot[censur_index_back,j]       = np.nan
				powerEz_plot[censur_index_back,j]       = np.nan
				power2E_plot[censur_index_back,j]       = np.nan
				power2E_ISR2_plot[censur_index_back,j]  = np.nan
				S_plot_x[censur_index_back,j]           = np.nan
				S_plot_y[censur_index_back,j]           = np.nan
				S_plot_z[censur_index_back,j]           = np.nan


	ndata4 = len(power2E_ISR2_plot)

	for i in range(len(idxNanEISR2)-1):
		if idxNanEISR2[i] < idxNanEISR2[i+1]:
			for j in range(len(a)):
				censur_index_front = np.arange(np.max([i-censur3[j],0]),i)

				power2E_ISR2_plot[censur_index_front,j] = np.nan

		if idxNanEISR2[i] > idxNanEISR2[i+1]:
			for j in range(len(a)):
				censur_index_back = np.arange(i,np.min([i+censur3[j],ndata4]))

				power2E_ISR2_plot[censur_index_back,j]  = np.nan

	#
	powerBx_plot    = AverageData(powerBx_plot,inTime,outTime)
	powerBy_plot    = AverageData(powerBy_plot,inTime,outTime)
	powerBz_plot    = AverageData(powerBz_plot,inTime,outTime)
	power2B_plot    = AverageData(power2B_plot,inTime,outTime)
	
	
	bb_xxyyzzss         = np.tile(powerBx_plot,(4,1,1))
	bb_xxyyzzss         = np.transpose(bb_xxyyzzss,[1,2,0])
	bb_xxyyzzss[:,:,1]  = powerBy_plot
	bb_xxyyzzss[:,:,2]  = powerBz_plot
	bb_xxyyzzss[:,:,3]  = power2B_plot
	bb_xxyyzzss         = bb_xxyyzzss.astype(float)


	# Output
	res["t"]            = Time(outTime,format="unix").datetime64
	res["f"]            = frequencyVec
	res["bb_xxyyzzss"]  = xr.DataArray(bb_xxyyzzss,coords=[res["t"],res["f"],["xx","yy","zz","ss"]],dims=["time","frequency","comp"])

	
	if wantEE:
		powerEx_plot      = AverageData(powerEx_plot,inTime,outTime)
		powerEy_plot      = AverageData(powerEy_plot,inTime,outTime)
		powerEz_plot      = AverageData(powerEz_plot,inTime,outTime)
		power2E_plot      = AverageData(power2E_plot,inTime,outTime)
		power2E_ISR2_plot = AverageData(power2E_ISR2_plot,inTime,outTime)
		power2E_ISR2_plot = power2E_ISR2_plot.astype(float)

		S_plot_x = AverageData(S_plot_x,inTime,outTime)
		S_plot_y = AverageData(S_plot_y,inTime,outTime)
		S_plot_z = AverageData(S_plot_z,inTime,outTime)
		[S_azimuth,S_elevation,S_r] = sfs.util.cart2sph(S_plot_x,S_plot_y,S_plot_z)
		
		ee_xxyyzzss         = np.tile(powerEx_plot,(4,1,1))
		ee_xxyyzzss         = np.transpose(ee_xxyyzzss,[1,2,0])
		ee_xxyyzzss[:,:,1]  = powerEy_plot
		ee_xxyyzzss[:,:,2]  = powerEz_plot
		ee_xxyyzzss[:,:,3]  = power2E_plot
		ee_xxyyzzss         = ee_xxyyzzss.astype(float)

		Poynting_XYZ        = np.tile(S_plot_x,(3,1,1))
		Poynting_XYZ        = np.transpose(Poynting_XYZ,[1,2,0])
		Poynting_XYZ[:,:,1] = S_plot_y
		Poynting_XYZ[:,:,2] = S_plot_z
		Poynting_XYZ        = Poynting_XYZ.astype(float)

		Poynting_RThPh          = np.tile(S_r,(3,1,1))
		Poynting_RThPh          = np.transpose(Poynting_RThPh,[1,2,0])
		Poynting_RThPh[...,1]   = np.pi/2-S_elevation
		Poynting_RThPh[...,2]   = S_azimuth
		Poynting_RThPh[...,1:]  = Poynting_RThPh[...,1:]*180/np.pi
		Poynting_RThPh          = Poynting_RThPh.astype(float)

		# Output
		res["ee_ss"]          = power2E_ISR2_plot.astype(float)
		res["ee_xxyyzzss"]    = xr.DataArray(ee_xxyyzzss,coords=[res["t"],res["f"],["xx","yy","zz","ss"]],dims=["time","frequency","comp"])
		res["pf_xyz"]         = xr.DataArray(Poynting_XYZ,coords=[res["t"],res["f"],["x","y","z"]],dims=["time","frequency","comp"])
		res["pf_rtp"]         = xr.DataArray(Poynting_RThPh,coords=[res["t"],res["f"],["rho","theta","phi"]],dims=["time","frequency","comp"])

	if wantPolarization :
	  # Define parameters for which we cannot compute the wave vector
	  indLowPlanarity   = planarity < 0.5
	  indLowEllipticity = np.abs(ellipticity) < .2
	  
	  thetaSVD_fac[indLowPlanarity] = np.nan
	  phiSVD_fac[indLowPlanarity]   = np.nan
	  
	  thetaSVD_fac[indLowEllipticity]   = np.nan
	  phiSVD_fac[indLowEllipticity]     = np.nan
	  
	  k_ThPhSVD_fac			= np.zeros((thetaSVD_fac.shape[0],thetaSVD_fac.shape[1],2))
	  k_ThPhSVD_fac[...,0]  = thetaSVD_fac
	  k_ThPhSVD_fac[...,1]  = phiSVD_fac
	  
	  # Output
	  res["dop"]            = xr.DataArray(np.real(degreeOfPolarization3D),coords=[res["t"],res["f"]],dims=["time","frequency"])
	  res["dop2d"]          = xr.DataArray(np.real(degreeOfPolarization2D),coords=[res["t"],res["f"]],dims=["time","frequency"])
	  res["planarity"]      = xr.DataArray(planarity,coords=[res["t"],res["f"]],dims=["time","frequency"])
	  res["ellipticity"]    = xr.DataArray(ellipticity,coords=[res["t"],res["f"]],dims=["time","frequency"])
	  res["k_tp"]           = xr.DataArray(k_ThPhSVD_fac,coords=[res["t"],res["f"],["theta","phi"]],dims=["time","frequency","comp"])


	return res