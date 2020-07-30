import numpy as np
import xarray as xr
import bisect
import multiprocessing as mp


from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_vec_xyz import ts_vec_xyz
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz


def calc_moms(nt,arguments):
	"""

	"""
	if len(arguments)>13:
		(isbrstdata,flag_same_e,flag_dE,stepTable,energy0,deltav0,energy1,deltav1,qe,SCpot,pmass,\
			flag_innerelec,W_innerelec,phitr,thetak,intenergies,pdist,deltaang) = arguments
	else : 
		(isbrstdata,flag_same_e,flag_dE,energy,deltav,qe,SCpot,pmass,\
			flag_innerelec,W_innerelec,phitr,thetak,intenergies,pdist,deltaang) = arguments
	if isbrstdata:
		if not flag_same_e or not flag_dE:
			energy = energy0
			deltav = deltav0

			if stepTable[nt]:
				energy = energy1
				deltav = deltav1

	v = np.real(np.sqrt(2*qe*(energy-SCpot.data[nt])/pmass))
	v[energy-SCpot.data[nt]-flag_innerelec*W_innerelec<0] = 0

	if isbrstdata:
		phij = phitr[nt,:]
	else:
		phij = phitr

	phij 	= phij[:,np.newaxis]
	
	n_psd = 0
	V_psd = np.zeros(3)
	P_psd = np.zeros((3,3))
	H_psd = np.zeros(3)

	Mpsd2n    = np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180))
	Mpsd2Vx   = -np.dot(np.cos(phij*np.pi/180),np.sin(thetak*np.pi/180)*np.sin(thetak*np.pi/180))
	Mpsd2Vy   = -np.dot(np.sin(phij*np.pi/180),np.sin(thetak*np.pi/180)*np.sin(thetak*np.pi/180))
	Mpsd2Vz   = -np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180)*np.cos(thetak*np.pi/180))
	Mpsdmfxx  = np.dot(np.cos(phij*np.pi/180)**2,np.sin(thetak*np.pi/180)**3)
	Mpsdmfyy  = np.dot(np.sin(phij*np.pi/180)**2,np.sin(thetak*np.pi/180)**3)
	Mpsdmfzz  = np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180)*np.cos(thetak*np.pi/180)**2)
	Mpsdmfxy  = np.dot(np.cos(phij*np.pi/180)*np.sin(phij*np.pi/180),np.sin(thetak*np.pi/180)**3)
	Mpsdmfxz  = np.dot(np.cos(phij*np.pi/180),np.cos(thetak*np.pi/180)*np.sin(thetak*np.pi/180)**2)
	Mpsdmfyz  = np.dot(np.sin(phij*np.pi/180),np.cos(thetak*np.pi/180)*np.sin(thetak*np.pi/180)**2)

	for ii in intenergies:
		tmp = np.squeeze(pdist[nt,ii,:,:])
		#n_psd_tmp1 = tmp .* Mpsd2n * v(ii)^2 * deltav(ii) * deltaang;
		#n_psd_e32_phi_theta(nt, ii, :, :) = n_psd_tmp1;
		#n_psd_e32(nt, ii) = n_psd_tmp

		# number density
		n_psd_tmp       = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2n,axis=0),axis=0)*v[ii]**2
		n_psd       	+= n_psd_tmp

		# Bulk velocity
		Vxtemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vx,axis=0),axis=0)*v[ii]**3
		Vytemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vy,axis=0),axis=0)*v[ii]**3
		Vztemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vz,axis=0),axis=0)*v[ii]**3
		V_psd[0]    	+= Vxtemp
		V_psd[1]    	+= Vytemp
		V_psd[2]    	+= Vztemp

		# Pressure tensor
		Pxxtemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxx,axis=0),axis=0)*v[ii]**4
		Pxytemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxy,axis=0),axis=0)*v[ii]**4
		Pxztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxz,axis=0),axis=0)*v[ii]**4
		Pyytemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfyy,axis=0),axis=0)*v[ii]**4
		Pyztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfyz,axis=0),axis=0)*v[ii]**4
		Pzztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfzz,axis=0),axis=0)*v[ii]**4
		P_psd[0,0]   	+= Pxxtemp
		P_psd[0,1]   	+= Pxytemp
		P_psd[0,2]   	+= Pxztemp
		P_psd[1,1]   	+= Pyytemp
		P_psd[1,2]   	+= Pyztemp
		P_psd[2,2]   	+= Pzztemp

		H_psd[0]     = Vxtemp*v[ii]**2
		H_psd[1]     = Vytemp*v[ii]**2
		H_psd[2]     = Vztemp*v[ii]**2
		
	return (n_psd,V_psd,P_psd,H_psd)



def psd_moments(pdist=None, SCpot=None, **kwargs):
	"""
	Computes moments from the FPI particle phase-space densities
	
	Parameters :
		pdist : DataArray
			3D skymap velocity distribution

		SCpot : DataArray
			Time series of the spacecraft potential

	Options :
		energyrange : list/ndarray 
			Set energy range in eV to integrate over [E_min E_max]. Energy range is applied to energy0 and the same 
			elements are used for energy1 to ensure that the same number of points are integrated over.

		noscpot : bool
			Set to 1 to set spacecraft potential to zero. Calculates moments 
			without correcting for spacecraft potential.

		enchannels : list/ndarray
			Set energy channels to integrate over [min max]; min and max between must be between 1 and 32.

		partialmoms : ndarray,DataArray
			Use a binary array (or DataArray) (pmomsarr) to select which psd points are used in the moments 
			calculation. pmomsarr must be a binary array (1s and 0s, 1s correspond to points used). 
			Array (or data of Dataarray) must be the same size as pdist.data.

		innerelec : "on"/"off"
			Innerelectron potential for electron moments

	Returns :
		n_psd : DataArray
			Time series of the number density (1rst moment)

		V_psd : DataArray
			Time series of the bulk velocity (2nd moment)

		P_psd : DataArrya
			Time series of the pressure tensor (3rd moment)

		P2_psd : DataArray
			Time series of the pressure tensor 

		T_psd : DataArray
			Time series of the temperature tensor 

		H_psd : DataArray
			??
	
	Example :
		>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
		>>> ePDist = pyrf.get_data("PDe_fpi_brst_l2",Tint,1)
		>>> SCpot = pyrf.get_data("V_edp_brst_l2",Tint,1)
		>>> particlemoments = pyrf.psd_moments(ePDist,SCpot,energyrange=[1 1000])

	"""
	

	flag_dE         = False
	flag_same_e     = False
	flag_innerelec  = False
	W_innerelec     = 3.5         # [eV] scpot + W_innerelec for electron moments calculation; 2018-01-26, wy;

	
	if pdist is None or SCpot is None:
		raise ValueError("psd_moments requires at least 2 arguments")

	if not isinstance(pdist,xr.Dataset):
		raise TypeError("pdist must be a Dataset")
	else :
		pdist.data.data *=1e12

	if not isinstance(SCpot,xr.DataArray):
		raise TypeError("Spacecraft potential must a DataArray")


	# Check if data is fast or burst resolution
	fieldName = pdist.attrs["FIELDNAM"]

	if "brst" in fieldName:
		isbrstdata = True
		print("notice : Burst resolution data is used")
	elif "brst" in fieldName:
		isbrstdata = False
		print("notice : Fast resolution data is used")
	else :
		raise TypeError("Could not identify if data is fast or burst.")

	phi           = pdist.phi.data
	thetak        = pdist.theta
	particletype  = pdist.attrs["species"]

	if isbrstdata:
		stepTable = pdist.attrs["esteptable"]
		energy0   = pdist.attrs["energy0"]
		energy1   = pdist.attrs["energy1"]
		etmp      = energy1 - energy0

		if all(etmp) == 0:
			flag_same_e = 1
	else:
		energy = pdist.energy
		etmp = energy[0,:]-energy[-1,:]
		
		if all(etmp) == 0:
			energy = energy[0,:]
		else:
			raise TypeError("Could not identify if data is fast or burst.")

	#resample SCpot to same resolution as particle distributions
	SCpot = resample(SCpot,pdist.time)

	#check theta dimensions
	thetasize = list(thetak.shape)

	intenergies = np.arange(32)

	if "energyrange" in kwargs:
		if isinstance(kwargs["energyrange"],(list,np.ndarray)) and len(kwargs["energyrange"]) == 2:
			if not isbrstdata :
				energy0 = energy

			Eminmax = kwargs["energyrange"]
			starte  = bisect.bisect_left(energy0,Eminmax[0])
			ende    = bisect.bisect_left(energy0,Eminmax[1])

			intenergies = np.arange(starte,ende)
			print("notice : Using partial energy range")

	if "noscpot" in kwargs:
		if isinstance(kwargs["noscpot"],bool) and not kwargs["noscpot"]:
			SCpot.data = np.zeros(SCpot.shape)
			print("notice : Setting spacecraft potential to zero")

	if "enchannels" in kwargs:
		if isinstance(kwargs["enchannels"],(listnp.ndarray)):
			intenergies = np.arange(kwargs["enchannels"][0],kwargs["enchannels"][1])

	if "partialmoms" in kwargs:
		partialmoms = kwargs["partialmoms"]
		if isinstance(partialmoms,xr.Dataarray):
			partialmoms = partialmoms.data

		# Check size of partialmoms
		if partialmoms.shape == pdist.data.shape:
			sumones     = np.sum(np.sum(np.sum(np.sum(partialmoms,axis=-1),axis=-1),axis=-1),axis=-1)
			sumzeros    = np.sum(np.sum(np.sum(np.sum(-partialmoms+1,axis=-1),axis=-1),axis=-1),axis=-1)

			if (sumones+sumzeros) == pdist.data.size:
				print("notice : partialmoms is correct. Partial moments will be calculated")
				pdist.data = pdist.data*partialmoms
			else :
				print("notice : All values are not ones and zeros in partialmoms. Full moments will be calculated")
		else :
			print("notice : Size of partialmoms is wrong. Full moments will be calculated")

	if "innerelec" in kwargs:
		innerelec_tmp = kwargs["innerelec"]
		if innerelec_tmp == "on" and particletype[0] == "e":
			flag_innerelec = True

	# Define constants
	qe = constants.e.value
	kb = constants.k_B.value

	if particletype[0] == "e":
		pmass       = constants.m_e.value
		print("notice : Particles are electrons")
	elif particletype[0] == "i":
		pmass       = constants.m_p.value
		SCpot.data  = -SCpot.data
		print("notice : Particles are ions")
	else:
		raise ValueError("Could not identify the particle type")
	  
	# Define arrays for output
	n_psd = np.zeros(len(pdist.time))
	#sizedist = size(pdist.data)
	#n_psd_e32 = zeros(length(pdist.time), 32);
	#n_psd_e32_phi_theta = zeros(sizedist(1), sizedist(2), sizedist(3), sizedist(4));
	V_psd   = np.zeros((len(pdist.time),3))
	P_psd   = np.zeros((len(pdist.time),3,3))
	P2_psd  = np.zeros((len(pdist.time),3,3))
	H_psd   = np.zeros((len(pdist.time),3))

	# angle between theta and phi points is 360/32 = 11.25 degrees
	deltaang = (11.25*np.pi/180)**2

	if isbrstdata :
		phitr = pdist.phi
	else : 
		phitr = phi;
		phisize = phitr.shape
		
		if phisize[1] > phisize[0] :
			phitr = phitr.T

	if "delta_energy_minus" in pdist.attrs and "delta_energy_plus" in pdist.attrs :
		flag_dE = True

	# Calculate speed widths associated with each energy channel.
	if isbrstdata: # Burst mode energy/speed widths
		if flag_same_e and flag_dE:
			energy      = energy0
			energyupper = energy+energy_plus
			energylower = energy-energy_minus
			vupper      = np.sqrt(2*qe*energyupper/pmass)
			vlower      = np.sqrt(2*qe*energylower/pmass)
			deltav      = vupper-vlower
		else :
			energyall   = np.hstack([energy0,energy1])
			energyall   = np.log10(np.sort(energyall))
			
			if np.abs(energyall[1]-energyall[0]) > 1e-4:
				temp0   = 2*energyall[0]-energyall[1]
			else:
				temp0   = 2*energyall[1]-energyall[2]

			if np.abs(energyall[63]-energyall[62]) > 1e-4:
				temp65  = 2*energyall[63]-energyall[62]
			else:
				temp65  = 2*energyall[63]-energyall[61]

			energyall       = np.hstack([temp0,energyall,temp65])
			diffenall       = np.diff(energyall)
			energy0upper    = 10**(np.log10(energy0)+diffenall[1:64:2]/2)
			energy0lower    = 10**(np.log10(energy0)-diffenall[0:63:2]/2)
			energy1upper    = 10**(np.log10(energy1)+diffenall[2:65:2]/2)
			energy1lower    = 10**(np.log10(energy1)-diffenall[1:64:2]/2)

			v0upper         = np.sqrt(2*qe*energy0upper/pmass)
			v0lower         = np.sqrt(2*qe*energy0lower/pmass)
			v1upper         = np.sqrt(2*qe*energy1upper/pmass)
			v1lower         = np.sqrt(2*qe*energy1lower/pmass)
			deltav0         = (v0upper-v0lower)*2.0
			deltav1         = (v1upper-v1lower)*2.0
			#deltav0(1) = deltav0(1)*2.7;
			#deltav1(1) = deltav1(1)*2.7;
	else : # Fast mode energy/speed widths
		energyall   = np.log10(energy)
		temp0       = 2*energyall[0]-energyall[1]
		temp33      = 2*energyall[31]-energyall[30]
		energyall   = np.hstack([temp0,energyall,temp33])
		diffenall   = np.diff(energyall)
		energyupper = 10**(np.log10(energy)+diffenall[1:33]/4)
		energylower = 10**(np.log10(energy)-diffenall[0:32]/4)
		vupper      = np.sqrt(2*qe*energyupper/pmass)
		vlower      = np.sqrt(2*qe*energylower/pmass)
		deltav      = (vupper-vlower)*2.0
		deltav[0]   = deltav[0]*2.7

	thetak 	= thetak.data[np.newaxis,:]

	#-----------------------------------------------------------------------------------------------------------------
	# New version parrallel
	#-----------------------------------------------------------------------------------------------------------------
	# args brst : (isbrstdata,flag_same_e,flag_dE,stepTable,energy0,deltav0,energy1,deltav1,qe,SCpot.data,pmass,\
	#				flag_innerelec,W_innerelec,phitr.data,thetak,intenergies,pdist.data.data,deltaang)
	# args fast : (isbrstdata,flag_same_e,flag_dE,energy,deltav,qe,SCpot.data,pmass,\
	#				flag_innerelec,W_innerelec,phitr.data,thetak,intenergies,pdist.data,deltaang)

	if isbrstdata:
		arguments = (isbrstdata,flag_same_e,flag_dE,stepTable,energy0,deltav0,energy1,deltav1,qe,SCpot.data,pmass,\
						flag_innerelec,W_innerelec,phitr.data,thetak,intenergies,pdist.data.data,deltaang)
	else :
		arguments = (isbrstdata,flag_same_e,flag_dE,energy,deltav,qe,SCpot.data,pmass,\
						flag_innerelec,W_innerelec,phitr.data,thetak,intenergies,pdist.data,deltaang)

	pool = mp.Pool(mp.cpu_count())
	res = pool.starmap(calc_moms,[(nt,arguments) for nt in range(len(pdist.time))])
	
	out = np.vstack(res)
	n_psd = np.array(out[:,0],dtype="float")
	V_psd = np.vstack(out[:,1][:])
	P_psd = np.vstack(out[:,2][:])
	P_psd = np.reshape(P_psd,(len(n_psd),3,3))
	H_psd = np.vstack(out[:,3][:])

	pool.close()
	#-----------------------------------------------------------------------------------------------------------------
	# Old version serial
	#-----------------------------------------------------------------------------------------------------------------
	"""
	for nt in tqdm(range(len(pdist.time))):
		if isbrstdata:
			if not flag_same_e or not flag_dE:
				energy = energy0
				deltav = deltav0

				if stepTable[nt]:
					energy = energy1
					deltav = deltav1
	  
		v = np.real(np.sqrt(2*qe*(energy-SCpot.data[nt])/pmass))
		v[energy-SCpot.data[nt]-flag_innerelec*W_innerelec<0] = 0

		if isbrstdata:
			phij = phitr.data[nt,:]
		else:
			phij = phitr.data

		phij 	= phij[:,np.newaxis]

		
		Mpsd2n    = np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180))
		Mpsd2Vx   = -np.dot(np.cos(phij*np.pi/180),np.sin(thetak*np.pi/180)*np.sin(thetak*np.pi/180))
		Mpsd2Vy   = -np.dot(np.sin(phij*np.pi/180),np.sin(thetak*np.pi/180)*np.sin(thetak*np.pi/180))
		Mpsd2Vz   = -np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180)*np.cos(thetak*np.pi/180))
		Mpsdmfxx  = np.dot(np.cos(phij*np.pi/180)**2,np.sin(thetak*np.pi/180)**3)
		Mpsdmfyy  = np.dot(np.sin(phij*np.pi/180)**2,np.sin(thetak*np.pi/180)**3)
		Mpsdmfzz  = np.dot(np.ones(phij.shape),np.sin(thetak*np.pi/180)*np.cos(thetak*np.pi/180)**2)
		Mpsdmfxy  = np.dot(np.cos(phij*np.pi/180)*np.sin(phij*np.pi/180),np.sin(thetak*np.pi/180)**3)
		Mpsdmfxz  = np.dot(np.cos(phij*np.pi/180),np.cos(thetak*np.pi/180)*np.sin(thetak*np.pi/180)**2)
		Mpsdmfyz  = np.dot(np.sin(phij*np.pi/180),np.cos(thetak*np.pi/180)*np.sin(thetak*np.pi/180)**2)

		for ii in intenergies:
			tmp = np.squeeze(pdist.data[nt,ii,:,:])
			#n_psd_tmp1 = tmp .* Mpsd2n * v(ii)^2 * deltav(ii) * deltaang;
			#n_psd_e32_phi_theta(nt, ii, :, :) = n_psd_tmp1;
			#n_psd_e32(nt, ii) = n_psd_tmp
			
			# number density
			n_psd_tmp       = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2n,axis=0),axis=0)*v[ii]**2
			n_psd[nt]       += n_psd_tmp
			
			# Bulk velocity
			Vxtemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vx,axis=0),axis=0)*v[ii]**3
			Vytemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vy,axis=0),axis=0)*v[ii]**3
			Vztemp          = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsd2Vz,axis=0),axis=0)*v[ii]**3
			V_psd[nt, 0]    += Vxtemp
			V_psd[nt, 1]    += Vytemp
			V_psd[nt, 2]    += Vztemp

			# Pressure tensor
			Pxxtemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxx,axis=0),axis=0)*v[ii]**4
			Pxytemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxy,axis=0),axis=0)*v[ii]**4
			Pxztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfxz,axis=0),axis=0)*v[ii]**4
			Pyytemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfyy,axis=0),axis=0)*v[ii]**4
			Pyztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfyz,axis=0),axis=0)*v[ii]**4
			Pzztemp         = deltav[ii]*deltaang*np.nansum(np.nansum(tmp*Mpsdmfzz,axis=0),axis=0)*v[ii]**4
			P_psd[nt,0,0]   += Pxxtemp
			P_psd[nt,0,1]   += Pxytemp
			P_psd[nt,0,2]   += Pxztemp
			P_psd[nt,1,1]   += Pyytemp
			P_psd[nt,1,2]   += Pyztemp
			P_psd[nt,2,2]   += Pzztemp
			
			
			H_psd[nt,0]     = Vxtemp*v[ii]**2
			H_psd[nt,1]     = Vytemp*v[ii]**2
			H_psd[nt,2]     = Vztemp*v[ii]**2
	"""

	# Compute moments in SI units
	P_psd           *= pmass
	V_psd           /= n_psd[:,np.newaxis]
	P2_psd[:,0,0]   = P_psd[:,0,0]
	P2_psd[:,0,1]   = P_psd[:,0,1]
	P2_psd[:,0,2]   = P_psd[:,0,2]
	P2_psd[:,1,1]   = P_psd[:,1,1]
	P2_psd[:,1,2]   = P_psd[:,1,2]
	P2_psd[:,2,2]   = P_psd[:,2,2]
	P2_psd[:,1,0]   = P2_psd[:,0,1]
	P2_psd[:,2,0]   = P2_psd[:,0,2]
	P2_psd[:,2,1]   = P2_psd[:,1,2]

	P_psd[:,0,0]    -= pmass*n_psd*V_psd[:,0]*V_psd[:,0]
	P_psd[:,0,1]    -= pmass*n_psd*V_psd[:,0]*V_psd[:,1]
	P_psd[:,0,2]    -= pmass*n_psd*V_psd[:,0]*V_psd[:,2]
	P_psd[:,1,1]    -= pmass*n_psd*V_psd[:,1]*V_psd[:,1]
	P_psd[:,1,2]    -= pmass*n_psd*V_psd[:,1]*V_psd[:,2]
	P_psd[:,2,2]    -= pmass*n_psd*V_psd[:,2]*V_psd[:,2]
	P_psd[:,1,0]    = P_psd[:,0,1]
	P_psd[:,2,0]    = P_psd[:,0,2]
	P_psd[:,2,1]    = P_psd[:,1,2]

	Ptrace          = np.trace(P_psd,axis1=1,axis2=2)
	T_psd 			= np.zeros(P_psd.shape)
	T_psd[...]      = P_psd[...]/(kb*n_psd[:,np.newaxis,np.newaxis])
	T_psd[:,1,0]    = T_psd[:,1,0]
	T_psd[:,2,0]    = T_psd[:,2,0]
	T_psd[:,2,1]    = T_psd[:,2,1]


	Vabs2           = np.linalg.norm(V_psd,axis=1)**2
	H_psd           *= pmass/2
	H_psd[:,0]      -= V_psd[:,0]*P_psd[:,0,0] + V_psd[:,1]*P_psd[:,0,1] + V_psd[:,2]*P_psd[:,0,2]
	H_psd[:,0]      -= 0.5*V_psd[:,0]*Ptrace + 0.5*pmass*n_psd*Vabs2*V_psd[:,0]
	H_psd[:,1]      -= V_psd[:,0]*P_psd[:,0,1] + V_psd[:,1]*P_psd[:,1,1] + V_psd[:,2]*P_psd[:,1,2]
	H_psd[:,1]      -= 0.5*V_psd[:,1]*Ptrace + 0.5*pmass*n_psd*Vabs2*V_psd[:,1]
	H_psd[:,2]      -= V_psd[:,0]*P_psd[:,0,2] + V_psd[:,1]*P_psd[:,1,2] + V_psd[:,2]*P_psd[:,2,2]
	H_psd[:,2]      -= 0.5*V_psd[:,2]*Ptrace + 0.5*pmass*n_psd*Vabs2*V_psd[:,2]

	# Convert to typical units (/cc, km/s, nP, eV, and ergs/s/cm^2).
	n_psd                   /= 1e6
	#n_psd_e32              /= 1e6
	#n_psd_e32_phi_theta    /= 1e6
	V_psd                   /= 1e3
	P_psd                   *= 1e9
	P2_psd                  *= 1e9
	T_psd                   *= kb/qe
	H_psd                   *= 1e3

	# Construct TSeries
	n_psd           = ts_scalar(pdist.time.data,n_psd)
	#n_psd_e32      = ts_scalar(pdist.time, n_psd_e32);
	#n_psd_skymap   = ts_skymap(pdist.time.data, n_psd_e32_phi_theta,energy, phi.data, thetak);
	V_psd           = ts_vec_xyz(pdist.time.data,V_psd)
	P_psd           = ts_tensor_xyz(pdist.time.data,P_psd)
	P2_psd          = ts_tensor_xyz(pdist.time.data,P2_psd)
	T_psd           = ts_tensor_xyz(pdist.time.data,T_psd)
	H_psd           = ts_vec_xyz(pdist.time.data,H_psd)

	return(n_psd,V_psd,P_psd,P2_psd,T_psd,H_psd)


