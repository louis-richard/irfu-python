import numpy as np
import xarray as xr
import bisect
import multiprocessing as mp
from astropy import constants

from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_vec_xyz import ts_vec_xyz
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz


# noinspection PyUnboundLocalVariable
def calc_moms(nt, arguments):
	"""

	"""
	if len(arguments) > 13:
		[isbrstdata, flag_same_e, flag_de, step_table, energy0, deltav0, energy1, deltav1, qe, scpot, pmass,
		 flag_inner_elec, w_inner_elec, phitr, thetak, intenergies, pdist, deltaang] = arguments
	else:
		[isbrstdata, flag_same_e, flag_de, energy, deltav, qe, scpot, pmass, flag_inner_elec, w_inner_elec, phitr,
		 thetak, intenergies, pdist, deltaang] = arguments

	if isbrstdata:
		if not flag_same_e or not flag_de:
			energy = energy0
			deltav = deltav0

			if step_table[nt]:
				energy = energy1
				deltav = deltav1

	v = np.real(np.sqrt(2 * qe * (energy - scpot.data[nt]) / pmass))
	v[energy - scpot.data[nt] - flag_inner_elec * w_inner_elec < 0] = 0

	if isbrstdata:
		phij = phitr[nt, :]
	else:
		phij = phitr

	phij = phij[:, np.newaxis]
	
	n_psd = 0
	v_psd = np.zeros(3)
	p_psd = np.zeros((3, 3))
	h_psd = np.zeros(3)

	psd2_n_mat = np.dot(np.ones(phij.shape), np.sin(thetak * np.pi / 180))
	psd2_v_x_mat = -np.dot(np.cos(phij * np.pi / 180), np.sin(thetak * np.pi / 180) * np.sin(thetak * np.pi / 180))
	psd2_v_y_mat = -np.dot(np.sin(phij * np.pi / 180), np.sin(thetak * np.pi / 180) * np.sin(thetak * np.pi / 180))
	psd2_v_z_mat = -np.dot(np.ones(phij.shape), np.sin(thetak * np.pi / 180) * np.cos(thetak * np.pi / 180))
	psd_mf_xx_mat = np.dot(np.cos(phij * np.pi / 180) ** 2, np.sin(thetak * np.pi / 180) ** 3)
	psd_mf_yy_mat = np.dot(np.sin(phij * np.pi / 180) ** 2, np.sin(thetak * np.pi / 180) ** 3)
	psd_mf_zz_mat = np.dot(np.ones(phij.shape), np.sin(thetak * np.pi / 180) * np.cos(thetak * np.pi / 180) ** 2)
	psd_mf_xy_mat = np.dot(np.cos(phij * np.pi / 180) * np.sin(phij * np.pi / 180), np.sin(thetak * np.pi / 180) ** 3)
	psd_mf_xz_mat = np.dot(np.cos(phij * np.pi / 180), np.cos(thetak * np.pi / 180) * np.sin(thetak * np.pi / 180) ** 2)
	psd_mf_yz_mat = np.dot(np.sin(phij * np.pi / 180), np.cos(thetak * np.pi / 180) * np.sin(thetak * np.pi / 180) ** 2)

	for ii in intenergies:
		tmp = np.squeeze(pdist[nt, ii, :, :])
		# n_psd_tmp1 = tmp .* psd2_n_mat * v(ii)^2 * deltav(ii) * deltaang;
		# n_psd_e32_phi_theta(nt, ii, :, :) = n_psd_tmp1;
		# n_psd_e32(nt, ii) = n_psd_tmp

		# number density
		n_psd_tmp = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd2_n_mat, axis=0), axis=0) * v[ii] ** 2
		n_psd += n_psd_tmp

		# Bulk velocity
		v_temp_x = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd2_v_x_mat, axis=0), axis=0) * v[ii] ** 3
		v_temp_y = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd2_v_y_mat, axis=0), axis=0) * v[ii] ** 3
		v_temp_z = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd2_v_z_mat, axis=0), axis=0) * v[ii] ** 3

		v_psd[0] += v_temp_x
		v_psd[1] += v_temp_y
		v_psd[2] += v_temp_z

		# Pressure tensor
		p_temp_xx = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_xx_mat, axis=0), axis=0) * v[ii] ** 4
		p_temp_xy = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_xy_mat, axis=0), axis=0) * v[ii] ** 4
		p_temp_xz = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_xz_mat, axis=0), axis=0) * v[ii] ** 4
		p_temp_yy = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_yy_mat, axis=0), axis=0) * v[ii] ** 4
		p_temp_yz = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_yz_mat, axis=0), axis=0) * v[ii] ** 4
		p_temp_zz = deltav[ii] * deltaang * np.nansum(np.nansum(tmp * psd_mf_zz_mat, axis=0), axis=0) * v[ii] ** 4

		p_psd[0, 0] += p_temp_xx
		p_psd[0, 1] += p_temp_xy
		p_psd[0, 2] += p_temp_xz
		p_psd[1, 1] += p_temp_yy
		p_psd[1, 2] += p_temp_yz
		p_psd[2, 2] += p_temp_zz

		h_psd[0] = v_temp_x * v[ii] ** 2
		h_psd[1] = v_temp_y * v[ii] ** 2
		h_psd[2] = v_temp_z * v[ii] ** 2
		
	return n_psd, v_psd, p_psd, h_psd


# noinspection PyUnboundLocalVariable
def psd_moments(pdist=None, scpot=None, **kwargs):
	"""
	Computes moments from the FPI particle phase-space densities
	
	Parameters :
		pdist : DataArray
			3D skymap velocity distribution

		scpot : DataArray
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

		v_psd : DataArray
			Time series of the bulk velocity (2nd moment)

		p_psd : DataArrya
			Time series of the pressure tensor (3rd moment)

		p2_psd : DataArray
			Time series of the pressure tensor 

		t_psd : DataArray
			Time series of the temperature tensor 

		h_psd : DataArray
			??
	
	Example :
		>>> from pyrfu import mms
		>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
		>>> ePDist = mms.get_data("PDe_fpi_brst_l2",Tint,1)
		>>> scpot = mms.get_data("V_edp_brst_l2",Tint,1)
		>>> particlemoments = mms.psd_moments(ePDist,scpot,energyrange=[1 1000])

	"""

	flag_de = False
	flag_same_e = False
	flag_inner_elec = False
	
	# [eV] scpot + w_inner_elec for electron moments calculation; 2018-01-26, wy;
	w_inner_elec = 3.5
	
	if pdist is None or scpot is None:
		raise ValueError("psd_moments requires at least 2 arguments")

	if not isinstance(pdist, xr.Dataset):
		raise TypeError("pdist must be a Dataset")
	else:
		pdist.data.data *= 1e12

	if not isinstance(scpot, xr.DataArray):
		raise TypeError("Spacecraft potential must a DataArray")

	# Check if data is fast or burst resolution
	field_name = pdist.attrs["FIELDNAM"]

	if "brst" in field_name:
		isbrstdata = True
		print("notice : Burst resolution data is used")
	elif "brst" in field_name:
		isbrstdata = False
		print("notice : Fast resolution data is used")
	else:
		raise TypeError("Could not identify if data is fast or burst.")

	phi = pdist.phi.data
	thetak = pdist.theta
	particletype = pdist.attrs["species"]

	if isbrstdata:
		step_table = pdist.attrs["estep_table"]
		energy0 = pdist.attrs["energy0"]
		energy1 = pdist.attrs["energy1"]
		e_tmp = energy1 - energy0

		if all(e_tmp) == 0:
			flag_same_e = 1
	else:
		energy = pdist.energy
		e_tmp = energy[0, :] - energy[-1, :]
		
		if all(e_tmp) == 0:
			energy = energy[0, :]
		else:
			raise TypeError("Could not identify if data is fast or burst.")

	# resample scpot to same resolution as particle distributions
	scpot = resample(scpot, pdist.time)

	intenergies = np.arange(32)

	if "energyrange" in kwargs:
		if isinstance(kwargs["energyrange"], (list, np.ndarray)) and len(kwargs["energyrange"]) == 2:
			if not isbrstdata:
				energy0 = energy

			e_min_max = kwargs["energyrange"]
			starte = bisect.bisect_left(energy0, e_min_max[0])
			ende = bisect.bisect_left(energy0, e_min_max[1])

			intenergies = np.arange(starte, ende)
			print("notice : Using partial energy range")

	if "noscpot" in kwargs:
		if isinstance(kwargs["noscpot"], bool) and not kwargs["noscpot"]:
			scpot.data = np.zeros(scpot.shape)
			print("notice : Setting spacecraft potential to zero")

	if "enchannels" in kwargs:
		if isinstance(kwargs["enchannels"], (list, np.ndarray)):
			intenergies = np.arange(kwargs["enchannels"][0], kwargs["enchannels"][1])

	if "partialmoms" in kwargs:
		partialmoms = kwargs["partialmoms"]
		if isinstance(partialmoms, xr.DataArray):
			partialmoms = partialmoms.data

		# Check size of partialmoms
		if partialmoms.shape == pdist.data.shape:
			sumones = np.sum(np.sum(np.sum(np.sum(partialmoms, axis=-1), axis=-1), axis=-1), axis=-1)
			sumzeros = np.sum(np.sum(np.sum(np.sum(-partialmoms + 1, axis=-1), axis=-1), axis=-1), axis=-1)

			if (sumones + sumzeros) == pdist.data.size:
				print("notice : partialmoms is correct. Partial moments will be calculated")
				pdist.data = pdist.data*partialmoms
			else:
				print("notice : All values are not ones and zeros in partialmoms. Full moments will be calculated")
		else:
			print("notice : Size of partialmoms is wrong. Full moments will be calculated")

	if "innerelec" in kwargs:
		innerelec_tmp = kwargs["innerelec"]
		if innerelec_tmp == "on" and particletype[0] == "e":
			flag_inner_elec = True

	# Define constants
	qe = constants.e.value
	kb = constants.k_B.value

	if particletype[0] == "e":
		pmass = constants.m_e.value
		print("notice : Particles are electrons")
	elif particletype[0] == "i":
		pmass = constants.m_p.value
		scpot.data = -scpot.data
		print("notice : Particles are ions")
	else:
		raise ValueError("Could not identify the particle type")

	# Define arrays for output
	# sizedist = size(pdist.data)
	# n_psd_e32 = zeros(length(pdist.time), 32);
	# n_psd_e32_phi_theta = zeros(sizedist(1), sizedist(2), sizedist(3), sizedist(4));

	p2_psd = np.zeros((len(pdist.time), 3, 3))

	# angle between theta and phi points is 360/32 = 11.25 degrees
	deltaang = (11.25 * np.pi / 180) ** 2

	if isbrstdata:
		phitr = pdist.phi
	else:
		phitr = phi
		phisize = phitr.shape
		
		if phisize[1] > phisize[0]:
			phitr = phitr.T

	if "delta_energy_minus" in pdist.attrs and "delta_energy_plus" in pdist.attrs:
		energy_minus, energy_plus = [pdist.attrs["delta_energy_plus"], pdist.attrs["delta_energy_plus"]]
		
		flag_de = True
	else:
		energy_minus, energy_plus = [None, None]

	# Calculate speed widths associated with each energy channel.
	if isbrstdata:  # Burst mode energy/speed widths
		if flag_same_e and flag_de:
			energy = energy0
			energy_upper = energy + energy_plus
			energy_lower = energy - energy_minus
			v_upper = np.sqrt(2 * qe * energy_upper / pmass)
			v_lower = np.sqrt(2 * qe * energy_lower / pmass)
			deltav = v_upper - v_lower
		else:
			energy_all = np.hstack([energy0, energy1])
			energy_all = np.log10(np.sort(energy_all))

			if np.abs(energy_all[1] - energy_all[0]) > 1e-4:
				temp0 = 2 * energy_all[0] - energy_all[1]
			else:
				temp0 = 2 * energy_all[1] - energy_all[2]

			if np.abs(energy_all[63] - energy_all[62]) > 1e-4:
				temp65 = 2 * energy_all[63] - energy_all[62]
			else:
				temp65 = 2 * energy_all[63] - energy_all[61]

			energy_all = np.hstack([temp0, energy_all, temp65])
			diff_en_all = np.diff(energy_all)
			energy0upper = 10 ** (np.log10(energy0) + diff_en_all[1:64:2] / 2)
			energy0lower = 10 ** (np.log10(energy0) - diff_en_all[0:63:2] / 2)
			energy1upper = 10 ** (np.log10(energy1) + diff_en_all[2:65:2] / 2)
			energy1lower = 10 ** (np.log10(energy1) - diff_en_all[1:64:2] / 2)

			v0upper = np.sqrt(2 * qe * energy0upper / pmass)
			v0lower = np.sqrt(2 * qe * energy0lower / pmass)
			v1upper = np.sqrt(2 * qe * energy1upper / pmass)
			v1lower = np.sqrt(2 * qe * energy1lower / pmass)
			deltav0 = (v0upper - v0lower) * 2.0
			deltav1 = (v1upper - v1lower) * 2.0

			# deltav0(1) = deltav0(1)*2.7
			# deltav1(1) = deltav1(1)*2.7

	else:  # Fast mode energy/speed widths
		energy_all = np.log10(energy)
		temp0 = 2 * energy_all[0] - energy_all[1]
		temp33 = 2 * energy_all[31] - energy_all[30]
		energy_all = np.hstack([temp0, energy_all, temp33])
		diff_en_all = np.diff(energy_all)
		energy_upper = 10 ** (np.log10(energy) + diff_en_all[1:33] / 4)
		energy_lower = 10 ** (np.log10(energy) - diff_en_all[0:32] / 4)
		v_upper = np.sqrt(2 * qe * energy_upper / pmass)
		v_lower = np.sqrt(2 * qe * energy_lower / pmass)
		deltav = (v_upper - v_lower) * 2.0
		deltav[0] = deltav[0] * 2.7

	thetak = thetak.data[np.newaxis, :]

	# New version parrallel
	# args brst : (isbrstdata,flag_same_e,flag_de,step_table,energy0,deltav0,energy1,deltav1,qe,scpot.data,pmass,
	# 					flag_inner_elec,w_inner_elec,phitr.data,thetak,intenergies,pdist.data.data,deltaang)
	# args fast : (isbrstdata,flag_same_e,flag_de,energy,deltav,qe,scpot.data,pmass, flag_inner_elec,w_inner_elec,
	# 					phitr.data,thetak,intenergies,pdist.data,deltaang)

	if isbrstdata:
		arguments = (
		isbrstdata, flag_same_e, flag_de, step_table, energy0, deltav0, energy1, deltav1, qe, scpot.data, pmass,
		flag_inner_elec, w_inner_elec, phitr.data, thetak, intenergies, pdist.data.data, deltaang)
	else:
		arguments = (
		isbrstdata, flag_same_e, flag_de, energy, deltav, qe, scpot.data, pmass, flag_inner_elec, w_inner_elec,
		phitr.data, thetak, intenergies, pdist.data, deltaang)

	pool = mp.Pool(mp.cpu_count())
	res = pool.starmap(calc_moms, [(nt, arguments) for nt in range(len(pdist.time))])
	out = np.vstack(res)

	n_psd = np.array(out[:, 0], dtype="float")
	v_psd = np.vstack(out[:, 1][:])
	p_psd = np.vstack(out[:, 2][:])
	p_psd = np.reshape(p_psd, (len(n_psd), 3, 3))
	h_psd = np.vstack(out[:, 3][:])

	pool.close()

	# Compute moments in SI units
	p_psd *= pmass
	v_psd /= n_psd[:, np.newaxis]
	p2_psd[:, 0, 0] = p_psd[:, 0, 0]
	p2_psd[:, 0, 1] = p_psd[:, 0, 1]
	p2_psd[:, 0, 2] = p_psd[:, 0, 2]
	p2_psd[:, 1, 1] = p_psd[:, 1, 1]
	p2_psd[:, 1, 2] = p_psd[:, 1, 2]
	p2_psd[:, 2, 2] = p_psd[:, 2, 2]
	p2_psd[:, 1, 0] = p2_psd[:, 0, 1]
	p2_psd[:, 2, 0] = p2_psd[:, 0, 2]
	p2_psd[:, 2, 1] = p2_psd[:, 1, 2]

	p_psd[:, 0, 0] -= pmass * n_psd * v_psd[:, 0] * v_psd[:, 0]
	p_psd[:, 0, 1] -= pmass * n_psd * v_psd[:, 0] * v_psd[:, 1]
	p_psd[:, 0, 2] -= pmass * n_psd * v_psd[:, 0] * v_psd[:, 2]
	p_psd[:, 1, 1] -= pmass * n_psd * v_psd[:, 1] * v_psd[:, 1]
	p_psd[:, 1, 2] -= pmass * n_psd * v_psd[:, 1] * v_psd[:, 2]
	p_psd[:, 2, 2] -= pmass * n_psd * v_psd[:, 2] * v_psd[:, 2]
	p_psd[:, 1, 0] = p_psd[:, 0, 1]
	p_psd[:, 2, 0] = p_psd[:, 0, 2]
	p_psd[:, 2, 1] = p_psd[:, 1, 2]

	p_trace = np.trace(p_psd, axis1=1, axis2=2)
	t_psd = np.zeros(p_psd.shape)
	t_psd[...] = p_psd[...] / (kb * n_psd[:, np.newaxis, np.newaxis])
	t_psd[:, 1, 0] = t_psd[:, 1, 0]
	t_psd[:, 2, 0] = t_psd[:, 2, 0]
	t_psd[:, 2, 1] = t_psd[:, 2, 1]

	v_abs2 = np.linalg.norm(v_psd, axis=1) ** 2
	h_psd *= pmass / 2
	h_psd[:, 0] -= v_psd[:, 0] * p_psd[:, 0, 0] + v_psd[:, 1] * p_psd[:, 0, 1] + v_psd[:, 2] * p_psd[:, 0, 2]
	h_psd[:, 0] -= 0.5 * v_psd[:, 0] * p_trace + 0.5 * pmass * n_psd * v_abs2 * v_psd[:, 0]
	h_psd[:, 1] -= v_psd[:, 0] * p_psd[:, 0, 1] + v_psd[:, 1] * p_psd[:, 1, 1] + v_psd[:, 2] * p_psd[:, 1, 2]
	h_psd[:, 1] -= 0.5 * v_psd[:, 1] * p_trace + 0.5 * pmass * n_psd * v_abs2 * v_psd[:, 1]
	h_psd[:, 2] -= v_psd[:, 0] * p_psd[:, 0, 2] + v_psd[:, 1] * p_psd[:, 1, 2] + v_psd[:, 2] * p_psd[:, 2, 2]
	h_psd[:, 2] -= 0.5 * v_psd[:, 2] * p_trace + 0.5 * pmass * n_psd * v_abs2 * v_psd[:, 2]

	# Convert to typical units (/cc, km/s, nP, eV, and ergs/s/cm^2).
	n_psd /= 1e6
	# n_psd_e32              /= 1e6
	# n_psd_e32_phi_theta    /= 1e6
	v_psd /= 1e3
	p_psd *= 1e9
	p2_psd *= 1e9
	t_psd *= kb / qe
	h_psd *= 1e3

	# Construct TSeries
	n_psd = ts_scalar(pdist.time.data, n_psd)
	# n_psd_e32 = ts_scalar(pdist.time, n_psd_e32);
	# n_psd_skymap = ts_skymap(pdist.time.data, n_psd_e32_phi_theta,energy, phi.data, thetak);
	v_psd = ts_vec_xyz(pdist.time.data, v_psd)
	p_psd = ts_tensor_xyz(pdist.time.data, p_psd)
	p2_psd = ts_tensor_xyz(pdist.time.data, p2_psd)
	t_psd = ts_tensor_xyz(pdist.time.data, t_psd)
	h_psd = ts_vec_xyz(pdist.time.data, h_psd)

	return n_psd, v_psd, p_psd, p2_psd, t_psd, h_psd
