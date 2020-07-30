import numpy as np
from .ts_skymap import ts_skymap


def dist_append(inp0=None,inp1=None):
	"""
	Concatenate two distribution skymaps along the time axis

	Note : the time series have to be in the correct time order

	Parameters :
		inp1 : DataArray
			3D skymap distribution at early times 

		inp2 : DataArray
			3D skymap distribution at late times 

	Returns :
		out : DataArray
			3D skymap of the concatenated 3D skymaps 

	"""

	if inp0 is None or inp1 is None:
		raise ValueError("dist_append requires at least two arguments")
		
	if not isinstance(inp0,xr.Dataset) or not isinstance(inp1,xr.Dataset):
		raise TypeError("inputs must be datasets")
		
	nt0 = len(inp0.data)
	nt1 = len(inp1.data)

	nEn = inp0.energy.shape[1]
	try :
		nPh = inp0.phi.shape[1]
	except IndexError:
		nPh = len(inp0.phi)

	nTh = len(inp0.theta)

	# time
	time = np.zeros(nt0+nt1)
	time[:nt0] = inp0.time.data
	time[nt0:nt0+nt1] = inp1.time.data

	# Azymutal angle
	if inp0.phi.ndim == 2:
		phi = np.zeros((nt0+nt1,nPh))
		phi[:nt0,...] = inp0.phi.data
		phi[nt0:nt0+nt1,...] = inp1.phi.data
	else :
		phi = inp0.phi.data

	# Elevation angle
	theta = inp0.theta.data

	# distribution
	data = np.zeros((nt0+nt1,nEn,nPh,nTh))
	data[:nt0,...] = inp0.data
	data[nt0:nt0+nt1,...] = inp1.data

	if "delta_energy_plus" in inp0.attrs:
		delta_energy_plus = np.zeros((nt0+nt1,nEn))
		delta_energy_plus[:nt0,...] = inp0.attrs["delta_energy_plus"].data
		delta_energy_plus[nt0:nt0+nt1,...] = inp1.attrs["delta_energy_plus"].data

	if "delta_energy_minus" in inp0.attrs:
		delta_energy_minus = np.zeros((nt0+nt1,nEn))
		delta_energy_minus[:nt0,...] = inp0.attrs["delta_energy_minus"].data
		delta_energy_minus[nt0:nt0+nt1,...] = inp1.attrs["delta_energy_minus"].data

	# Energy
	#pdb.set_trace()
	if inp0.attrs["tmmode"] == "brst":
		stepTable = np.hstack([inp0.attrs["esteptable"],inp1.attrs["esteptable"]])
		out = ts_skymap(time, data, None, phi, theta, energy0=inp0.energy0,energy1=inp0.energy1, esteptable=stepTable)
	else :
		energy = np.zeros((nt0+nt1,nEn))
		energy[:nt0,...] = inp0.energy.data
		energy[nt0:nt0+nt1,...] = inp1.energy.data
		out = ts_skymap(time, data, energy, phi, theta)
	


	# attributes
	attrs = {}
	attrs = inp0.attrs
	attrs.pop("esteptable")
	if "delta_energy_minus" in inp0.attrs:
		attrs["delta_energy_minus"] = delta_energy_minus

	if "delta_energy_plus" in inp0.attrs:
		attrs["delta_energy_plus"] = delta_energy_plus

	#out = xr.Dataset(outdict,attrs=attrs)
	
	for k in attrs:
		out.attrs[k] = attrs[k]

	return out