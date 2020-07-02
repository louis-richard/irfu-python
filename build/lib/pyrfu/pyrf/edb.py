import numpy as np
import xarray as xr

from .resample import resample
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz




def edb(inp=None, b0=None, angle_lim=20, flag_method="E.B=0"):
	"""
	Compute Ez under assumption E.B=0 or E.B~=0

	Parameters :
		- inp               [xarray]                Input time serie
		- b0                [xarray]                Background magnetic field
		- flag_method       [str]                   Assumption on the direction of the measured electric field
		- angle_lim         [float]                 B angle with respect to the spin plane should be less than 
													angle_lim degrees otherwise Ez is set to 0
	
	Returns :
		- ed                [xarray]                E field output
		- d                 [xarray]                B elevation angle above spin plane

	"""

	if inp is None:
		raise ValueError("edb requires at least two inputs")
	if b0 is None:
		raise ValueError("edb requires at least two inputs")
	
	if flag_method == "Eperp+NaN":
		defaultValue = np.nan
		flag_method = "E.B=0"

	if len(b0) != len(inp):
		b0 = resample(b0,inp)

	le = inp.shape[1]
	lb = b0.shape[1]

	bd = b0.data
	ed = inp.data
	ed[:,-1] *= defaultValue

	if flag_method.lower() == "e.b=0":
		# Calculate using assumption E.B=0
		d = np.arctan2(bd[:,2],np.sqrt(bd[:,0]**2+bd[:,1]**2))*180/np.pi

		ind = np.abs(d) > angle_lim
		if True in ind:
			ed[ind,2] = -(ed[ind,0]*bd[ind,0]+ed[ind,1]*bd[ind,1])/bd[ind,2]

	elif flag_method.lower() == "epar":
		# Calculate using assumption E.B=0
		d = np.arctan2(bd[:,2],np.sqrt(bd[:,0]**2+bd[:,1]**2))*180/np.pi

		ind = np.abs(d) < angle_lim


		if True in ind:
			ed[ind,2] = (ed[ind,0]*bd[ind,0]+ed[ind,1]*bd[ind,1])
			ed[ind,2] = ed[ind,2]*bd[ind,2]/(bd[ind,0]**2+bd[ind,1]**2);

	ed  = ts_vec_xyz(inp.time.data,ed,{"UNITS":inp.attrs["UNITS"]})
	d   = ts_scalar(inp.time.data, d,{"UNITS" : "degres"})
	return (ed,d)