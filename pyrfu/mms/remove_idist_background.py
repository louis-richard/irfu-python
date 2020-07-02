import numpy as np
import xarray as xr
from astropy import constants

from ..pyrf.ts_tensor_xyz import ts_tensor_xyz



def remove_idist_background(N_i=None,V_gse_i=None,P_gse_i=None,N_bg_i=None,P_bg_i=None):
	"""
	Removes penetrating radiation background from ion moments

	Parameters :
		- N_i               [xarray]                Ion number density
		- V_gse_i           [xarray]                Ion bulk velocity
		- P_gse_i           [xarray]                Ion pressure tensor 
		- N_bg_i            [xarray]                Background ion number density
		- P_bg_i            [xarray]                Background ion pressure scalar

	Returns :
		- N_i_new           [xarray]                Corrected ion number density
		- V_gse_i_new       [xarray]                Corrected ion bulk velocity
		- P_gse_i_new       [xarray]                Corrected ion pressure tensor


	References:
		- MMS DIS Penetrating radiation correction methods: 
			https://lasp.colorado.edu/galaxy/display/MFDPG/Penetrating+Radiation+in+DIS+Data

	"""
	
	if (N_i is None) or (V_gse_i is None) or (P_gse_i is None) or (N_bg_i is None) or (P_bg_i is None):
		raise ValueError("remove_idist_background requires exactly 5 arguments")

	if not isinstance(N_i, xr.DataArray):
		raise TypeError("N_i must be a DataArray")

	if not isinstance(V_gse_i, xr.DataArray):
		raise TypeError("V_gse_i must be a DataArray")

	if not isinstance(P_gse_i, xr.DataArray):
		raise TypeError("P_gse_i must be a DataArray")

	if not isinstance(N_bg_i, xr.DataArray):
		raise TypeError("N_bg_i must be a DataArray")

	if not isinstance(P_bg_i, xr.DataArray):
		raise TypeError("P_bg_i must be a DataArray")

	# Number density
	N_i_new      = N_i - N_bg_i.data

	# Bulk velocity
	V_gse_i_new = V_gse_i
	V_gse_i_new *= N_i/N_i_new
	
	# Pressure tensor
	P_gse_i_new_data = np.zeros(P_gse_i.shape)
	
	# P_xx_i
	P_gse_i_new_data[:,0,0] += P_gse_i.data[:,0,0]
	P_gse_i_new_data[:,0,0] -= P_bg_i.data
	P_gse_i_new_data[:,0,0] += constants.m_p.value*N_i.data*V_gse_i.data[:,0]*V_gse_i.data[:,0]
	P_gse_i_new_data[:,0,0] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,0]*V_gse_i_new.data[:,0]

	# P_yy_i
	P_gse_i_new_data[:,1,1] += P_gse_i.data[:,1,1]
	P_gse_i_new_data[:,1,1] -= P_bg_i.data
	P_gse_i_new_data[:,1,1] += constants.m_p.value*N_i.data*V_gse_i.data[:,1]*V_gse_i.data[:,1]
	P_gse_i_new_data[:,1,1] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,1]*V_gse_i_new.data[:,1]

	# P_zz_i
	P_gse_i_new_data[:,2,2] += P_gse_i.data[:,2,2]
	P_gse_i_new_data[:,2,2] -= P_bg_i.data
	P_gse_i_new_data[:,2,2] += constants.m_p.value*N_i.data*V_gse_i.data[:,2]*V_gse_i.data[:,2]
	P_gse_i_new_data[:,2,2] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,2]*V_gse_i_new.data[:,2]

	# P_xy_i & P_yx_i
	P_gse_i_new_data[:,0,1] += P_gse_i.data[:,0,1]
	P_gse_i_new_data[:,0,1] += constants.m_p.value*N_i.data*V_gse_i.data[:,0]*V_gse_i.data[:,1]
	P_gse_i_new_data[:,0,1] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,0]*V_gse_i_new.data[:,1]
	P_gse_i_new_data[:,1,0] = P_gse_i_new_data[:,0,1]

	# P_xz_i & P_zx_i
	P_gse_i_new_data[:,0,2] += P_gse_i.data[:,0,2]
	P_gse_i_new_data[:,0,2] += constants.m_p.value*N_i.data*V_gse_i.data[:,0]*V_gse_i.data[:,2]
	P_gse_i_new_data[:,0,2] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,0]*V_gse_i_new.data[:,2]
	P_gse_i_new_data[:,2,0] = P_gse_i_new_data[:,0,2]

	# P_yz_i & P_zy_i
	P_gse_i_new_data[:,1,2] += P_gse_i.data[:,1,2]
	P_gse_i_new_data[:,1,2] += constants.m_p.value*N_i.data*V_gse_i.data[:,1]*V_gse_i.data[:,2]
	P_gse_i_new_data[:,1,2] -= constants.m_p.value*N_i_new.data*V_gse_i_new.data[:,1]*V_gse_i_new.data[:,2]
	P_gse_i_new_data[:,2,1] = P_gse_i_new_data[:,1,2]

	P_gse_i_new = ts_tensor_xyz(P_gse_i.time.data,P_gse_i_new_data)
	
	return (N_i_new,V_gse_i_new,P_gse_i_new)