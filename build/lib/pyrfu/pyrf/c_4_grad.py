import numpy as np
import xarray as xr

from .resample import resample
from .c_4_k import c_4_k
from .normalize import normalize
from .dot import dot
from .cross import cross



def c_4_grad(R=None, B=None,method="grad"):
	"""
	Calculate gradient of physical field using 4 spacecraft technique. 
	
	Parameters :
		R : list of DataArray
			Time series of the positions of the spacecraft

		B : list of DataArray
			Time series of the magnetic field at the corresponding positions

		method : str
			Method flag : 
				"grad" -> compute gradient (default)
				"div" -> compute divergence
				"curl" -> compute curl
				"bdivb" -> compute b.div(b)
				"curv" -> compute curvature

	Returns :
		- out : DataArray
			Time series of the derivative of the input field corresponding to the method

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
		>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
		>>> gradB = pyrf.c_4_grad(Rxyz,Bxyz,"grad")

	Reference : 
		ISSI book  Eq. 14.16, 14.17 p. 353

	See also : 
		c_4_k
	
	"""

	if (R is None) or (B is None):
		raise ValueError("c_4_grad requires at least 2 arguments")

	if not isinstance(R,list) or len(R) != 4:
		raise TypeError("R must be a list of 4SC position")

	if not isinstance(B,list) or len(B) != 4:
		raise TypeError("B must be a list of 4SC position")

	for i in range(4):
		if not isinstance(R[i],xr.DataArray):
			raise TypeError("Spacecraft position must be DataArray")

		if not isinstance(B[i],xr.DataArray):
			raise TypeError("Magnetic field must be DataArray")

	# Unpack spacecraft position and magnetic field
	R1, R2, R3, R4 = R 
	B1, B2, B3, B4 = B
		
	# Resample with respect to 1st spacecraft
	rR1 = resample(R1,B1)
	rR2 = resample(R2,B1)
	rR3 = resample(R3,B1)
	rR4 = resample(R4,B1)

	rB1 = resample(B1,B1)
	rB2 = resample(B2,B1)
	rB3 = resample(B3,B1)
	rB4 = resample(B4,B1)

	# Compute reciprocal vectors in barycentric coordinates (see c_4_k)
	K1, K2, K3, K4 = c_4_k(rR1,rR2,rR3,rR4)

	# Magnetic field at the center of mass of the tetrahedron
	b = 0.25*(rB1 + rB2 + rB3 + rB4)

	# Gradient of scalar/vector
	if len(rB1.shape) == 1:
		gradb = np.zeros((len(rB1),3))
		for i in range(1,5):
			gradb += eval("K"+str(i)+".data*np.tile(rB"+str(i)+".data,(3,1)).T")
	else :
		gradb = np.zeros((len(rB1),3,3))
		gradb[:,0,0] = (K1[:,0]*rB1[:,0] + K2[:,0]*rB2[:,0] + K3[:,0]*rB3[:,0] + K4[:,0]*rB4[:,0]).data
		gradb[:,1,0] = (K1[:,0]*rB1[:,1] + K2[:,0]*rB2[:,1] + K3[:,0]*rB3[:,1] + K4[:,0]*rB4[:,1]).data
		gradb[:,2,0] = (K1[:,0]*rB1[:,2] + K2[:,0]*rB2[:,2] + K3[:,0]*rB3[:,2] + K4[:,0]*rB4[:,2]).data
		gradb[:,0,1] = (K1[:,1]*rB1[:,0] + K2[:,1]*rB2[:,0] + K3[:,1]*rB3[:,0] + K4[:,1]*rB4[:,0]).data
		gradb[:,1,1] = (K1[:,1]*rB1[:,1] + K2[:,1]*rB2[:,1] + K3[:,1]*rB3[:,1] + K4[:,1]*rB4[:,1]).data
		gradb[:,2,1] = (K1[:,1]*rB1[:,2] + K2[:,1]*rB2[:,2] + K3[:,1]*rB3[:,2] + K4[:,1]*rB4[:,2]).data
		gradb[:,0,2] = (K1[:,2]*rB1[:,0] + K2[:,2]*rB2[:,0] + K3[:,2]*rB3[:,0] + K4[:,2]*rB4[:,0]).data
		gradb[:,1,2] = (K1[:,2]*rB1[:,1] + K2[:,2]*rB2[:,1] + K3[:,2]*rB3[:,1] + K4[:,2]*rB4[:,1]).data
		gradb[:,2,2] = (K1[:,2]*rB1[:,2] + K2[:,2]*rB2[:,2] + K3[:,2]*rB3[:,2] + K4[:,2]*rB4[:,2]).data

	# Gradient
	if method.lower() == "grad":
		outdata = gradb

	# Divergence
	elif method.lower() == "div":
		divb = np.zeros(len(rB1))
		for i in range(4):
			divb += eval("dot(K"+str(i+1)+",rB"+str(i+1)+").data")
		outdata = divb
	
	# Curl
	elif method.lower() == "curl":
		curlb = np.zeros((len(rB1),3))
		for i in range(4):
			curlb += eval("cross(K"+str(i+1)+",rB"+str(i+1)+").data")
		outdata = curlb
		
	# B.div(B)
	elif method.lower() == "bdivb":
		bdivb = np.zeros(b.data.shape)
		bdivb[:,0]  = np.sum(b.data*gradb[:,0,:],axis=1)
		bdivb[:,1]  = np.sum(b.data*gradb[:,1,:],axis=1)
		bdivb[:,2]  = np.sum(b.data*gradb[:,2,:],axis=1)
		outdata     = bdivb
		
	# Curvature
	elif method.lower() == "curv":
		bhat1   = normalize(rB1)
		bhat2   = normalize(rB2)
		bhat3   = normalize(rB3)
		bhat4   = normalize(rB4)
		curv    = c_4_grad([rR1,rR2,rR3,rR4],[bhat1,bhat2,bhat3,bhat4],method="bdivb")
		outdata = curv.data
		
	if len(outdata.shape) == 1:
		out = xr.DataArray(outdata,coords=[rB1.time],dims=["time"])

	elif len(outdata.shape) == 2:
		out = xr.DataArray(outdata,coords=[rB1.time,["x","y","z"]],dims=["time","comp"])

	elif len(outdata.shape) == 3:
		out = xr.DataArray(outdata,coords=[rB1.time,["x","y","z"],["x","y","z"]],dims=["time","vcomp","hcomp"])

	return out