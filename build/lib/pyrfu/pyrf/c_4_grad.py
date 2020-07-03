import numpy as np
import xarray as xr

from .resample import resample
from .c_4_k import c_4_k
from .norm import norm
from .dot import dot
from .cross import cross



def c_4_grad(R1=None,R2=None,R3=None,R4=None,B1=None,B2=None,B3=None,B4=None,method="grad"):
	"""
	Calculate gradient of physical field using 4 spacecraft technique. Reference: ISSI book  Eq. 14.16, 14.17 p. 353

	See also : c_4_k
	
	Parameters :
		- R1...R4           [xarray]                Time series of the positions of the spacecraft
		- B1...B4           [xarray]                Time series of the magnetic field at the corresponding positions
		- method            [str]                   Method : grad (default), div, curl, bdivb, curv

	Returns :
		- out               [xarray]                Time serie of the derivative of the input field corresponding to 
													the method
	
	"""

	if not isinstance(R1,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(R2,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(R3,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(R4,xr.DataArray): raise TypeError("Inputs must be DataArrays")
		
	if not isinstance(B1,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(B2,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(B3,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(B4,xr.DataArray): raise TypeError("Inputs must be DataArrays")
		
	rR1 = resample(R1,B1)
	rR2 = resample(R2,B1)
	rR3 = resample(R3,B1)
	rR4 = resample(R4,B1)

	rB1 = resample(B1,B1)
	rB2 = resample(B2,B1)
	rB3 = resample(B3,B1)
	rB4 = resample(B4,B1)

	[K1,K2,K3,K4] = c_4_k(rR1,rR2,rR3,rR4)

	b = 0.25*(rB1 + rB2 + rB3 + rB4)

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

	if method.lower() == "grad":
		outdata = gradb

	elif method.lower() == "div":
		divb = np.zeros(len(rB1))
		for i in range(4):
			divb += eval("dot(K"+str(i+1)+",rB"+str(i+1)+").data")
		outdata = divb
		
	elif method.lower() == "curl":
		curlb = np.zeros((len(rB1),3))
		for i in range(4):
			curlb += eval("cross(K"+str(i+1)+",rB"+str(i+1)+").data")
		outdata = curlb
		
	elif method.lower() == "bdivb":
		bdivb = np.zeros(b.data.shape)
		bdivb[:,0]  = np.sum(b.data*gradb[:,0,:],axis=1)
		bdivb[:,1]  = np.sum(b.data*gradb[:,1,:],axis=1)
		bdivb[:,2]  = np.sum(b.data*gradb[:,2,:],axis=1)
		outdata     = bdivb
		
	elif method.lower() == "curv":
		bhat1   = norm(rB1)
		bhat2   = norm(rB2)
		bhat3   = norm(rB3)
		bhat4   = norm(rB4)
		curv    = c_4_grad(rR1,rR2,rR3,rR4,bhat1,bhat2,bhat3,bhat4,method="bdivb")
		outdata = curv.data
		
	if len(outdata.shape) == 1:
		out = xr.DataArray(outdata,coords=[rB1.time],dims=["time"])

	elif len(outdata.shape) == 2:
		out = xr.DataArray(outdata,coords=[rB1.time,["x","y","z"]],dims=["time","comp"])

	elif len(outdata.shape) == 3:
		out = xr.DataArray(outdata,coords=[rB1.time,["x","y","z"],["x","y","z"]],dims=["time","vcomp","hcomp"])

	return out