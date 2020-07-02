import xarray as xr

from .cross import cross
from .dot import dot



def c_4_k(r1=None,r2=None,r3=None,r4=None):
	"""
	Calculate reciprocal vectors in barycentric coordinates. Reference: ISSI book 14.7
	
	Note : The units of reciprocal vectors are the same as [1/r]

	Parameters :
		- r1...r4           [xarray]                Position of the spacecrafts

	Returns :
		- k1...k4           [xarray]                Reciprocal vectors in barycentric coordinates
	
	"""

	if not isinstance(r1,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(r2,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(r3,xr.DataArray): raise TypeError("Inputs must be DataArrays")
	if not isinstance(r4,xr.DataArray): raise TypeError("Inputs must be DataArrays")
		
	R = [r1,r2,r3,r4,r1,r2,r3]
	for j in range(4):
		cc      = cross(R[2+j]-R[1+j],R[3+j]-R[1+j])
		dr12    = R[j]-R[1+j]
		exec("vars()['K"+str(j+1)+"'] = cc/dot(cc,dr12)")

	return(vars()['K1'],vars()['K2'],vars()['K3'],vars()['K4'])