import xarray as xr

from .cross import cross
from .dot import dot



def c_4_k(R=None):
	"""
	Calculate reciprocal vectors in barycentric coordinates. Reference: ISSI book 14.7
	
	Parameters :
		R : list of DataArray
			Position of the spacecrafts

	Returns :
		K : list of DataArray
			Reciprocal vectors in barycentric coordinates
	
	Note : 
		The units of reciprocal vectors are the same as [1/r]

	"""

	if R is None:
		raise ValueError("c_4_k requires one argument")

	if not isinstance(R,list) or len(R) != 4:
		raise TypeError("R must be a list of 4SC position")

	for i in range(4):
		if not isinstance(R[i],xr.DataArray):
			raise TypeError("Spacecraft position must be DataArray")
		
	r1, r2, r3, r4 = R

	r = [r1,r2,r3,r4,r1,r2,r3]
	K = [None]*4

	for j in range(4):
		cc 		= cross(r[2+j]-r[1+j],r[3+j]-r[1+j])
		dr12 	= r[j]-r[1+j]
		K[j] 	= cc/dot(cc,dr12)

	return K