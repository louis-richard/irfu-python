from astropy import constants

from .c_4_grad import c_4_grad
from .cross import cross
from .resample import resample


def c_4_j(R=None, B=None):
	"""
	Calculate current from using 4 spacecraft technique in addition one can obtain average magnetic field and jxB 
	values. Estimate also divergence B as the error estimate
	
	Parameters :
		R : list of DataArrays
			Time series of the spacecraft position [km]

		B : list of DataArray
			Time series of the magnetic field [nT]

	Returns :
		j : DataArray
			Time series of the current density j = curl(B)/mu0 [A.m^{-2}]

		divB : DataArray
			Time series of the divergence of the magnetic field div(B)/mu0 [A.m^{-2}]

		Bav : DataArray
			Time series of the magnetic field at the center of mass of the tetrahedron, 
			sampled at 1st SC time steps [nT]

		jxB : DataArray
			Time series of the j x B force [T.A]. jxB = ((B.div)B + grad(B^2/2))/mu0 = divTshear+divPb

		divTshear : DataArray
			Time series of the part of the divergence of stress associated with curvature units 
			divTshear = (1/muo) (B div) B. [T A/m^2]

		divPb : DataArray
			Time series of the gradient of the magnetic pressure divPb = grad(B^2/2)/mu0

	Example : 
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
		>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
		>>> j, divB, B, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)

	Reference : 
		ISSI book  Eq. 14.16, 14.17 p. 353

	See also : 
		c_4_k

	"""




	mu0 = constants.mu0.value

	Bav = (B[0]+resample(B[1],B[0])+resample(B[2],B[0])+resample(B[3],B[0]))/4

	# Estimate divB/mu0. unit is A/m2
	divB 	= c_4_grad(R,B,"div")
	divB 	*= 1.0e-3*1e-9/mu0 									# to get right units why 

	# estimate current j [A/m2]
	curl_B 	= c_4_grad(R,B,"curl")
	j		= curl_B*1.0e-3*1e-9/mu0 							# to get right units [A.m^{-2}]

	# estimate jxB force [T A/m2]
	jxB = 1e-9*cross(j,Bav)										# to get units [T.A.m^{-2}]


	# estimate divTshear = (1/muo) (B*div)B [T A/m2]
	BdivB 		= c_4_grad(R,B,"bdivb")
	divTshear	= BdivB*1.0e-3*1e-9*1e-9/mu0 					# to get right units [T.A.m^{-2}]

	# estimate divPb = (1/muo) grad (B^2/2) = divTshear-jxB
	divPb = divTshear-jxB

	return (j,divB,Bav,jxB,divTshear,divPb)

