from astropy import constants
from .avg_4sc import avg_4sc

from .c_4_grad import c_4_grad
from .cross import cross


def c_4_j(r_list=None, b_list=None):
	"""
	Calculate current from using 4 spacecraft technique in addition one can obtain average magnetic field and jxB 
	values. Estimate also divergence B as the error estimate
	
	Parameters :
		r_list : list of DataArrays
			Time series of the spacecraft position [km]

		b_list : list of DataArray
			Time series of the magnetic field [nT]

	Returns :
		j : DataArray
			Time series of the current density j = curl(B)/mu0 [A.m^{-2}]

		div_b : DataArray
			Time series of the divergence of the magnetic field div(B)/mu0 [A.m^{-2}]

		b_avg : DataArray
			Time series of the magnetic field at the center of mass of the tetrahedron, 
			sampled at 1st SC time steps [nT]

		jxb : DataArray
			Time series of the j x B force [T.A]. jxB = ((B.div)B + grad(B^2/2))/mu0 = divTshear+divPb

		div_t_shear : DataArray
			Time series of the part of the divergence of stress associated with curvature units 
			divTshear = (1/muo) (B div) B. [T A/m^2]

		div_pb : DataArray
			Time series of the gradient of the magnetic pressure divPb = grad(B^2/2)/mu0

	Example : 
		>>> import numpy as np
		>>> from pyrfu import mms, pyrf
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

	b_avg = avg_4sc(b_list)

	# Estimate divB/mu0. unit is A/m2
	div_b = c_4_grad(r_list, b_list, "div")

	# to get right units
	div_b *= 1.0e-3*1e-9/mu0

	# estimate current j [A/m2]
	curl_b = c_4_grad(r_list, b_list, "curl")

	# to get right units [A.m^{-2}]
	j = curl_b*1.0e-3*1e-9/mu0

	# estimate jxB force [T A/m2]
	jxb = 1e-9*cross(j, b_avg)

	# estimate divTshear = (1/muo) (B*div)B [T A/m2]
	b_div_b = c_4_grad(r_list, b_list, "bdivb")

	# to get right units [T.A.m^{-2}]
	div_t_shear = b_div_b*1.0e-3*1e-9*1e-9/mu0

	# estimate divPb = (1/muo) grad (B^2/2) = divTshear-jxB
	div_pb = div_t_shear - jxb

	return j, div_b, b_avg, jxb, div_t_shear, div_pb
