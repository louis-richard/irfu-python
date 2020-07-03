from astropy import constants

from .c_4_grad import c_4_grad
from .cross import cross

def c_4_j(r1,r2,r3,r4,b1,b2,b3,b4):
	"""
	Calculate current from using 4 spacecraft technique in addition one can obtain average magnetic field and jxB 
	values. Estimate also divergence B as the error estimate
	

	%
	%  [j,divB,B,jxB,curvature,divTshear,divPb] = c_4_j(R1,R2,R3,R4,B1,B2,B3,B4)
	%  [j,divB,B,jxB,divTshear,divPb] = c_4_j('R?','B?')
	%  
	%
	%  r1..r4 are row vectors
	%         column 1     is time
	%         column 2-4   are satellite position in km
	%  b1..b4 are row vectors
	%         column 1     is time b1 time is used for all interpolation of r1..r4 and b2..b4
	%         column 2-4   is magnetic field components in nT
	%  j      is row vector,
	%         column 1     time
	%         column 2-4   current, units A/m^2
	%  divB   column 1     time
	%         column 2     div(B)/mu0, units A/m^2
	%  B      - average magnetic field, sampled at b1 time steps [nT]
	%  jxB    - j x B force [T A]
	%         - jxB=(1/muo) ( (B div)B + grad (B^2/2) )= divTshear+divPb
	%  divTshear = (1/muo) (B div) B.  the part of the divergence of stress 
	%                                   associated with curvature units [T A/m^2]
	%  divPb = (1/muo) grad(B^2/2). gradient of magnetic pressure
	% 
	%   See also C_4_K
	%
	%  Reference: ISSI book  Eq. 14.16, 14.17

	% TODO fix that it works for vector inputs without time column!
	"""


	mu0 = constants.mu0.value

	# Estimate divB/mu0. unit is A/m2
	divB, B = c_4_grad(Rxyz1,Rxyz2,Rxyz3,Rxyz4,Bxyz1,Bxyz2,Bxyz3,Bxyz4,"divr1")
	divB, B = c_4_grad(r1,r2,r3,r4,b1,b2,b3,b4,"divr1")
	divB 	*= 1.0e-3*1e-9/mu0 									# to get right units why 

	# estimate current j [A/m2]
	curl_B 	= c_4_grad(r1,r2,r3,r4,b1,b2,b3,b4,"curl")
	j		*= 1.0e-3*1e-9/mu0 									# to get right units [A/m2]

	# estimate jxB force [T A/m2]
	jxB = 1e-9*cross(j,B)										# to get units [T A/m2]


	# estimate divTshear = (1/muo) (B*div)B [T A/m2]
	BdivB 		= c_4_grad(r1,r2,r3,r4,b1,b2,b3,b4,"bdivb")
	divTshear	*= 1.0e-3*1e-9*1e-9/mu0 									# to get right units [T A/m2]

	# estimate divPb = (1/muo) grad (B^2/2) = divTshear-jxB
	divPb = divTshear-jxB

	return (j,divB,B,jxB,divTshear,divPb)

