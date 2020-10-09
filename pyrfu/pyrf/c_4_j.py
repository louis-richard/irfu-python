#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_4_j.py

@author : Louis RICHARD
"""
from astropy import constants
from .avg_4sc import avg_4sc

from .c_4_grad import c_4_grad
from .cross import cross


def c_4_j(r_list=None, b_list=None):
	"""
	Calculate current density :math:`\\mathbf{J}` from using 4 spacecraft technique, the divergence of the magnetic
	field :math:`\\nabla . \\mathbf{B}`, magnetic field at the center of mass of the tetrahedron,
	:math:`\\mathbf{J}\\times\\mathbf{B}` force, part of the divergence of stress associated with curvature
	:math:`\\nabla.\\mathbf{T}_{shear}` and gradient of the magnetic pressure :math:`\\nabla P_b`.
	Where :

	.. math::

		\\mathbf{J} = \\frac{\\nabla \\times \\mathbf{B}}{\\mu_0}

		\\mathbf{J}\\times\\mathbf{B} = \\nabla.\\mathbf{T}_{shear} + \\nabla P_b

		\\nabla.\\mathbf{T}_{shear} = \\frac{(\\mathbf{B}.\\nabla) \\mathbf{B}}{\\mu_0}

		\\nabla P_b = \\nabla \\frac{B^2}{2\\mu_0}

	
	Parameters
	----------
	r_list : list of xarray.DataArray
		Time series of the spacecraft position [km]

	b_list : list of xarray.DataArray
		Time series of the magnetic field [nT]

	Returns
	-------
	j : xarray.DataArray
		Time series of the current density [A.m^{-2}]

	div_b : xarray.DataArray
		Time series of the divergence of the magnetic field [A.m^{-2}]

	b_avg : xarray.DataArray
		Time series of the magnetic field at the center of mass of the tetrahedron,
		sampled at 1st SC time steps [nT]

	jxb : xarray.DataArray
		Time series of the :math:`\\mathbf{J}\\times\\mathbf{B}` force [T.A].

	div_t_shear : xarray.DataArray
		Time series of the part of the divergence of stress associated with curvature units [T A/m^2].

	div_pb : xarray.DataArray
		Time series of the gradient of the magnetic pressure.

	Example
	-------
	>>> import numpy as np
	>>> from pyrfu import mms, pyrf
	>>> # Time interval
	>>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
	>>> # Spacecraft indices
	>>> mms_list = np.arange(1,5)
	>>> # Load magnetic field and spacecraft position
	>>> b_mms = [mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id) for mms_id in mms_list]
	>>> r_mms = [mms.get_data("R_gse", tint, mms_id) for mms_id in mms_list]
	>>> j, divB, B, jxB, divTshear, divPb = pyrf.c_4_j(r_mms, b_mms)

	Reference
	---------
	ISSI book  Eq. 14.16, 14.17 p. 353

	See also
	--------
	c_4_k

	"""

	mu0 = constants.mu0.value

	b_avg = avg_4sc(b_list)

	# Estimate divB/mu0. unit is A/m2
	div_b = c_4_grad(r_list, b_list, "div")

	# to get right units
	div_b *= 1.0e-3 * 1e-9 / mu0

	# estimate current j [A/m2]
	curl_b = c_4_grad(r_list, b_list, "curl")

	# to get right units [A.m^{-2}]
	j = curl_b * 1.0e-3 * 1e-9 / mu0

	# estimate jxB force [T A/m2]
	jxb = 1e-9 * cross(j, b_avg)

	# estimate divTshear = (1/muo) (B*div)B [T A/m2]
	b_div_b = c_4_grad(r_list, b_list, "bdivb")

	# to get right units [T.A.m^{-2}]
	div_t_shear = b_div_b * 1.0e-3 * 1e-9 * 1e-9 / mu0

	# estimate divPb = (1/muo) grad (B^2/2) = divTshear-jxB
	div_pb = div_t_shear - jxb

	return j, div_b, b_avg, jxb, div_t_shear, div_pb
