.. pyrfu documentation master file, created by
   sphinx-quickstart on Thu Jul 16 10:17:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################
Welcome to pyrfu's documentation!
##################################
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   support
   

************
:mod:`pyrf`
************


.. module:: pyrf
   :platform: Unix, Windows
   :synopsis: Generic functions
.. moduleauthor:: Louis Richard <louis.richard@irfu.se>

:func:`pyrf.agyro_coeff`
=========================
.. py:function:: agyro_coeff(P=None)

Computes agyrotropy coefficient [1]_

Parameters :
	P : DataArray
		Time series of the pressure tensor
	
Returns :
	Q : DataArray
		Time series of the agyrotropy coefficient of the specie

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and electron pressure tensor
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
	>>> Pexyz = mms.get_data("Pe_gse_fpi_fast_l2",Tint,1)
	>>> 
	>>> # Rotate electron pressure tensor to field aligned coordinates
	>>> Pexyzfac = pyrf.rotate_tensor(Pexyz,"fac",Bxyz,"pp")
	>>> 
	>>> # Compute agyrotropy coefficient
	>>> Qe = pyrf.agyro_coeff(Pexyzfac)

.. [1] M. Swisdak "Quantifying gyrotropy in magnetic reconnection" vol. 43, pp. 43-49, 2016


:func:`avg_4sc`
=========================
.. py:function:: avg_4sc(B=None)
	
Computes the input quantity at the center of mass of the MMS tetrahedron

Parameters :
	*B* : list of DataArray
		List of the time series of the quantity for each spacecraft

Returns :
	*Bavg* : DataArray
		Time series of the input quantity a the enter of mass of the MMS tetrahedron

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
	>>> 
	>>> # Compute the magnetic field at the center of mass of the MMS tetrahedron
	>>> Bxyzavg = pyrf.avg_4sc(Bxyz)

	

:func:`c_4_grad`
=========================
.. py:function:: c_4_grad(R=None, B=None,method="grad")

Calculate gradient of physical field using 4 spacecraft technique. 

Parameters :
	*R* : list of DataArray
		Time series of the positions of the spacecraft

	*B* : list of DataArray
		Time series of the magnetic field at the corresponding positions

	*method* : str
		Method flag : 
			"grad" -> compute gradient (default)
			"div" -> compute divergence
			"curl" -> compute curl
			"bdivb" -> compute b.div(b)
			"curv" -> compute curvature

Returns :
	*out* : DataArray
		Time series of the derivative of the input field corresponding to the method

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and spacecraft position
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
	>>> 
	>>> # Compute gradient of the magnetic field
	>>> gradB = pyrf.c_4_grad(Rxyz,Bxyz,"grad")

Reference : 
	ISSI book  Eq. 14.16, 14.17 p. 353

See also : 
	:func:`c_4_k`


:func:`c_4_j`
=============
.. py:function:: c_4_j(R=None, B=None)
	
Calculate current from using 4 spacecraft technique in addition one can obtain average magnetic field and :math:`\mathbf{J}\times\mathbf{B}` 
values. Estimate also divergence B as the error estimate

Parameters :
	*R* : list of DataArrays
		Time series of the spacecraft position [km]

	*B* : list of DataArray
		Time series of the magnetic field [nT]

Returns :
	*j* : DataArray
		Time series of the current density :math:`\mathbf{J} = \mu_0^{-1} \nabla \times \mathbf{B}`

	*divB* : DataArray
		Time series of the divergence of the magnetic field :math:`\mu_0^{-1} \nabla . \mathbf{B}`

	*Bav* : DataArray
		Time series of the magnetic field at the center of mass of the tetrahedron, 
		sampled at 1st SC time steps [nT] 

	*jxB* : DataArray
		Time series of the :math:`\mathbf{J}\times\mathbf{B} = \mu_0^{-1}\left[\left(\mathbf{B}.\nabla\right)\mathbf{B} + \nabla \left(\frac{B^2}{2}\right)\right]` force.

	*divTshear* : DataArray
		Time series of the part of the divergence of stress associated with curvature units :math:`\mu_0^{-1}\left(\mathbf{B}.\nabla\right)\mathbf{B}`.

	*divPb* : DataArray
		Time series of the gradient of the magnetic pressure :math:`\mu_0^{-1}\nabla \left(\frac{B^2}{2}\right)`

Example : 
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and spacecraft position
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
	>>> 
	>>> # Compute current density, etc
	>>> j, divB, B, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)

Reference : 
	ISSI book  Eq. 14.16, 14.17 p. 353

See also : 
	:func:`c_4_k`

:func:`c_4_k`
.. py:function:: c_4_k(R=None)

Calculate reciprocal vectors in barycentric coordinates.

Parameters :
	*R* : list of DataArray
		Time series of the position of the spacecrafts

Returns :
	*K* : list of DataArray
		Time series of the reciprocal vectors in barycentric coordinates

Reference: 
	ISSI book 14.7

Note : 
	The units of reciprocal vectors are the same as [1/r]


:func:`calc_disprel_tm`
========================
.. py:function:: calc_disprel_tm(V=None, dV=None, T=None, dT=None)

Computes dispersion relation from velocities and period given by the timing method

Parameters :
	*V* : DataArray
		Time series of the velocities

	*dV* : DataArray
		Time series of the error on velocities

	*T* : DataArray
		Time series of the periods
	
	*dT* : DataArray
		Time series of the error on period

Returns :
	*out* : Dataset
		DataSet containing the frequency, the wavelength, the wavenumber. Also includes the errors and the fit 
		(e.g Vph phase velocity)

See also :
	c_4_v_xcorr

:func:`calc_dt`
===============
.. py:function:: calc_dt(inp=None)

Computes time step of the input time series

Parameters :
	*inp* : DataArray
		Time series of the input variable

Returns :
	*out* : float
		Time step in seconds

:func:`calc_fs`
================
.. py:function:: calc_fs(inp=None)

Computes the sampling frequency of the input time series

Parameters :
	*inp* : DataArray
		Time series of the input variable

Returns :
	*out* : float
		Sampling frequency in Hz

:func:`convert_fac`
.. py:function:: convert_fac(inp=None, Bbgd=None, r=np.array([1,0,0]))

Transforms to a field-aligned coordinate (FAC) system defined as:
R_parallel_z aligned with the background magnetic field
R_perp_y defined by R_parallel cross the position vector of the spacecraft (nominally eastward at the equator)
R_perp_x defined by R_perp_y cross R_par
If inp is one vector along r direction, out is inp[perp, para] projection

Parameters :
	*inp* : DataArray
		Time series of the input field

	*Bbgd* : DataArray
		Time series of the background magnetic field

	*r* : DataArray/ndarray/list
		Position vector of spacecraft

Returns :
	*out* : DataArray
		Time series of the input field in field aligned coordinates system

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field (FGM) and electric field (EDP)
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
	>>> 
	>>> # Convert to field aligned coordinates
	>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])

Note : 
	all input parameters must be in the same coordinate system

:func:`cross`
==============
.. py:function:: cross(inp1=None, inp2=None)
	
Computes cross product of two fields.

Parameters :
	inp1 : DataArray
		Time series of the first field :math:`\mathbf{Y}`

	inp2 : DataArray
		Time series of the second field :math:`\mathbf{Y}`

Returns :
	out : DataArray
		Time series of the cross product :math:`\mathbf{Z} = \mathbf{X}\times\mathbf{Y}` 

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic and electric fields
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,ic)
	>>> 
	>>> # Magnitude of the magnetic field
	>>> Bmag = pyrf.norm(Bxyz)
	>>> 
	>>> # Compute ExB drit velocity
	>>> ExBxyz = pyrf.cross(Exyz,Bxyz)/Bmag**2

:func:`dec_parperp`
====================
.. py:function:: dec_parperp(inp=None, b0=None, flagspinplane=False)
	
Decomposes a vector into par/perp to B components. If flagspinplane decomposes components to the projection of B 
into the XY plane. Alpha_XY gives the angle between B0 and the XY plane.

Parameters :
	*inp* : DataArray
		Time series of the field to decompose

	*b0* : DataArray
		Time series of the background magnetic field

Options :
	*flagspinplane* : bool
		Flag if True gives the projection in XY plane
	
Returns :
	*apar* : DataArray
		Time series of the input field parallel to the background magnetic field

	*aperp* : DataArray
		Time series of the input field perpendicular to the background magnetic field

	*alpha* : DataArray
		Time series of the angle between the background magnetic field and the XY plane

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field (FGM) and electric field (EDP)
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
	>>> 
	>>> # Decompose Exyz into parallel and perpendicular to Bxyz components
	>>> Epar, Eperp, alpha = pyrf.dec_parperp(Exyz,Bxyz)

	
:func:`dist_append`
====================
.. py:function:: dist_append(inp0=None,inp1=None)

Concatenate two distribution skymaps along the time axis

Note : the time series have to be in the correct time order

Parameters :
	*inp1* : DataArray
		3D skymap distribution at early times 

	*inp2* : DataArray
		3D skymap distribution at late times 

Returns :
	*out* : DataArray
		3D skymap of the concatenated 3D skymaps

:func:`dot`
============
.. py:function:: dot(inp1=None, inp2=None)
	
Computes dot product of two fields

Parameters : 
	inp1 : DataArray
		Time series of the first field :math:`\mathbf{X}`

	inp2 : DataArray
		Time series of the second field :math:`\mathbf{Y}`

Returns :
	out : DataArray
		Time series of the dot product :math:`\mathbf{Z} = \mathbf{X}\times\mathbf{Y}`

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field, electric field and spacecraft position
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
	>>> Exyz = [mms.get_data("E_gse_edp_fast_l2",Tint,i) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,i) for i in ic]
	>>> Jxyz, divB, B, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
	>>> 
	>>> # Compute the electric at the center of mass of the tetrahedron
	>>> Exyzavg = pyrf.avg_4sc(Exyz)
	>>> 
	>>> # Compute J.E dissipation
	>>> JE = pyrf.dot(Jxyz,Exyz)
	

:func:`dynamicp`
================
.. py:function:: dynamicp(N=None, V=None, s="i")

Computes dynamic pressure

Parameters :
	*N* : DataArray
		Time series of the number density of the specie
	*V* : DataArray
		Time series of the bulk velocity of the specie

Options :
	*s* : "i"/"e"
		Specie (default "i")

Returns :
	*Pdyn* : DataArray
		Time series of the dynamic pressure of the specie :math:`P_{dyn,\alpha} = n_\alpha V_\alpha^2`

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Ion bluk velocity
	>>> Vixyz = mms.get_data("Vi_gse_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Remove spintone
	>>> STixyz = mms.get_data("STi_gse_fpi_fast_l2",Tint,ic)
	>>> Vixyz = Vixyz-STixyz
	>>> 
	>>> # Ion number density
	>>> Ni = mms.get_data("Ni_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Compute dynamic pressure
	>>> Pdyn = pyrf.dynamicp(Ni,Vixyz, s="i")


:func:`e_vxb`
==============
.. py:function:: e_vxb(v=None, b=None, flag="vxb")
	
Computes the convection electric field :math:`\mathbf{V}\times\mathbf{B}` (default) or the ExB drift velocity :math:`\frac{\mathbf{E}\times\mathbf{B}}{B^2}` (flag="exb")

Parameters :
	v : DataArray
		Time series of the velocity/electric field

	b : DataArray
		Time series of the magnetic field

	flag : str
		Method flag : 
			"vxb" -> computes convection electric field (default)
			"exb" -> computes ExB drift velocity

Returns :
	out : DataArray
		Time series of the convection electric field/ExB drift velocity

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,1)
	>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,1)
	>>> 
	>>> # Compute ExB drift velocity
	>>> ExB = pyrf.e_vxb(Exyz,Bxyz,"ExB")


:func:`ebsp`
=============
.. py:function:: ebsp(e=None, dB=None, fullB=None, B0=None, xyz=None, freq_int=None, **kwargs)
	
Calculates wavelet spectra of E&B and Poynting flux using wavelets (Morlet wavelet). Also computes polarization 
parameters of B using SVD. SVD is performed on spectral matrices computed from the time series of B using wavelets
and then averaged over a number of wave periods.

Parameters :
	*e* : DataArray
		Time series of the wave electric field

	*dB* : DataArray
		Time series of the wave magnetic field

	*fullB* : DataArray
		Time series of the high resolution background magnetic field used for :math:`\mathbf{E}.\mathbf{B}=0`

	*B0* : DataArray
		Time series of the background magnetic field used for field aligned coordinates

	*xyz* : DataArray
		Time series of the position time series of spacecraft used for field aligned coordinates

	*freq_int* : str/list/ndarray
		Frequency interval : 
			* "pc12" 			-> [0.1, 5.0],
			* "pc35" 			-> [2e-3, 0.1],
			* [fmin, fmax] 	-> arbitrary interval [fmin,fmax]

Options : 
	*polarization* : bool
		Computes polarization parameters (default False)

	*noresamp* : bool
		No resampling, *E* and *dB* are given at the same time line (default False)

	*fac* : bool
		Uses FAC coordinate system (defined by *B0* and optionally *xyz*), otherwise no coordinate system 
		transformation is performed (default False)

	*dEdotB_0* : bool
		Computes *dEz* from :math:`\delta\mathbf{B}.\mathbf{B} = 0`, uses *fullB* (default False)

	*fullB_dB* : bool
		*dB* contains DC field (default False)

	*nAv* : int
		Number of wave periods to average (default 8)

	*facMatrix* : ndarray
		Specify rotation matrix to FAC system (default None)

	*mwidthcoef* : int/float
		Specify coefficient to multiple Morlet wavelet width by. (default 1)

Returns : 
	*res* : Dataset
		Dataset with :
			* *t* : DataArray
				Time

			* *f* : DataArray
				Frequencies

			* *bb_xxyyzzss* : DataArray
				:math:`\delta\mathbf{B}` power spectrum with :
					[...,0] -> x,
					[...,1] -> y,
					[...,2] -> z,
					[...,3] -> sum

			* *ee_xxyyzzss* : DataArray
				:math:`\mathbf{E}` power spectrum with :
					[...,0] -> x,
					[...,1] -> y,
					[...,2] -> z,
					[...,3] -> sum

			* *ee_ss* : DataArray
				:math:`\mathbf{E}` power spectrum (xx+yy spacecraft coordinates, e.g. ISR2)

			* *pf_xyz* : DataArray
				Poynting flux (xyz)

			* *pf_rtp* : DataArray
				Poynting flux (r, theta, phi) [angles in degrees]

			* *dop* : DataArray
				3D degree of polarization

			* *dop2d* : DataArray
				2D degree of polarization in the polarization plane

			* *planarity* : DataArray
				Planarity of polarization

			* *ellipticity* : DataArray
				Ellipticity of polarization ellipse

			* *k* : DataArray
				:math:`k`-vector (theta, phi FAC) [angles in degrees]

See also :
	pl_ebsp, :func:`convert_fac`

Notes :
	This software was developed as part of the MAARBLE (Monitoring, Analyzing and Assessing Radiation Belt 
	Energization and Loss) collaborative research project which has received funding from the European 
	Community's Seventh Framework Program (FP7-SPACE-2011-1) under grant agreement n. 284520.


Examples :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:42.000","2015-10-30T05:15:54.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 3
	>>> 
	>>> # Load spacecraft position
	>>> Tintl = pyrf.extend_tint(Tint,[-100,100])
	>>> Rxyz = mms.get_data("R_gse",Tintl,ic)
	>>> 
	>>> # Load background magnetic field, electric field and magnetic field fluctuations
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
	>>> Bscm = mms.get_data("B_gse_scm_brst_l2",Tint,ic)
	>>> 
	>>> # Polarization analysis
	>>> polarization = pyrf.ebsp(Exyz,Bscm,Bxyz,Bxyz,Rxyz,freq_int=[10,4000],polarization=True,fac=True)

	
		
:func:`edb`
============
.. py:function:: edb(E=None, b0=None, angle_lim=20, flag_method="E.B=0")

Compute Ez under assumption E.B=0 or E.B~=0

Parameters :
	*E* : DataArray
		Time series of the electric field

	*b0* : DataArray
		Time series of the background magnetic field

	*flag_method* : str
		Assumption on the direction of the measured electric field :
			* "E.B=0" -> E.B = 0
			* "Epar" 	-> E field along the B projection is coming from parallel electric field

	*angle_lim* : float
		B angle with respect to the spin plane should be less than angle_lim degrees otherwise Ez is set to 0

Returns :
	*ed* : DataArray
		Time series of the electric field output

	*d* : DataArray
		Time series of the B elevation angle above spin plane

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field, electric field and spacecraft position
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
	>>> Exyz = [mms.get_data("E_gse_edp_fast_l2",Tint,i) for i in ic]
	>>> 
	>>> # Compute Ez
	>>> Ed, d = pyrf.edb(Exyz,Bxyz)

	
:func:`end`
=============
.. py:function:: end(inp=None,fmt="unix")

Gives the last time of the time series

Parameters :
	*inp* : DataArray
		Time series of the input variable

	*fmt* : str
		Format of the output time (see Rots et al. 2015 https://arxiv.org/pdf/1409.7583.pdf)

Returns :
	*out* : float/str
		Value of the last time in the desired format


:func:`extend_tint`
====================
.. py:function:: extend_tint(Tint,ext=[-60,60])

Extends time interval

Parameters :
	*Tint* : list of str
		Reference time interval to extend

	*ext* : list of flot/int
		Number of seconds to extend time interval [left extend, right extend]

Returns :
	*tint* : list of str
		Extended time interval

Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:42.000","2015-10-30T05:15:54.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 3
	>>> 
	>>> # Load spacecraft position
	>>> Tintl = pyrf.extend_tint(Tint,[-100,100])
		
	
:func:`filt`
=============
.. py:function:: filt(inp=None, fmin=0, fmax=1, n=-1)
	
Filters input quantity

Parameters :
	*inp* : DataArray
		Time series of the variable to filter

	*fmin* : float
		Lower limit of the frequency range

	*fmax* : float
		Upper limit of the frequency range

	*n* : int
		Order of the elliptic filter

Returns : 
	*out* : DataArray
		Time series of the filtered signal

Example :
	>>> # Time interval
	>>> Tint = ["2017-07-18T13:03:34.000","2017-07-18T13:07:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic and electric fields
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
	>>> 
	>>> # Convert E to field aligned coordinates
	>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
	>>> 
	>>> # Bandpass filter E waveform
	>>> fmin = 4
	>>> Exyzfachf = pyrf.filt(Exyzfac,fmin,0,3)
	>>> Exyzfaclf = pyrf.filt(Exyzfac,0,fmin,3)

	
:func:`fname`
==============
.. py:function:: fname(Tint=None[, frmt=1])

Creates a string corresponding to time interval for output plot naming

Parameters :
	Tint : list of str
		Time interval

Options :
	*frmt* : int
		Format of the output :
			* 1 : "%Y%m%d_%H%M" (default),
			* 2 : "%y%m%d%H%M%S",
			* 3 : "%Y%m%d_%H%M%S"_"%H%M%S",
			* 4 : "%Y%m%d_%H%M%S"_"%Y%m%d_%H%M%S"

Returns :
	*out* : str
		String corresponding to the time interval in the desired format.

	
:func:`gradient`
=================
.. py:function:: gradient(inp=None)
	
Computes time derivative of the input variable

Parameters :
	inp : DataArray
		Time series of the input variable

Returns :
	out : DataArray
		Time series of the time derivative of the input variable

Example :
	>>> # Time interval
	>>> Tint = ["2017-07-18T13:03:34.000","2017-07-18T13:07:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> 
	>>> # Time derivative of the magnetic field
	>>> dtB = pyrf.gradient(Bxyz)


:func:`histogram`
=================
.. py:function:: histogram(inp=None[, nbins=100, normed=True])

Computes 1D histogram of the *inp* with *nbins* bins

Parameters :
    *inp* : DataArray
        Time series of the input scalar variable
    
Options :
    *nbins* : int
        Number of bins
    
    *normed* : bool
        Normalize the PDF
    
Returns :
    *out* : DataArray
        1D distribution of the input time series
            


:func:`histogram2d`
====================
.. py:function:: histogram2d(inp1=None, inp2=None[, nbins=100])
	
Computes 2d histogram of inp2 vs inp1 with nbins number of bins

Parameters :
	*inp1* : DataArray
		Time series of the x values

	*inp2* : DataArray
		Time series of the y values
	
Options :
	*nbins* : int
		Number of bins (default 100)

Returns :
	*out* : DataArray
		2D map of the density of inp2 vs inp1

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,1) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,1) for i in ic]
	>>> 
	>>> # Compute current density, etc
	>>> J, divB, Bavg, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
	>>> 
	>>> # Compute magnitude of B and J
	>>> Bmag = pyrf.norm(Bavg)
	>>> Jmag = pyrf.norm(J)
	>>> 
	>>> # Histogram of |J| vs |B|
	>>> HBJ = pyrf.histogram2d(Bmag,Jmag)

	
:func:`integrate`
=================
.. py:function:: integrate(inp=None, time_step=None)

Integrate time series

Parameters :
	*inp* : DataArray
		Time series of the variable to integrate

Options :
	*time_step* : float
		Time steps threshold. All time_steps larger than 3*time_step are assumed data gaps, default is that 
		time_step is the smallest value of all time_steps of the time series

Returns :
	*out* : DataArray
		Time series of the time integrated input

Example :
	>>> # Time interval
	>>> Tint = ["2015-12-14T01:17:40.200","2015-12-14T01:17:41.500"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)
	>>> 
	>>> # Convert electric field to field aligned coordinates
	>>> Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
		

:func:`iso2unix`
=================
.. py:function:: iso2unix(t=None)
	
Converts time in iso format to unix

Parameters :
	*t* : list of str
		Time

Returns :
	*out* : list of float
		Time in unix format

	
:func:`mean_bins`
=================
.. py:function:: mean_bins(x=None, y=None[, nbins=10])

Computes mean of values of *y* corresponding to bins of *x*

Parameters :
	*x* : DataArray
		Time series of the quantity of bins

	*y* : DataArray
		Time series of the quantity to the mean

Options :
	*nbins* : int
		Number of bins (default 10)
	
Returns :
	*out* : Dataset
		Dataset with :
			* *bins* : DataArray
				Bin values of the *x* variable

			* *data* : DataArray
				Mean values of *y* corresponding to each bin of *x*

			* *sigma* : DataArray
				Standard deviation

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,1) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,1) for i in ic]
	>>> 
	>>> # Compute current density, etc
	>>> J, divB, Bavg, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
	>>> 
	>>> # Compute magnitude of B and J
	>>> Bmag = pyrf.norm(Bavg)
	>>> Jmag = pyrf.norm(J)
	>>> 
	>>> # Mean value of |J| for 10 bins of |B|
	>>> MeanBJ = pyrf.mean_bins(Bmag,Jmag)
		

:func:`medfilt`
================
.. py:function:: medfilt(inp=None[, npts=11])
	
Applies a median filter over *npts* points to *inp*

Parameters :
	*inp* : DataArray
		Time series of the input variable

Options :
	*npts* : float/int
		Number of points of median filter (default 11)

Returns :
	*out* : DataArry
		Time series of the median filtered input variable

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,1) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,1) for i in ic]
	>>> 
	>>> # Compute current density, etc
	>>> J, divB, Bavg, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
	>>> 
	>>> # Get J sampling frequency
	>>> fs = pyrf.calc_fs(J)
	>>> 
	>>> # Median filter over 1s
	>>> J = pyrf.medfilt(J,fs)


:func:`median_bins`
====================
.. py:function:: median_bins(x=None, y=None[, nbins=10])
	
Computes median of values of *y* corresponding to bins of *x*

Parameters :
	*x* : DataArray
		Time series of the quantity of bins

	*y* : DataArray
		Time series of the quantity to the median

Options :
	*nbins* : int
		Number of bins (default 10)
	
Returns :
	*out* : Dataset
		Dataset with :
			* *bins* : DataArray
				Bin values of the *x* variable

			* *data* : DataArray
				Median values of *y* corresponding to each bin of *x*
				
			* *sigma* : DataArray
				Standard deviation

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft indices
	>>> ic = np.arange(1,5)
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = [mms.get_data("B_gse_fgm_srvy_l2",Tint,1) for i in ic]
	>>> Rxyz = [mms.get_data("R_gse",Tint,1) for i in ic]
	>>> 
	>>> # Compute current density, etc
	>>> J, divB, Bavg, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz,Bxyz)
	>>> 
	>>> # Compute magnitude of B and J
	>>> Bmag = pyrf.norm(Bavg)
	>>> Jmag = pyrf.norm(J)
	>>> 
	>>> # Median value of |J| for 10 bins of |B|
	>>> MedBJ = pyrf.mean_bins(Bmag,Jmag)


:func:`minvar`
===============
.. py:function:: minvar(inp=None[, flag="mvar"])
	
Compute the minimum variance frame

Parameters :
	*inp* : DataArray
		Time series of the quantity to find minimum variance frame

Options :
	*flag* : str
		Constrain (default "mvar")

Returns : 
	*out* : DataArray
		Time series of the input quantity in LMN coordinates

	*l* : array
		Eigenvalues l[0]>l[1]>l[2]

	*V* : array
		Eigenvectors LMN coordinates

See also :
	:func:`new_xyz`

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> 
	>>> # Compute MVA frame
	>>> Blmn, l, V = pyrf.minvar(Bxyz)


:func:`movmean`
================
.. py:function:: movmean(inp=None[, npts=100])
	
Computes running average of the *inp* over *npts* points.

Parameters :
	*inp* : DataArray
		Time series of the input variable

Options :
	*npts* : int
		Number of points to average over (default 100)

Returns :
	*out* : DataArray
		Time series of the input variable averaged over *npts* points

Notes :
	Works also with 3D skymap distribution

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load ion pressure tensor
	>>> Pixyz = mms.get_data("Pi_gse_fpi_brst_l2",Tint,ic)
	>>> 
	>>> # Running average the pressure tensor over 10s
	>>> fs = pyrf.calc_fs(Pixyz)
	>>> Pixyz = pyrf.movmean(Pixyz,10*fs)


:func:`new_xyz`
===============
.. py:function:: new_xyz(inp=None,M=None)

Transform the input field to the new frame

Parameters:
	*inp* : DataArray
		Time series of the input field in the original coordinate system

	*M* : array
		Transformation matrix

Returns :
	*out* : DataArray
		Time series of the input in the new frame

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,ic)
	>>> 
	>>> # Compute MVA frame
	>>> Blmn, l, V = pyrf.minvar(Bxyz)
	>>> 
	>>> # Move electric field to the MVA frame
	>>> Elmn = pyrf.new_xyz(Exyz,V)


:func:`norm`
=============
.. py:function:: norm(inp=None)

Computes the magnitude of the input field

Parameters :
	*inp* : DataArray
		Time series of the input field

Returns :
	*out* : DataArray
		Time series of the magnitude of the input field

Example : 
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> 
	>>> # Compute magnitude of the magnetic field
	>>> Bmag = pyrf.norm(Bxyz)


:func:`normalize`
=================
.. py:function:: normalize(inp=None)

Normalizes the input field

Parameter :
	*inp* : DataArray
		Time series of the input field

Returns :
	*out* : DataArray
		Time series of the normalized input field

Example :
	>>> # Time interval
	>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> 
	>>> # Compute the normalized magnetic field
	>>> b = pyrf.normalize(Bxyz)


:func:`plasma_calc`
====================
.. py:function:: plasma_calc(B=None, Ti=None, Te=None, Ni=None, Ne=None)

Computes plasma parameters including characteristic length and time scales

Parameters :
	*B* : DataArray
		Time series of the magnetic field :math:`[\textrm{nT}]`

	*Ti* : DataArray
		Time series of the ions temperature :math:`[\textrm{eV}]`

	*Te* : DataArray
		Time series of the electrons temperature :math:`[\textrm{eV}]`

	*Ni* : DataArray
		Time series of the ions number density :math:`[\textrm{cm}^{-3}]`

	*Ne* : DataArray
		Time series of the electrons number density :math:`[\textrm{cm}^{-3}]`

Returns :
	*out* : Dataset:
		Dataset of the plasma parameters :
			* *time* : DataArray
				Time

			* *Wpe* : DataArray
				Time series of the electron plasma frequency :math:`[\textrm{rad}.\textrm{s}^{-1}]`

			* *Fpe* : DataArray
				Time series of the electron plasma frequency :math:`[\textrm{Hz}]`

			* *Wce* : DataArray
				Time series of the electron cyclotron frequency :math:`[\textrm{rad}.\textrm{s}^{-1}]`

			* *Fce* : DataArray
				Time series of the electron cyclotron frequency :math:`[\textrm{Hz}]`

			* *Wpp* : DataArray
				Time series of the ion plasma frequency :math:`[\textrm{rad}.\textrm{s}^{-1}]`

			* *Fpp* : DataArray
				Time series of the ion plasma frequency :math:`[\textrm{Hz}]`

			* *Fcp* : DataArray
				Time series of the ion cyclotron frequency :math:`[\textrm{Hz}]`

			* *Fuh* : DataArray
				Time series of the upper hybrid frequency :math:`[\textrm{Hz}]`

			* *Flh* : DataArray
				Time series of the lower hybrid frequency :math:`[\textrm{Hz}]`

			* *Va* : DataArray
				Time series of the Alfvèn velocity (ions) :math:`[\textrm{m}.\textrm{s}^{-1}]`

			* *Vae* : DataArray
				Time series of the Alfvèn velocity (electrons) :math:`[\textrm{m}.\textrm{s}^{-1}]`

			* *Vte* : DataArray
				Time series of the electron thermal velocity :math:`[\textrm{m}.\textrm{s}^{-1}]`

			* *Vtp* : DataArray
				Time series of the electron thermal velocity :math:`[\textrm{m}.\textrm{s}^{-1}]`

			* *Vts* : DataArray
				Time series of the sound speed :math:`[\textrm{m}.\textrm{s}^{-1}]`

			* *gamma_e* : DataArray
				Time series of the electron Lorentz factor

			* *gamma_p* : DataArray
				Time series of the electron Lorentz factor

			* *Le* : DataArray
				Time series of the electron inertial length :math:`[\textrm{m}]`

			* *Li* : DataArray
				Time series of the electron inertial length :math:`[\textrm{m}]`

			* *Ld* : DataArray
				Time series of the Debye length :math:`[\textrm{m}]`

			* *Nd* : DataArray
				Time series of the number of electrons in the Debye sphere

			* *Roe* : DataArray
				Time series of the electron Larmor radius :math:`[\textrm{m}]`

			* *Rop* : DataArray
				Time series of the ion Larmor radius :math:`[\textrm{m}]`

			* *Ros* : DataArray
				Time series of the length associated to the sound speed :math:`[\textrm{m}]`

Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field, ion/electron temperature and number density
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Tixyz = mms.get_data("Ti_gse_fpi_fast_l2",Tint,ic)
	>>> Texyz = mms.get_data("Te_gse_fpi_fast_l2",Tint,ic)
	>>> Ni = mms.get_data("Ni_fpi_fast_l2",Tint,ic)
	>>> Ne = mms.get_data("Ne_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Compute scalar temperature
	>>> Tixyzfac = pyrf.rotate_tensor(Tixyz,"fac",Bxyz,"pp")
	>>> Texyzfac = pyrf.rotate_tensor(Texyz,"fac",Bxyz,"pp")
	>>> Ti = pyrf.trace(Tixyzfac)
	>>> Te = pyrf.trace(Texyzfac)
	>>> 
	>>> # Compute plasma parameters
	>>> pparam = pyrf.plasma_calc(Bxyz,Ti,Te,Ni,Ne)



:func:`pres_anis`
=================
.. py:function:: pres_anis(P=None, B=None)
	
Compute pressure anisotropy factor: :math:`\mu_0\frac{P_\parallel - P_\perp}{B^2}`

Parameters :
	*P* : DataArray
		Time series of the pressure tensor
	*B* : DataArray
		Time series of the background magnetic field

Returns :
	*p_anis* : DataArray
		Time series of the pressure anisotropy

See also :
	:func:`rotate_tensor`

Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field, ion/electron temperature and number density
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Pixyz = mms.get_data("Pi_gse_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Compute pressure anistropy
	>>> p_anis = pyrf.pres_anis(Pxyz,Bxyz)

:func:`resample`
=================
.. py:function:: resample(inp=None,ref=None,**kwargs)

Resample inp to the time line of ref. If sampling of X is more than two times higher than Y, we average X, otherwise
we interpolate X.

Parameters :
	*inp* : DataArray
		Time series to resample

	*ref* : DataArray
		Reference time line

Options :
	*method* : str
		Method of interpolation "spline", "linear" etc. (default "linear") if method is given then interpolate 
		independent of sampling.

	*fs* : float
		Sampling frequency of the Y signal, 1/window

	*window* : int/float/array
		Length of the averaging window, 1/fsample

	*mean* : bool
		Use mean when averaging

	*median* : bool
		Use median instead of mean when averaging

	*max* : bool
		Use max instead of mean when averaging

Returns :
	*out* : DataArray
		Resampled input to the reference time line using the selected method
	

Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and electric field
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Exyz = mms.get_data("E_gse_edp_fast_l2",Tint,ic)
	>>> 
	>>> # Resample magnetic field with respect to electric field
	>>> Bxyz = pyrf.resample(Bxyz,Exyz)



:func:`rotate_tensor`
======================
.. py:function:: rotate_tensor(*args)
	
Rotates pressure or temperature tensor into another coordinate system

Parameters :
	*PeIJ*/*Peall* : DataArray
		Time series of either separated terms of the tensor or the complete tensor. 
		If columns (PeXX,PeXY,PeXZ,PeYY,PeYZ,PeZZ)

	*flag* : str
		Flag of the target coordinates system : 
			Field-aligned coordinates "fac", requires background magnetic field *Bback*, optional 
			flag "pp" P_perp1 = P_perp2 or "qq" P_perp1 and P_perp2 are most unequal, sets P23 to zero.

			Arbitrary coordinate system "rot", requires new x-direction *xnew*, new y and z directions 
			*ynew*, *znew* (if not included y and z directions are orthogonal to *xnew* and closest to the 
			original y and z directions)
 
			GSE coordinates "gse", requires MMS spacecraft number 1--4 MMSnum

Returns : 
	*Pe* : DataArray
		Time series of the pressure or temperature tensor in field-aligned, user-defined, or GSE coordinates.
		For "fac" Pe = [Ppar P12 P13; P12 Pperp1 P23; P13 P23 Pperp2].
		For "rot" and "gse" Pe = [Pxx Pxy Pxz; Pxy Pyy Pyz; Pxz Pyz Pzz]
 
Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and ion temperature tensor
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Tixyz = mms.get_data("Ti_gse_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Compute ion temperature in field aligned coordinates
	>>> Tixyzfac = pyrf.rotate_tensor(Tixyz,"fac",Bxyz,"pp")




:func:`start`
==============
.. py:function:: start(inp=None[, fmt="unix"])
	
Gives the first time of the time series

Parameters :
	inp : DataArray
		Time series

Option :
	fmt : str
		Format of the output (default "unix")

Returns :
	out : float/str
		Value of the first time in the desired format
	

:func:`t_eval`
===============
.. py:function:: t_eval(inp=None,t=None)
	
Evaluates the input time series at the target time

Parameters :
	inp : DataArray
		Time series if the input to evaluate

	t : array
		Times at which the input will be evaluated

Returns :
	out : DataArray
		Time series of the input at times t

Example :
	
	



:func:`trace`
=============
.. py:function:: trace(inp=None)

Computes trace of the time series of 2nd order tensors

Parameters :
	inp : DataArray
		Time series of the input 2nd order tensor.

Returns :
	out : DataArray
		Time series of the trace of the input tensor

Example :
	>>> # Time interval
	>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
	>>> 
	>>> # Spacecraft index
	>>> ic = 1
	>>> 
	>>> # Load magnetic field and ion temperature
	>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
	>>> Tixyz = mms.get_data("Ti_gse_fpi_fast_l2",Tint,ic)
	>>> 
	>>> # Rotate to ion temperature tensor to field aligned coordinates
	>>> Tixyzfac = pyrf.rotate_tensor(Tixyz,"fac",Bxyz,"pp")
	>>> 
	>>> # Compute scalar temperature
	>>> Ti = pyrf.trace(Tixyzfac)


:func:`ts_append`
=================
.. py:function:: ts_append(inp1=None,inp2=None)

Concatenate two time series along the time axis

Parameters :
	*inp1* : DataArray
		Time series of the first input (early times)

	*inp2* : DataArray
		Time series of the second input (late times)

Returns :
	*out* : DataArray
		Concatenated time series

Note : 
	The time series have to be in the correct time order


:func:`ts_scalar`
==================
.. py:function:: ts_scalar(t=None, data=None[, attrs=None])

Create a time series containing a 0th order tensor

Parameters :
	t : np.ndarray
		Array of times

	data : np.ndarray
		Data corresponding to the time list

Options :
	attrs : dict
		Attributes of the data list

Returns :
	out DataArray
		0th order tensor time series

:func:`ts_skymap`
==================
.. py:function:: ts_skymap(time,data,energy,phi,theta,**kwargs)

Creates a skymap of the distribution function

Parameters :
	time : np.ndarray
		List of times

	data : np.ndarray
		Values of the distribution function

	energy : np.ndarray
		Energy levels

	phi : np.ndarray
		Azimuthal angles
		
	theta : np.ndarray
		Elevation angles

Returns :
	out : DataArray


:func:`ts_tensor:xyz`
======================
.. py:function:: ts_tensor_xyz(t=None, data=None[, attrs=None])
	
Create a time series containing a 2nd order tensor

Parameters :
	t : np.ndarray
		Array of times

	data : np.ndarray
		Data corresponding to the time list

Options :
	attrs : dict
		Attributes of the data list

Returns :
	out : DataArray
		2nd order tensor time series


:func:`ts_vec_xyz`
===================
.. py:function:: ts_vec_xyz(t=None, data=None[, attrs=None])

Create a time series containing a 1st order tensor

Parameters :
	*t* : np.ndarray
		Array of times  
	*data* : np.ndarray
		Data corresponding to the time list

Options :
	*attrs* : dict
		Attributes of the data list

Returns :
	*out* : DataArray
		1st order tensor time series

:func:`vht`
============
.. py:function:: vht(e=None,b=None[,flag=1])
	
Estimate velocity of the De Hoffman-Teller frame from the velocity estimate the electric field eht=-vhtxb

Parameters :
	*e* : DataArray
		Time series of the electric field

	*b* : DataArray
		Time series of the magnetic field

	*flag* : int 
		If 2 assumed no Ez.

Returns :
	*vht* : array
		De Hoffman Teller frame velocity [km/s]

	*vht* : DataArray
		Time series of the electric field in the De Hoffman frame             

	*dvht* : array
		Error of De Hoffman Teller frame


:func:`wavelet`
===============
.. py:function:: wavelet(inp=None,**kwargs)
	
Calculate wavelet spectrogram based on fast FFT algorithm

Parameters :
	*inp* : DataArray
		Input quantity

Options :
	*fs* : int/float
		Sampling frequency of the input time series

	*f* : list/np.ndarray
		Vector [fmin fmax], calculate spectra between frequencies fmin and fmax

	*nf* : int/float
		Number of frequency bins

	*wavelet_width* : int/float
		Width of the Morlet wavelet, default 5.36

	*linear* : float
		Linear spacing between frequencies of df

	*returnpower* : bool
		Set to True (default) to return the power, False for complex wavelet transform

	*cutedge* : bool
		Set to True (default) to set points affected by edge effects to NaN, False to keep edge affect points

Returns :
	*out* : DataArray/Dataset
		Wavelet transform of the input



:func:`wavepolarize_means`
==========================
.. py:function:: wavepolarize_means(Bwave=None, Bbgd= None, **kwargs)

Analysis the polarization of magnetic wave using "means" method

Parameters :
	*Bwave* : DataArray
		Time series of the magnetic field from SCM

	*Bbgd* : DataArray
		Time series of the magnetic field from FGM.
	
Options :
	*minPsd* : float
		Threshold for the analysis (e.g 1.0e-7). Below this value, the SVD analysis is meaningless if minPsd is 
		not given, SVD analysis will be done for all waves. (default 1e-25)

	*nopfft* : int
		Number of points in FFT. (default 256)

Returns :
	*Bpsd* : DataArray
		Power spectrum density of magnetic filed wave.

	*degpol* : DataArray
		Spectrogram of the degree of polarization (form 0 to 1).

	*waveangle* : DataArray
		(form 0 to 90)

	*elliptict* : DataArray
		Spectrogram of the ellipticity (form -1 to 1)

	*helict* : DataArray
		Spectrogram of the helicity (form -1 to 1)

Example :
	>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd)
	>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd,1.0e-7)
	>>> [Bpsd,degpol,waveangle,elliptict,helict] = pyrf.wavepolarize_means(Bwave,Bbgd,1.0e-7,256)

Notice: Bwave and Bbgd should be from the same satellite and in the same coordinates 

WARNING: If one component is an order of magnitude or more  greater than the other two then the polarization 
results saturate and erroneously indicate high degrees of polarization at all times and frequencies. Time 
series should be eyeballed before running the program.
For time series containing very rapid changes or spikes the usual problems with Fourier analysis arise.
Care should be taken in evaluating degree of polarization results.
For meaningful results there should be significant wave power at the frequency where the polarization 
approaches 100%. Remember comparing two straight lines yields 100% polarization.

************
:mod:`mms`
************

.. module:: mms
   :platform: Unix, Windows
   :synopsis: Dedicated functions for MMS mission (see Burch et al. 2016 http://link.springer.com/10.1007/s11214-015-0164-9)
.. moduleauthor:: Louis Richard <louis.richard@irfu.se>


.. py:function:: calc_feeps_omni(inp_dset)
	
    Computes the omni-directional FEEPS spectrograms from a Dataset that contains the spectrograms of all eyes.
    
    Parameters:
        *inp_dset* : Dataset
        	Dataset with energy spectrum of every eyes

    Returns:
        *out* : DataArray
        	OMNI energy spectrum from the input

.. py:function:: db_get_ts(dsetName="", cdfName="", trange=None)
	
	Get variable time series in the cdf file

	Parameters :
		*dsetName* : str
			Name of the dataset

		*cdfName* : str
			Name of the target field in cdf file

		*trange* : list of str
			Time interval

	Returns : 
		*out* : DataArray
			Time series of the target variable

.. py:function:: feeps_remove_sun(inp_dset)
	
	Removes the sunlight contamination from FEEPS data

	Parameters :
		*inp_dset* : Dataset
			Dataset of energy spectrum of all eyes (see get_feeps_alleyes)

	Returns :
		*out* : Dataset
			Dataset of cleaned energy spectrum of all eyes 

	Example : 
		>>> Tint = ["2017-07-18T13:04:00.000","2017-07-18T13:07:00.000"]
		>>> iCPS = mms.get_feeps_alleyes("CPSi_brst_l2",Tint,2)
		>>> iCPS_clean = mms.feeps_split_integral_ch(iCPS)
		>>> iCPS_clean_sun_removed = mms.feeps_remove_sun(iCPS_clean)

.. py:function:: feeps_spin_avg(inp_dset_omni)
    
    This function will spin-average the omni-directional FEEPS energy spectra
    
    Parameters:
        *inp_dset_omniv* : DataArray
            Spectrogram of all eyes in OMNI

    Returns:
        *out* : DataArray
            Spin-averaged OMNI energy spectrum


.. py:function:: get_feeps_energy_table(probe, eye, sensor_id)
    
    This function returns the energy table based on each spacecraft and eye.
    Based on the table from: FlatFieldResults_V3.xlsx
       
    from Drew Turner, 1/19/2017
    
    Parameters:
        *probe* : str
            probe #, e.g., "4" for MMS4

        *eye* : str
            sensor eye #

        *sensor_id* : int
            sensor ID
    Returns:
        *Energy_table* : list
        	Energy table
        
    Notes:
        BAD EYES are replaced by NaNs
        - different original energy tables are used depending on if the sensor head is 6-8 (ions) or not (electrons)
        Electron Eyes: 1, 2, 3, 4, 5, 9, 10, 11, 12
        Ion Eyes: 6, 7, 8

.. py:function:: get_feeps_oneeye(tar_var="fluxe_brst_l2", eId="bottom-4", trange=None, mmsId=1)
    
    Load energy spectrum all the target eye

    Parameters :
        *tar_var* : str
            target variable "{data_units}{specie}_{data_rate}_{level}"
                - data_units : 
                    - "flux" : intensity (1/cm sr)
                    - "count" : counts (-)
                    - "CPS" : counts per second (1/s)
                
                - specie : 
                    - "i" : ion
                    - "e" : electron
                
                - data_rate : brst/srvy
                
                - level : l1/l1b/l2/l3??

        *eId* : str
            index of the eye "{deck}-{id}"
                - deck : top/bottom
                - id : see get_feeps_active_eyes

        *trange* : list of str
            Time interval

        *mmsId* : int/str
            Index of the spacecraft


.. py:function:: get_data(varStr,tint,mmsId[, silent=False])

	Load a variable. varStr must in var (see below)

	Parameters :
		*varStr* : str
			Key of the target variable (see below)

		*tint* : list of str
			Time interval

		*mmsId* : str/int
			Index of the target spacecraft

		*silent* : bool
			Set to False (default) to follow the loading

	Returns :
		*out* : DataArray
			Time series of the target variable of measured by the target spacecraft over the selected time interval
	
	Example :
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> gseB = mms.get_data("B_gse_fgm_brst_l2",Tint,1)

	EPHEMERIS :
	"R_gse", "R_gsm"

	FGM : 
	"B_gsm_fgm_srvy_l2", "B_gsm_fgm_brst_l2", "B_gse_fgm_srvy_l2",
	"B_gse_fgm_brst_l2", "B_bcs_fgm_srvy_l2", "B_bcs_fgm_brst_l2",
	"B_dmpa_fgm_srvy_l2", "B_dmpa_fgm_brst_l2"

	DFG & AFG :
	"B_gsm_dfg_srvy_l2pre", "B_gse_dfg_srvy_l2pre", "B_dmpa_dfg_srvy_l2pre",
	"B_bcs_dfg_srvy_l2pre", "B_gsm_afg_srvy_l2pre", "B_gse_afg_srvy_l2pre",
	"B_dmpa_afg_srvy_l2pre", "B_bcs_afg_srvy_l2pre"

	SCM :
	"B_gse_scm_brst_l2"

	EDP :
	"Phase_edp_fast_l2a", "Phase_edp_slow_l2a", "Sdev12_edp_slow_l2a",
	"Sdev34_edp_slow_l2a", "Sdev12_edp_fast_l2a", "Sdev34_edp_fast_l2a",
	"E_dsl_edp_brst_l2", "E_dsl_edp_fast_l2", "E_dsl_edp_brst_ql", 
	"E_dsl_edp_fast_ql", "E_dsl_edp_slow_l2", "E_gse_edp_brst_l2", 
	"E_gse_edp_fast_l2", "E_gse_edp_slow_l2", "E2d_dsl_edp_brst_l2pre", 
	"E2d_dsl_edp_fast_l2pre", "E2d_dsl_edp_brst_ql", "E2d_dsl_edp_fast_ql", 
	"E2d_dsl_edp_l2pre", "E2d_dsl_edp_fast_l2pre", "E2d_dsl_edp_brst_l2pre", 
	"E_dsl_edp_l2pre", "E_dsl_edp_fast_l2pre", "E_dsl_edp_brst_l2pre", 
	"E_dsl_edp_slow_l2pre", "E_ssc_edp_brst_l2a", "E_ssc_edp_fast_l2a", 
	"E_ssc_edp_slow_l2a", "V_edp_fast_sitl", "V_edp_slow_sitl", 
	"V_edp_slow_l2", "V_edp_fast_l2", "V_edp_brst_l2"

	FPI Ions : 
	"Vi_dbcs_fpi_brst_l2", "Vi_dbcs_fpi_fast_l2", "Vi_dbcs_fpi_l2",
	"Vi_gse_fpi_ql", "Vi_gse_fpi_fast_ql", "Vi_dbcs_fpi_fast_ql",
	"Vi_gse_fpi_fast_l2", "Vi_gse_fpi_brst_l2", "partVi_gse_fpi_brst_l2",
	"Ni_fpi_brst_l2", "partNi_fpi_brst_l2", "Ni_fpi_brst",
	"Ni_fpi_fast_l2", "Ni_fpi_ql", "Enfluxi_fpi_fast_ql",
	"Enfluxi_fpi_fast_l2", "Tperpi_fpi_brst_l2", "Tparai_fpi_brst_l2",
	"partTperpi_fpi_brst_l2", "partTparai_fpi_brst_l2", "Ti_dbcs_fpi_brst_l2",
	"Ti_dbcs_fpi_brst", "Ti_dbcs_fpi_fast_l2", "Ti_gse_fpi_ql",
	"Ti_dbcs_fpi_ql", "Ti_gse_fpi_brst_l2", "Pi_dbcs_fpi_brst_l2",
	"Pi_dbcs_fpi_brst", "Pi_dbcs_fpi_fast_l2", "Pi_gse_fpi_ql",
	"Pi_gse_fpi_brst_l2"

	FPI Electrons :
	"Ve_dbcs_fpi_brst_l2", "Ve_dbcs_fpi_brst", "Ve_dbcs_fpi_ql",
	"Ve_dbcs_fpi_fast_l2", "Ve_gse_fpi_ql", "Ve_gse_fpi_fast_l2",
	"Ve_gse_fpi_brst_l2", "partVe_gse_fpi_brst_l2", "Enfluxe_fpi_fast_ql",
	"Enfluxe_fpi_fast_l2", "Ne_fpi_brst_l2", "partNe_fpi_brst_l2",
	"Ne_fpi_brst", "Ne_fpi_fast_l2", "Ne_fpi_ql",
	"Tperpe_fpi_brst_l2", "Tparae_fpi_brst_l2", "partTperpe_fpi_brst_l2",
	"partTparae_fpi_brst_l2", "Te_dbcs_fpi_brst_l2", "Te_dbcs_fpi_brst",
	"Te_dbcs_fpi_fast_l2", "Te_gse_fpi_ql", "Te_dbcs_fpi_ql",
	"Te_gse_fpi_brst_l2", "Pe_dbcs_fpi_brst_l2", "Pe_dbcs_fpi_brst",
	"Pe_dbcs_fpi_fast_l2", "Pe_gse_fpi_ql", "Pe_gse_fpi_brst_l2",

	HPCA : 
	"Nhplus_hpca_srvy_l2", "Nheplus_hpca_srvy_l2", "Nheplusplus_hpca_srvy_l2",
	"Noplus_hpca_srvy_l2", "Tshplus_hpca_srvy_l2", "Tsheplus_hpca_srvy_l2",
	"Tsheplusplus_hpca_srvy_l2", "Tsoplus_hpca_srvy_l2", "Vhplus_dbcs_hpca_srvy_l2",
	"Vheplus_dbcs_hpca_srvy_l2", "Vheplusplus_dbcs_hpca_srvy_l2", "Voplus_dbcs_hpca_srvy_l2",
	"Phplus_dbcs_hpca_srvy_l2", "Pheplus_dbcs_hpca_srvy_l2", "Pheplusplus_dbcs_hpca_srvy_l2",
	"Poplus_dbcs_hpca_srvy_l2", "Thplus_dbcs_hpca_srvy_l2", "Theplus_dbcs_hpca_srvy_l2",
	"Theplusplus_dbcs_hpca_srvy_l2", "Toplus_dbcs_hpca_srvy_l2", "Vhplus_gsm_hpca_srvy_l2",
	"Vheplus_gsm_hpca_srvy_l2", "Vheplusplus_gsm_hpca_srvy_l2", "Voplus_gsm_hpca_srvy_l2",
	"Nhplus_hpca_brst_l2", "Nheplus_hpca_brst_l2", "Nheplusplus_hpca_brst_l2",
	"Noplus_hpca_brst_l2", "Tshplus_hpca_brst_l2", "Tsheplus_hpca_brst_l2",
	"Tsheplusplus_hpca_brst_l2", "Tsoplus_hpca_brst_l2", "Vhplus_dbcs_hpca_brst_l2",
	"Vheplus_dbcs_hpca_brst_l2", "Vheplusplus_dbcs_hpca_brst_l2", "Voplus_dbcs_hpca_brst_l2",
	"Phplus_dbcs_hpca_brst_l2", "Pheplus_dbcs_hpca_brst_l2", "Pheplusplus_dbcs_hpca_brst_l2",
	"Poplus_dbcs_hpca_brst_l2", "Thplus_dbcs_hpca_brst_l2", "Theplus_dbcs_hpca_brst_l2",
	"Theplusplus_dbcs_hpca_brst_l2", "Toplus_dbcs_hpca_brst_l2", "Vhplus_gsm_hpca_brst_l2",
	"Vheplus_gsm_hpca_brst_l2", "Vheplusplus_gsm_hpca_brst_l2", "Voplus_gsm_hpca_brst_l2",
	"Phplus_gsm_hpca_brst_l2", "Pheplus_gsm_hpca_brst_l2", "Pheplusplus_gsm_hpca_brst_l2",
	"Poplus_gsm_hpca_brst_l2", "Thplus_gsm_hpca_brst_l2", "Theplus_gsm_hpca_brst_l2",
	"Theplusplus_gsm_hpca_brst_l2", "Toplus_gsm_hpca_brst_l2"



.. bibliography:: ../references.bib

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
