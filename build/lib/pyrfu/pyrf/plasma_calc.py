import numpy as np
import xarray as xr
from astropy import constants

from .resample import resample



def plasma_calc(B=None,Ti=None,Te=None,Ni=None,Ne=None):
	"""
	Computes plasma parameters including characteristic length and time scales

	Parameters :
		B : DataArray
			Time series of the magnetic field [nT]

		Ti : DataArray
			Time series of the ions temperature [eV]

		Te : DataArray
			Time series of the electrons temperature [eV]

		Ni : DataArray
			Time series of the ions number density [cm^{-3}]

		Ne : DataArray
			Time series of the electrons number density [cm^{-3}]

	Returns :
		out : Dataset:
			Dataset of the plasma parameters :
				* time : DataArray
					Time

				* Wpe : DataArray
					Time series of the electron plasma frequency [rad.s^{-1}]

				* Fpe : DataArray
					Time series of the electron plasma frequency [Hz]

				* Wce : DataArray
					Time series of the electron cyclotron frequency [rad.s^{-1}]

				* Fce : DataArray
					Time series of the electron cyclotron frequency [Hz]

				* Wpp : DataArray
					Time series of the ion plasma frequency [rad.s^{-1}]

				* Fpp : DataArray
					Time series of the ion plasma frequency [Hz]

				* Fcp : DataArray
					Time series of the ion cyclotron frequency [Hz]

				* Fuh : DataArray
					Time series of the upper hybrid frequency [Hz]

				* Flh : DataArray
					Time series of the lower hybrid frequency [Hz]

				* Va : DataArray
					Time series of the Alfvèn velocity (ions) [m.s^{-1}]

				* Vae : DataArray
					Time series of the Alfvèn velocity (electrons) [m.s^{-1}]

				* Vte : DataArray
					Time series of the electron thermal velocity [m.s^{-1}]

				* Vtp : DataArray
					Time series of the electron thermal velocity [m.s^{-1}]

				* Vts : DataArray
					Time series of the sound speed [m.s^{-1}]

				* gamma_e : DataArray
					Time series of the electron Lorentz factor

				* gamma_p : DataArray
					Time series of the electron Lorentz factor

				* Le : DataArray
					Time series of the electron inertial length [m]

				* Li : DataArray
					Time series of the electron inertial length [m]

				* Ld : DataArray
					Time series of the Debye length [m]

				* Nd : DataArray
					Time series of the number of electrons in the Debye sphere

				* Roe : DataArray
					Time series of the electron Larmor radius [m]

				* Rop : DataArray
					Time series of the ion Larmor radius [m]

				* Ros : DataArray
					Time series of the length associated to the sound speed [m]

	Example :
		>>> # Time interval
		>>> Tint = ["2015-10-30T05:15:20.000","2015-10-30T05:16:20.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field, ion/electron temperature and number density
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
		>>> Tixyz = mms.get_data("Ti_gse_fpi_fast_l2",Tint,ic)
		>>> Texyz = mms.get_data("Te_gse_fpi_fast_l2",Tint,ic)
		>>> Ni = mms.get_data("Ni_fpi_fast_l2",Tint,ic)
		>>> Ne = mms.get_data("Ne_fpi_fast_l2",Tint,ic)
		>>> # Compute scalar temperature
		>>> Tixyzfac = pyrf.rotate_tensor(Tixyz,"fac",Bxyz,"pp")
		>>> Texyzfac = pyrf.rotate_tensor(Texyz,"fac",Bxyz,"pp")
		>>> Ti = pyrf.trace(Tixyzfac)
		>>> Te = pyrf.trace(Texyzfac)
		>>> # Compute plasma parameters
		>>> pparam = pyrf.plasma_calc(Bxyz,Ti,Te,Ni,Ne)

	"""


	if B is None or Ti is None or Te is None or Ni is None or Ne is None:
		raise ValueError("plasma_calc requires at least 5 arguments")
	
	if not isinstance(B,xr.DataArray):
		raise TypeError("Inpouts must be DataArrays")

	if not isinstance(Ti,xr.DataArray):
		raise TypeError("Inpouts must be DataArrays")

	if not isinstance(Te,xr.DataArray):
		raise TypeError("Inpouts must be DataArrays")

	if not isinstance(Ni,xr.DataArray):
		raise TypeError("Inpouts must be DataArrays")

	if not isinstance(Ne,xr.DataArray):
		raise TypeError("Inpouts must be DataArrays")

	# Get constants
	e       = constants.e.value
	m_p     = constants.m_p.value
	m_e     = constants.m_e.value
	mu0     = constants.mu0.value
	c       = constants.c.value
	epso    = constants.eps0.value
	mp_me   = m_p/m_e

	# Resample all variables with respect to the magnetic field
	nt = len(B)
	if len(Ti) != nt: Ti = resample(Ti,B).data
	if len(Te) != nt: Te = resample(Te,B).data
	if len(Ni) != nt: Ni = resample(Ni,B).data
	if len(Ne) != nt: Ne = resample(Ne,B).data

	# Transform number density and magnetic field to SI units
	Ne = 1e6*Ne
	Ni = 1e6*Ni
	if B.ndim == 2:
		B_SI = 1e-9*np.linalg.norm(B,axis=1)
	else :
		B_SI = 1e-9*np.linalg.norm(B,axis=1)

	
	Wpe = np.sqrt(Ne*e**2/(m_e*epso)) # rad/s
	Wce = e*B_SI/m_e;   # rad/s
	Wpp = np.sqrt(Ni*e**2/(m_p*epso))

	Va  = B_SI/np.sqrt(mu0*Ni*m_p)
	Vae = B_SI/np.sqrt(mu0*Ne*m_e);
	Vte = c*np.sqrt(1-1/(Te*e/(m_e*c**2)+1)**2);           # m/s (relativ. correct), particle with Vte has energy e*Te
	Vtp = c*np.sqrt(1-1/(Ti*e/(m_p*c**2)+1)**2);           # m/s
	Vts = np.sqrt((Te*e+3*Ti*e)/m_p);                      # Sound speed formula (F. Chen, Springer 1984). Relativistic?

	gamma_e = 1/np.sqrt(1-(Vte/c)**2);
	gamma_p = 1/np.sqrt(1-(Vtp/c)**2);

	Le = c/Wpe
	Li = c/Wpp
	Ld = Vte/(Wpe*np.sqrt(2)) # Debye length scale, sqrt(2) needed because of Vte definition
	Nd = Ld*epso*m_e*Vte**2/e**2;                           # number of e- in Debye sphere

	Fpe = Wpe/(2*np.pi) # Hz
	Fce = Wce/(2*np.pi)
	Fuh = np.sqrt(Fce**2+Fpe**2);
	Fpp = Wpp/(2*np.pi)
	Fcp = Fce/mp_me;
	Flh = np.sqrt(Fcp*Fce/(1+Fce**2/Fpe**2)+Fcp**2)

	Roe = m_e*c/(e*B_SI)*np.sqrt(gamma_e**2-1); # m, relativistically correct
	Rop = m_p*c/(e*B_SI)*np.sqrt(gamma_p**2-1); # m, relativistically correct
	Ros = Vts/(Fcp*2*np.pi) # m

	out = xr.Dataset({"time"		: B.time.data,			\
						"Wpe" 		: (["time"], Wpe), 		\
						"Wce" 		: (["time"], Wce), 		\
						"Wpp" 		: (["time"], Wpp), 		\
						"Va" 		: (["time"], Va), 		\
						"Vae" 		: (["time"], Vae), 		\
						"Vte" 		: (["time"], Vte), 		\
						"Vtp" 		: (["time"], Vtp),		\
						"Vts" 		: (["time"], Vts), 		\
						"gamma_e" 	: (["time"], gamma_e), 	\
						"gamma_p" 	: (["time"], gamma_p), 	\
						"Le" 		: (["time"], Le), 		\
						"Li" 		: (["time"], Li), 		\
						"Ld" 		: (["time"], Ld), 		\
						"Nd" 		: (["time"], Nd),		\
						"Fpe" 		: (["time"], Fpe),	 	\
						"Fce" 		: (["time"], Fce), 		\
						"Fuh" 		: (["time"], Fuh), 		\
						"Fpp" 		: (["time"], Fpp), 		\
						"Fcp" 		: (["time"], Fcp), 		\
						"Flh" 		: (["time"], Flh), 		\
						"Roe" 		: (["time"], Roe), 		\
						"Rop" 		: (["time"], Rop), 		\
						"Ros" 		: (["time"], Ros)})
	
	return out