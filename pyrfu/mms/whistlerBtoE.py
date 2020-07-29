import numpy as np
from astropy import constants


from ..pyrf.plasma_calc import plasma_calc

def whistlerBtoE(B2=None, freq=None, thetak=None, Bmag=None, ne=None):
	"""
	Computes electric field E power as a function of frequency for whistler waves using B power and cold plasma theory

	Parameters :
		B2 : DataArray
			Time series of the power of whistler magnetic field in nT^2 Hz^{-1}

		freq : array
			frequencies in Hz corresponding B2

		thetak : float
			wave-normal angle of whistler waves in radians

		Bmag : DataArray
			Time series of the magnitude of the magnetic field in nT

		ne : DataArray
			Time series of the electron number density in cm^{-3}

	Returns :
		E2 : DataArray
			Time series of the electric field power

	Example :
		>>> Epower = mms.whistlerBtoE(Bpower,freq,thetak,Bmag,ne)
	"""

	# Calculate plasma parameters
	pparam 	= plasma_calc(Bmag,ne,ne,ne,ne)
	fpe 	= pparam.Fpe
	fce 	= pparam.Fce

	c = constants.c.value

	# Check input
	if len(B2) != len(freq):
	    E2 = None
	    raise IndexError("B2 and freq lengths do not agree!")

	# Calculate cold plasma parameters
	R = 1 - fpe**2/(freq*(freq - fce))
	L = 1 - fpe**2/(freq*(freq + fce))
	P = 1 - fpe**2/freq**2
	D = 0.5*(R - L)
	S = 0.5*(R + L)

	n2 = R*L*np.sin(thetak)**2 
	n2 += P*S*(1 + np.cos(thetak)**2)
	n2 -= np.sqrt((R*L - P*S)**2*np.sin(thetak)**4 + 4*(P**2)*(D**2)*np.cos(thetak)**2)
	n2 /= (2*(S*np.sin(thetak)**2 + P*np.cos(thetak)**2))

	Etemp1 = (P - n2*np.sin(thetak)**2)**2.*((D/(S - n2))**2 + 1) + (n2*np.cos(thetak)*np.sin(thetak))**2
	Etemp2 = (D/(S - n2))**2*(P - n2*np.sin(thetak)**2)**2+P**2*np.cos(thetak)**2;

	E2 = (c**2/n2)*Etemp1/Etemp2*B2*1e-12

	return E2