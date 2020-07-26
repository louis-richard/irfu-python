from dateutil import parser
from astropy.time import Time

def extend_tint(Tint,ext=[-60,60]):
	"""
	Extends time interval

	Parameters :
		Tint : list of str
			Reference time interval to extend

		ext : list of flot/int
			Number of seconds to extend time interval [left extend, right extend]

	Returns :
		tint : list of str
			Extended time interval

	Example :
		>>> # Time interval
		>>> Tint = ["2015-10-30T05:15:42.000","2015-10-30T05:15:54.000"]
		>>> # Spacecraft index
		>>> ic = 3
		>>> # Load spacecraft position
		>>> Tintl = pyrf.extend_tint(Tint,[-100,100])
		
	"""

	# Convert to unix format
	tstart  = Time(parser.parse(Tint[0]),format="datetime").unix
	tstop   = Time(parser.parse(Tint[1]),format="datetime").unix

	# extend interval
	tstart  = tstart+ext[0]
	tstop   = tstop+ext[1]

	# back to iso format
	tstart  = Time(tstart,format="unix").iso
	tstop   = Time(tstop,format="unix").iso

	tint = [tstart,tstop]

	return tint