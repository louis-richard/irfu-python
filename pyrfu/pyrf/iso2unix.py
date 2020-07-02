from astropy.time import Time

def iso2unix(t=None):
	"""
	Converts time in iso format to unix

	Parameters :

	"""
	out = Time(t,format="iso").unix
	return out