from astropy.time import Time

def iso2unix(t=None):
	"""
	Converts time in iso format to unix

	Parameters :
		t : list of str
			Time

	Returns :
		out : list of float
			Time in unix format

	"""
	
	if t is None:
		raise ValueError("iso2unix requires at least one argument")

	out = Time(t,format="iso").unix
	
	return out