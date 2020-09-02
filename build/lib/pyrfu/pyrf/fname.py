from dateutil import parser


def fname(tint=None, frmt=1):
	"""
	Creates a string corresponding to time interval for output plot naming

	Parameters :
		- Tint : list of str
			Time interval
		- frmt : int
			Format of the output :
				1 -> "%Y%m%d_%H%M",
				2 -> "%y%m%d%H%M%S",
				3 -> "%Y%m%d_%H%M%S"_"%H%M%S",
				4 -> "%Y%m%d_%H%M%S"_"%Y%m%d_%H%M%S"
	
	Returns :
		- out : str
			String corresponding to the time interval in the desired format.

	"""
	
	if tint is None:
		raise ValueError("fname requires at least one argument")

	if not isinstance(tint, list) or len(tint) != 2:
		raise TypeError("Time interval must be a list")

	if len(tint) != 2:
		raise TypeError("Time interval must have two elements")

	t1 = parser.parse(tint[0])
	t2 = parser.parse(tint[1])

	if frmt == 1:
		out = t1.strftime("%Y%m%d_%H%M")
	elif frmt == 2:
		out = t1.strftime("%y%m%d%H%M%S")
	elif frmt == 3:
		out = "_".join([t1.strftime("%Y%m%d_%H%M%S"), t2.strftime("%H%M%S")])
	elif frmt == 4:
		out = "_".join([t1.strftime("%Y%m%d_%H%M%S"), t2.strftime("%Y%m%d_%H%M%S")])
	else:
		raise ValueError("Unknown format")

	return out
