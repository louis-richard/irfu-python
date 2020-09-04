#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extend_tint.py

@author : Louis RICHARD
"""

from dateutil import parser
from astropy.time import Time


def extend_tint(tint=None, ext=None):
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
		>>> from pyrfu import pyrf
		>>> # Time interval
		>>> tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]
		>>> # Spacecraft index
		>>> ic = 3
		>>> # Load spacecraft position
		>>> tintl = pyrf.extend_tint(tint,[-100,100])
		
	"""

	if ext is None:
		ext = [-60, 60]

	# Convert to unix format
	tstart, tstop = [Time(parser.parse(tint_bound), format="datetime").unix for tint_bound in tint]

	# extend interval
	tstart, tstop = [tstart + ext[0], tstop + ext[1]]

	# back to iso format
	tstart, tstop = [Time(bound, format="unix").iso for bound in [tstart, tstop]]

	tint = [tstart, tstop]

	return tint
