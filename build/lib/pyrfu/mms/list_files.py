import os
import re
import bisect
from .mms_config import CONFIG
# Time modules
import datetime
from astropy.time import Time
from dateutil import parser
from dateutil.rrule import rrule, DAILY



def list_files(trange=None,probe="1",Var=None):
	"""
	Find files in the data directories of the target instrunent, datatype, data rate, probe and level during the target 
	time interval

	Pramaeters : 
		- data_path         [str]                   Path of data
		- trange            [list]                  Time ineterval
		- probe             [str/int]               Index of the spacecraft
		- instrument        [str]                   Name of the instrument
		- datatype          [str]                   Datatype (dis-moms, dis-dist, etc)
		- data_rate         [str]                   Data rate
		- level             [str]                   Level of data

	Returns :
		- files             [list]                  List of files corresponding to the parameters in the selected
													time interval
	"""
	#print(CONFIG)
	"""
	try :
		from __main__ import data_path
	except ImportError:
		data_path = '/Users/louisr/Documents/PhD/data/mms'
	"""
	data_path = CONFIG["local_data_dir"]
	
	if Var is None:
		raise ValueError("Var is empty")
	
	files_out = []
	
	if type(probe) == int:
		probe = str(probe)
	# directory and file name search patterns
	#   -assume directories are of the form:
	#      (srvy, SITL): spacecraft/instrument/rate/level[/datatype]/year/month/
	#      (brst): spacecraft/instrument/rate/level[/datatype]/year/month/day/
	#   -assume file names are of the form:
	#      spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_version.cdf

	file_name = 'mms'+probe+'_'+Var["inst"]+'_'+Var["tmmode"]+'_'+Var["lev"]\
				+'(_)?.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf'
	
	days = rrule(DAILY, dtstart=parser.parse(parser.parse(trange[0]).strftime('%Y-%m-%d')),\
				 until=parser.parse(trange[1])-datetime.timedelta(seconds=1))

	if Var["dtype"] == "" or Var["dtype"] == None:
		level_and_dtype = Var["lev"]
	else:
		level_and_dtype = os.sep.join([Var["lev"], Var["dtype"]])
#    pdb.set_trace()
	for date in days:
		if Var["tmmode"] == 'brst':
			local_dir = os.sep.join([data_path, 'mms'+probe,Var["inst"],Var["tmmode"], level_and_dtype, \
									 date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')])
		else:
			local_dir = os.sep.join([data_path, 'mms'+probe,Var["inst"],Var["tmmode"], level_and_dtype, \
									 date.strftime('%Y'), date.strftime('%m')])

		if os.name == 'nt':
			full_path = os.sep.join([re.escape(local_dir)+os.sep, file_name])
		else:
			full_path = os.sep.join([re.escape(local_dir), file_name])
		
		
		regex = re.compile(full_path)
		for root, dirs, files in os.walk(local_dir):
			for file in files:
				this_file = os.sep.join([root, file])

				matches = regex.match(this_file)
				if matches:
					this_time = parser.parse(matches.groups()[1])
					if (this_time >= parser.parse(parser.parse(trange[0]).strftime('%Y-%m-%d'))
							and this_time <= parser.parse(trange[1])-datetime.timedelta(seconds=1)):
						if this_file not in files_out:
							files_out.append({'file_name':file,'timetag':'','full_name':this_file,'file_size':''})


	in_files = files_out
	
	file_name = 'mms.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf'

	file_times = []

	regex = re.compile(file_name)
	
	for file in in_files:
		matches = regex.match(file['file_name'])
		if matches:
			file_times.append((file['file_name'], parser.parse(matches.groups()[0]).timestamp(),\
							   file['timetag'], file['file_size']))

	# sort in time
	sorted_files = sorted(file_times, key=lambda x: x[1])

	times = [t[1] for t in sorted_files]

	idx_min = bisect.bisect_left(times, parser.parse(trange[0]).timestamp())

	# note: purposefully liberal here; include one extra file so that we always get the burst mode data
	if idx_min == 0:
		files_in_interval = [{'file_name': f[0],'timetag': f[2],'file_size': f[3]} for f in sorted_files[idx_min:]]
	else:
		files_in_interval = [{'file_name': f[0],'timetag': f[2],'file_size': f[3]} for f in sorted_files[idx_min-1:]]

	local_files = []

	file_names = [f['file_name'] for f in files_in_interval]

	for file in files_out:
		if file['file_name'] in file_names:
			local_files.append(file['full_name'])

	return sorted(local_files)