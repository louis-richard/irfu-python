
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Title     :   pyRF.py
# Subject   :   Library of functions based on MATLAB irf library
# Author    :   Louis RICHARD
# E-MAIL    :   louis.richard@irfu.se
# Created   :   21 - Dec - 19
# Updated   :   02 - June - 20
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Notes :
# Comments should be formated as :
#   
#   """
#   Description
#
#   Paramters :
#       - inp               [type]                  Input
#
#   Returns :
#       - out               [type]                  Output
#
#   """
# 
# 
#-----------------------------------------------------------------------------------------------------------------------

import os
import re
# debugger
import pdb
import bisect
# array modules
import numpy as np
import xarray as xr
# MPI
import multiprocessing as mp
# cdf module
from spacepy import pycdf
# physical constants
from astropy import constants
# signal porcessing, and optimizing
from scipy import interpolate, optimize, signal, fft

import pyfftw

import warnings 
import sfs
import pvlib
from psychopy import tools
import psychopy.tools.coordinatetools as ct
from tqdm import tqdm
import time

# Time modules
import datetime
from astropy.time import Time
from dateutil import parser
from dateutil.rrule import rrule, DAILY

# plot modules
import seaborn as sns
from cycler import cycler
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import axes3d
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib._color_data as mcd



"""
color = [mcd.XKCD_COLORS["xkcd:navy"],\
		 mcd.XKCD_COLORS["xkcd:khaki"],\
		 mcd.XKCD_COLORS["xkcd:crimson"],\
		 mcd.XKCD_COLORS["xkcd:darkgreen"]]

"""
plt.style.use("seaborn-whitegrid")
date_form = mdates.DateFormatter("%H:%M:%S")
sns.set_context("paper")
#plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rc('lines', linewidth=0.5)
color = ["k","b","r","g"]
plt.close("all")

def log(message=""):
	sepline = "#"+"-"*69
	print(sepline)
	print("# {}".format(message))
	print(sepline)
	return

#-----------------------------------------------------------------------------------------------------------------------
def dt642unix(t=None):
	out = Time(t,format="datetime64").unix
	return out
#-----------------------------------------------------------------------------------------------------------------------
def ts_time(t=None,fmt="unix"):
	t = Time(t,format=fmt).datetime64
	out = xr.DataArray(t,coords=[t],dims=["time"])
	return out
#------------------------------------------------------------------------------------------------------------------------
def c_4_v(r1=None, r2=None, r3=None, r4=None, x=None):
	"""
	Calculate velocity or time shift of discontinuity.

	Parameters : 
		- r1...r4           [xarray]                Positon of the spacecraft
		- x                 [list]                  Crossing times or time and velocity
	
	Returns :
		- out               [ndarray]               Discontinuity velocity or time shift with respect to mms1
	"""
	if isinstance(x,np.ndarray) and x.dtype == np.datetime64:
		flag='v_from_t';
		x = Time(x,format="datetime64").unix
	elif x[1] > 299792.458:
		flag = "v_from_t"
	else :
		flag = "dt_from_v"


	def get_vol_ten(r1=None, r2=None, r3=None, r4=None, t=None):
		
		if len(t) == 1:
			t = np.array([t,t,t,t])
			
		tckr1x  = interpolate.interp1d(r1.time.data,r1.data[:,0])
		tckr1y  = interpolate.interp1d(r1.time.data,r1.data[:,1])
		tckr1z  = interpolate.interp1d(r1.time.data,r1.data[:,2])
		r1      = np.array([tckr1x(t[0]),tckr1y(t[0]),tckr1z(t[0])])

		tckr2x  = interpolate.interp1d(r2.time.data,r2.data[:,0])
		tckr2y  = interpolate.interp1d(r2.time.data,r2.data[:,1])
		tckr2z  = interpolate.interp1d(r2.time.data,r2.data[:,2])
		r2      = np.array([tckr2x(t[0]),tckr2y(t[0]),tckr2z(t[0])])

		tckr3x  = interpolate.interp1d(r3.time.data,r3.data[:,0])
		tckr3y  = interpolate.interp1d(r3.time.data,r3.data[:,1])
		tckr3z  = interpolate.interp1d(r3.time.data,r3.data[:,2])
		r3      = np.array([tckr3x(t[0]),tckr3y(t[0]),tckr3z(t[0])])

		tckr4x  = interpolate.interp1d(r4.time.data,r4.data[:,0])
		tckr4y  = interpolate.interp1d(r4.time.data,r4.data[:,1])
		tckr4z  = interpolate.interp1d(r4.time.data,r4.data[:,2])
		r4      = np.array([tckr4x(t[0]),tckr4y(t[0]),tckr4z(t[0])])
		
		# Volumetric tensor with SC1 as center.
		dR = (np.vstack([r2,r3,r4]) - np.tile(r1,(3,1))).T
		
		return dR

	if flag.lower() == "v_from_t":
		# Time input, velocity output
		t   = x
		dR  = get_vol_ten(r1,r2,r3,r4,t)
		dt  = np.array(t[1:])-t[0]
		m   = np.linalg.solve(dR,m)
		
		out = m/np.linalg.norm(m)**2 # "1/v vector"

	elif flag.lower() == "dt_from_v":
		# Time and velocity input, time output
		tc  = x[0] # center time
		v   = np.array(x[1:]) # Input velocity
		m   = v/np.linalg.norm(v)**2

		dR = get_vol_ten(r1,r2,r3,r4,tc)
		
		dt  = np.matmul(dR,m)
		out = np.hstack([0,dt])

	return out
#-----------------------------------------------------------------------------------------------------------------------
def Pointing_flux(E=None, B=None, Bo=None):
	"""
	Estimates Poynting flux S and Poynting flux along Bo from electric field E and magnetic field B.

	If E and B have different sampling then the lowest sampling is resampled at the highest sampling

	Paraneters : 
		- E                 [xarray]                Time serie of the electric field
		- B                 [xarray]                Time serie of the magnetic field
		- Bo                [xarray]                Time serie of the direction to project the Pointing flux (optionnal)

	Returns :
		- S                 [xarray]                Time serie of the Pointing flux
		- Sz                [xarray]                Time serie of the projection of the Pointing flux (only if Bo)
		- intS              [xarray]                Time serie of the time integral of the Pointing flux 
													(if Bo integral along Bo)
	"""

	if E is None:
		raise ValueError("Pointing_flux requires at least two inputs")
	
	if B is None:
		raise ValueError("Pointing_flux requires at least two inputs")
	
	# check which Poynting flux to calculate
	flag_Sz     = False
	flag_intSz  = False
	flag_intS   = False

	if Bo is None:
		flag_intS = True
	else :
		flag_Sz     = True
		flag_intSz  = True

	# resample if necessary
	Fs_E = 1e9/(E.time.data[1]-E.time.data[0]).astype(float)
	Fs_B = 1e9/(B.time.data[1]-B.time.data[0]).astype(float)

	# interval where both E & B exist
	tmin    = Time(max([min(gseE.time.data),min(gseB.time.data)]),format="datetime64").iso
	tmax    = Time(min([max(gseE.time.data),max(gseB.time.data)]),format="datetime64").iso
	Tint    = [tmin,tmax]
	ee      = tlim(E,Tint)
	bb      = tlim(B,Tint)


	if Fs_E < Fs_B:
		e   = resample(ee,bb)
		b   = bb
		Fs  = Fs_B
	elif Fs_E > Fs_B:
		b   = resample(bb,ee)
		e   = ee
		Fs  = Fs_E
	else : 
		prnt("assuming the same sampling. Interpolating B and E to 2x E sampling.")
	"""
	else
	  disp('assuming the same sampling. Interpolating B and E to 2x E sampling.');
	  t=sort([ee(:,1);ee(:,1)+0.5/Fs_E])
	  e=irf_resamp(ee,t)
	  b=irf_resamp(bb,t);Fs=2*Fs_E;
	end
	"""

	# Calculate Poynting flux
	S = np.cross(e,b)/(4*np.pi/1e7)*1e-9
	S = ts_vec_xyz(e.time.data,S)

	if flag_Sz:
	  bm = resample(Bo,e)
	  Sz = dot(norm(bm),S)


	# time integral of Poynting flux along ambient magnetic field
	if flag_intSz:
	 ssz    = Sz
	 idx    = np.isnan(Sz.data)
	 ssz[idx]  = 0 # set to zero points where Sz=NaN
	 intSz  = ssz
	 intSz  = np.cumsum(ssz)/Fs
	 return(S,Sz,intSz)

	if flag_intS:  # time integral of all Poynting flux components
	 ss         = S
	 idx        = np.isnan(S[:,2].data)
	 ss[idx]    = 0 # set to zero points where Sz=NaN
	 intS       = ss
	 intS       = np.cumsum(ss)/Fs
	 Sz         = intS
	 return(S,intS)
#-----------------------------------------------------------------------------------------------------------------------
def waveftt(x=None, window='hamming', frame_overlap=10, frame_length=20, fs=None):
	"""
	Short-Time Fourier Transform
	
	Parameters :
		- x                 [xarray]                One dimension DataArray
		- window            [str]                   Window function such as rectwin, hamming (default)
		- frame_overlap     [float]                 Length of each frame overlaps in second
		- frame_length      [float]                 Length of each frame in second. 
		- fs                [float]                 Sampling frequency

	Return :    
		- S                 [array]                 Spectrogram of x
		- T                 [array]                 Value corresponds to the center of each frame (x-axis) in sec
		- F                 [array]                 Vector of frequencies (y-axis) in Hz
		
	"""
	
	if not isinstance(x,xr.DataArray):
		raise TypeError("x must be a DataArray")

	if fs is None:
		dt = np.median(np.diff(x.time.data).astype(float))*1e-9
		fs = 1/dt

	nperseg     = np.round(frame_length*fs).astype(int)     # convert ms to points
	noverlap    = np.round(frame_overlap*fs).astype(int)    # convert ms to points

	[F,T,S] = signal.spectrogram(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,mode='complex')

	return (F,T,S)
#-----------------------------------------------------------------------------------------------------------------------
def car2sph(inp=None,direction_flag=1):
	"""
	Computes magnitude, theta and phi angle from column vector xyz (first coloumn is x ....) 
	theta is 0 at equator.

	direction_flag = -1  -> to make transformation in opposite direction

	Parameters :
		- inp               [xarray]                Time serie to convert
		- direction_flag    [+-1]                   Set to 1 (default) to transform from cartesian to spherical
													coordinates.
													Set to -1 to transform from spherical to cartesian coordinates.

	Return : 
		- out               [xarray]                Input field in spherical/cartesian coordinate system

	"""


	if inp == None:
		raise ValueError("car2sph requires a least one argument")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if inp.attrs["TENSOR_ORDER"] != 1 or inp.data.ndim != 2:
		raise TypeError("Input must be vector field")  

	xyz = inp.data

	if direction_flag == -1:
		r   = xyz[:,0]
		st  = np.sin(xyz[:,1]*np.pi/180)
		ct  = np.cos(xyz[:,1]*np.pi/180)
		sp  = np.sin(xyz[:,2]*np.pi/180)
		cp  = np.cos(xyz[:,2]*np.pi/180)
		z   = r*st
		x   = r*ct*cp
		y   = r*ct*sp

		outdata = np.hstack([x,y,z])

	else :
		xy  = xyz[:,0]**2 + xyz[:,1]**2
		r   = np.sqrt(xy+xyz[:,2]**2)
		t   = np.arctan2(xyz[:,2],np.sqrt(xy))*180/np.pi
		p   = np.arctan2(xyz[:,1],xyz[:,0])*180/np.pi

		outdata = np.hstack([r,t,p])

	out = ts_vec_xyz(inp.time.data,outdata,inp.attrs)

	return out
#------------------------------------------------------------------------------------------------------------------------
def ed_nrf(e=None, b=None, v=None, flag=0):
	"""
	Find E and B in MP system given B and MP normal vector
	
	Parameters :
		- e                 [xarray]                Time serie of the electric field
		- b                 [xarray]                Time serie of the magnetic field
		- v                 []
	"""

	if e is None:
		raise ValueError("eb_nrf requires at least 3 arguments")
	elif not isinstance(e,xr.DataArray):
		raise TypeError("e must be a DataArray")

	if b is None:
		raise ValueError("eb_nrf requires at least 3 arguments")
	elif not isinstance(b,xr.DataArray):
		raise TypeError("b must be a DataArray")

	if v is None:
		raise ValueError("eb_nrf requires at least 3 arguments")
	elif not isinstance(v,xr.DataArray):
		raise TypeError("v must be a DataArray")

	if isinstance(flag,int):
		if flag == 1 :
			flag_case = "B"
		else :
			flag_case = "A"

	elif isinstance(flag,np.ndarray) and np.size(flag) == 3:
		L_direction = flag
		flag_case = "C"


	if flag_case == "A":
		be = resample(b,e).data

		nl = be/np.linalg.norm(be,axis=0)[:,None] # along the B
		nn = np.cross(np.cross(be,v),be) # closest to given vn vector
		nn = nn/np.linalg.norm(nn)[:,None]
		nm = np.cross(nn,nl) # in (vn x b) direction

		# estimate e in new coordinates
		en  = dot(e,nn)
		el  = dot(e,nl)
		em  = dot(e,nm)
		emp = np.hstack([el,em,en])

	elif flag_case == "B":
		nn = v/np.linalg.norm(v)
		nm = norm(np.cross(nn,np.mean(b)))
		nl = cross(nm,nn)

		# estimate e in new coordinates
		en  = dot(e,nn)
		el  = dot(e,nl)
		em  = dot(e,nm)
		emp = np.hstack([el,em,en])

	elif flag_case == "C":
		nn = norm(v)
		nm = norm(np.cross(nn,L_direction))
		nl = cross(nm,nn)

		# estimate e in new coordinates
		en = dot(e,nn,1);
		el = dot(e,nl,1);
		em = dot(e,nm,1);

		emp = np.hstack([el,em,en])

	out = xr.DataArray(e.time.data,emp,e.attrs)

	return out
#-----------------------------------------------------------------------------------------------------------------------
def lowpass(inp=None,fcut=None,fhz=None):
	"""
	Filter the data through low or highpas filter with max frequency fcut and subtract from the original

	Parameters :
		- inp               [xarray]                Input variable
		- fcut              [float]                 Cutoff frequency
		- fhz               [float]                 Sampling frequency

	Returns :
		- out               [xarray]                Filtered input variable

		
	"""

	if inp is None or fcut is None or fhz is None:
		raise ValueError("lowpass requires at least 3 arguments")

	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	if not isinstance(fcut,float):
		raise TypeError("fcut must be a float")

	if not isinstance(fhz,float):
		raise TypeError("fhz must be a float")

	fnyq = fhz/2

	rp = 0.1
	rs = 60
	norder = 4
	dedata = signal.detrend(data,type='linear')
	rest = data - dedata

	[b,a] = signal.ellip(norder,rp,rs,fcut/fnyq,output='ba')

	outdata = signal.filtfilt(b,a,dedata) + rest


	out = xr.DataArray(outdata,coords=inp.coords,dims=inp.dims)

	return out
#-----------------------------------------------------------------------------------------------------------------------
def solidangle(VecA=None,VecB=None,VecC=None):
	"""
	Calculates the solid angle of three vectors making up a triangle in a unit sphere with the sign taken into account

	Parameters :
		- VecA              [arrray]                First vector
		- VecB              [arrray]                Second vector
		- VecC              [arrray]                Third vector

	Return :
		- angle             [float]                 Solid angle

	"""

	
	if VecA is None or VecB is None or VecC is None:
		raise ValueError("solidangle requires at least 3 arguments")

	# Check VecA is a vector
	if not isinstance(VecA,np.ndarray):
		raise TypeError("VecA must be a np.ndarray")
	elif VecA.ndim != 1:
		raise TypeError("VecA must be a one dimension np.ndarray")
	elif np.size(VecA) != 3:
		raise TypeError("VecA must have only 3 components")

	# Check VecB is a vector
	if not isinstance(VecB,np.ndarray):
		raise TypeError("VecB must be a np.ndarray")
	elif VecB.ndim != 1:
		raise TypeError("VecB must be a one dimension np.ndarray")
	elif np.size(VecB) != 3:
		raise TypeError("VecB must have only 3 components")

	# Check VecC is a vector
	if not isinstance(VecC,np.ndarray):
		raise TypeError("VecC must be a np.ndarray")
	elif VecC.ndim != 1:
		raise TypeError("VecC must be a one dimension np.ndarray")
	elif np.size(VecC) != 3:
		raise TypeError("VecC must have only 3 components")


	# Calculate the smaller angles between the vectors around origin
	a = np.arccos(np.sum(VecC*VecB)); 
	b = np.arccos(np.sum(VecA*VecC)); 
	c = np.arccos(np.sum(VecB*VecA)); 

	#Calculate the angles in the spherical triangle (Law of Cosines)
	A = np.arccos((np.cos(a)-np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)));
	B = np.arccos((np.cos(b)-np.cos(a)*np.cos(c))/(np.sin(a)*np.sin(c)));
	C = np.arccos((np.cos(c)-np.cos(b)*np.cos(a))/(np.sin(b)*np.sin(a)));

	# Calculates the Surface area on the unit sphere (solid angle)
	angle = (A+B+C-np.pi)
	# Calculate the sign of the area
	var         = np.cross(VecC,VecB)
	div         = np.sum(var*VecA)
	signarea    = np.sign(div)

	# Solid angle with sign taken into account
	angle = signarea*angle

	return angle
#-----------------------------------------------------------------------------------------------------------------------
def mean(inp=None, r=None, b=None, z=None):
	"""
	Put inp into mean field coordinates defined by position vector r and magnetic field b 
	if earth magnetic dipole axis z is given then uses another algorithm (good for auroral passages)

	Parameters :
		- inp               [xarray]                Input field to put into MF coordinates
		- r                 [xarray]                Position of the spacecraft
		- b                 [xarray]                Magnetic field
		- z                 [xarray]                Earth magnetic dipole axis

	Returns :
		- out               [xarray]                Input field in mean field coordinates

	"""


	# Check if there are at least 3 arguments
	if inp is None or r is None or b is None:
		raise ValueError("mean requires at least 3 arguments")

	# Check if inp is DataArray
	if not isinstance(inp,xr.DataArray):
		raise TypeError("inp must be a DataArray")

	# Check if r is DataArray
	if not isinstance(r,xr.DataArray):
		raise TypeError("r must be a DataArray")

	# Check if b is DataArray
	if not isinstance(b,xr.DataArray):
		raise TypeError("b must be a DataArray")

	# 
	if not z is None:
		flag_dipole = True
	
		if not isinstance(z,xr.DataArray):
			raise TypeError("z must be a DataArray")
		elif len(z) != len(inp):
			zz = resample(z,inp)
		else :
			zz = z
	else :
		flag_dipole = False


	if len(r) != len(inp):
		rr = resample(r,inp)
	else :
		rr = r

	if len(b) != len(inp):
		bb = resample(b,inp)
	else :
		bb = b

	zv = norm(bb)

	if not flag_dipole:
		yv = cross(zv,rr)
		yv /= np.linalg.norm(yv,axis=1)[:,None]
	else :
		ss      = np.sum(b*r)
		ind     = ss > 0
		ss      = -1*np.ones(ss.shape)
		ss[ind] = 1
		yv      = np.cross(zz,bb)*ss[:,None]
		yv      /= np.linalg.norm(yv,axis=1)[:,None]


	xv = np.cross(yv,zv)

	# in case rotation axis is used as reference uncomment next line
	# rot_axis=rr;rot_axis(:,[2 3])=0;yv=irf_norm(irf_cross(irf_cross(bb,rot_axis),bb));xv=irf_cross(yv,zv);

	outdata         = np.zeros(inp.data.shape)
	outdata[:,0]    = np.sum(xv*inp,axis=1)
	outdata[:,1]    = np.sum(yv*inp,axis=1)
	outdata[:,2]    = np.sum(zv*inp,axis=1)

	out = ts_vec_xyz(inp.time.data,outdata,inp.attrs)
#-----------------------------------------------------------------------------------------------------------------------
def corr_deriv(x1=None, x2=None, fla=False):
	"""
	Correlate the derivatives of two time series

	Parameters :
		- x1, x2            [xarray]                Signals to correlate
		- fla               [bool]                  Flag if False (default) returns time instants of common highest 
													first and second derivatives. If True returns time instants of 
													common highest first derivative and zeros crossings

	Return :
		- t1_d, t2_d        [ndarray]               Time instants of common highest first derivatives
		- t1_dd, t2_dd      [ndarray]               Time instants of common highest second derivatives or zero crossings

	"""


	# 1st derivative
	tx1 = Time(x1.time.data,format="datetime64").unix
	x1 = x1.data
	dtx1 = tx1[:-1]+0.5*np.diff(tx1)
	dx1 = np.diff(x1)

	tx2 = Time(x2.time.data,format="datetime64").unix
	x2 = x2.data
	dtx2 = tx2[:-1]+0.5*np.diff(tx2)
	dx2 = np.diff(x2)

	ind_zeros1 = np.where(np.sign(dx1[:-1]*dx1[1:])<0)[0]
	if ind_zeros1 == 0: ind_zeros1 = ind_zeros1[1:]

	ind_zeros2 = np.where(np.sign(dx2[:-1]*dx12[1:])<0)[0]
	if ind_zeros2 == 0: ind_zeros2 = ind_zeros2[1:]

	ind_zeros1_plus = np.where(dx1[ind_zeros1-1]-dx1[ind_zeros1]>0)[0]
	ind_zeros2_plus = np.where(dx2[ind_zeros2-1]-dx2[ind_zeros2]>0)[0]

	ind_zeros1_minu = np.where(dx1[ind_zeros1-1]-dx1[ind_zeros1]<0)[0]
	ind_zeros2_minu = np.where(dx2[ind_zeros2-1]-dx2[ind_zeros2]<0)[0]

	ind1_plus       = ind_zeros1[ind_zeros1_plus]
	ind1_minu       = ind_zeros1[ind_zeros1_minu]
	t_zeros1_plus   = dtx1[ind1_plus]+(dtx1[ind1_plus+1]-dtx1[ind1_plus])\
							/(1+np.abs(dx1[ind1_plus+1])/np.abs(dx1[ind1_plus]))
	t_zeros1_minu   = dtx1[ind1_minu]+(dtx1[ind1_minu+1]-dtx1[ind1_minu])\
							/(1+np.abs(dx1[ind1_minu+1])/np.abs(dx1[ind1_minu]))

	ind2_plus       = ind_zeros2[ind_zeros2_plus]
	ind2_minu       = ind_zeros2[ind_zeros2_minu]
	t_zeros2_plus   = dtx2[ind2_plus]+(dtx2[ind2_plus+1]-dtx2[ind2_plus])\
							/(1+np.abs(dx2[ind2_plus+1])/np.abs(dx2[ind2_plus]))
	t_zeros2_minu   = dtx2[ind2_minu]+(dtx2[ind2_minu+1]-dtx2[ind2_minu])\
							/(1+np.abs(dx2[ind2_minu+1])/np.abs(dx2[ind2_minu]))

	# Remove repeating points
	t_zeros1_plus = np.delete(t_zeros1_plus,np.where(np.diff(t_zeros1_plus)==0)[0])
	t_zeros2_plus = np.delete(t_zeros2_plus,np.where(np.diff(t_zeros2_plus)==0)[0])

	# Define identical pairs of two time axis
	[t1_d_plus,t2_d_plus] = find_closest(t_zeros1_plus,t_zeros2_plus)
	[t1_d_minu,t2_d_minu] = find_closest(t_zeros1_minu,t_zeros2_minu)


	t1_d = np.vstack([t1_d_plus,t1_d_minu])
	t1_d = t1_d[t1_d[:,0].argsort(),]

	t2_d = np.vstack([t2_d_plus,t2_d_minu])
	t2_d = t2_d[t2_d[:,0].argsort(),]

	if fla:
		# zero crossings 
		ind_zeros1 = np.where(np.sign(x1[:-1]*x1[1:])<0)[0]
		ind_zeros2 = np.where(np.sign(x2[:-1]*x2[1:])<0)[0]

		ind_zeros1 = np.delete(ind_zeros1,np.where(ind_zeros1==1)[0])
		ind_zeros2 = np.delete(ind_zeros2,np.where(ind_zeros2==1)[0])

		ind_zeros1_plus = np.where(x1[ind_zeros1-1]-x1[ind_zeros1]>0)[0]
		ind_zeros2_plus = np.where(x2[ind_zeros2-1]-x2[ind_zeros2]>0)[0]

		ind_zeros1_minu = np.where(x1[ind_zeros1-1]-x1[ind_zeros1]<0)[0]
		ind_zeros2_minu = np.where(x2[ind_zeros2-1]-x2[ind_zeros2]<0)[0]

		ind1_plus       = ind_zeros1[ind_zeros1_plus]
		ind1_minu       = ind_zeros1[ind_zeros1_minu]
		t_zeros1_plus   = tx1[ind1_plus]+(tx1[ind1_plus+1]-tx1[ind1_plus])\
							/(1+np.abs(x1[ind1_plus+1])/np.abs(x1[ind1_plus]))
		t_zeros1_minu   = tx1[ind1_minu]+(tx1[ind1_minu+1]-tx1[ind1_minu])\
							/(1+np.abs(x1[ind1_minu+1])/np.abs(x1[ind1_minu]))

		ind2_plus       = ind_zeros2[ind_zeros2_plus]
		ind2_minu       = ind_zeros2[ind_zeros2_minu]
		t_zeros2_plus   = tx2[ind2_plus]+(tx2[ind2_plus+1]-tx2[ind2_plus])\
							/(1+np.abs(x2[ind2_plus+1])/np.abs(x2[ind2_plus]))
		t_zeros2_minu   = tx2[ind2_minu]+(tx2[ind2_minu+1]-tx2[ind2_minu])\
							/(1+np.abs(x2[ind2_minu+1])/np.abs(x2[ind2_minu]))

	else :
		# 2nd derivative
		ddtx1 = dtx1[:-1]+0.5*np.diff(dtx1)
		ddx1 = np.diff(dx1)

		ddtx2 = dtx2[:-1]+0.5*np.diff(dtx2)
		ddx2 = np.diff(dx2)

		ind_zeros1 = np.where(np.sign(ddx1[:-1]*ddx1[1:])<0)[0]
		ind_zeros2 = np.where(np.sign(ddx2[:-1]*ddx2[1:])<0)[0]

		ind_zeros1 = np.delete(ind_zeros1,np.where(ind_zeros1==1)[0])
		ind_zeros2 = np.delete(ind_zeros2,np.where(ind_zeros2==1)[0])

		ind_zeros1_plus = np.where(ddx1[ind_zeros1-1]-ddx1[ind_zeros1]>0)[0]
		ind_zeros2_plus = np.where(ddx2[ind_zeros2-1]-ddx2[ind_zeros2]>0)[0]

		ind_zeros1_minu = np.where(ddx1[ind_zeros1-1]-ddx1[ind_zeros1]<0)[0]
		ind_zeros2_minu = np.where(ddx2[ind_zeros2-1]-ddx2[ind_zeros2]<0)[0]

		ind1_plus       = ind_zeros1[ind_zeros1_plus]
		ind1_minu       = ind_zeros1[ind_zeros1_minu]
		t_zeros1_plus   = ddtx1[ind1_plus]+(ddtx1[ind1_plus+1]-ddtx1[ind1_plus])\
							/(1+np.abs(ddx1[ind1_plus+1])/np.abs(ddx1[ind1_plus]))
		t_zeros1_minu   = ddtx1[ind1_minu]+(ddtx1[ind1_minu+1]-ddtx1[ind1_minu])\
							/(1+np.abs(ddx1[ind1_minu+1])/np.abs(ddx1[ind1_minu]))

		ind2_plus       = ind_zeros2[ind_zeros2_plus]
		ind2_minu       = ind_zeros2[ind_zeros2_minu]
		t_zeros2_plus   = ddtx2[ind2_plus]+(ddtx2[ind2_plus+1]-ddtx2[ind2_plus])\
							/(1+np.abs(ddx2[ind2_plus+1])/np.abs(ddx2[ind2_plus]))
		t_zeros2_minu   = ddtx2[ind2_minu]+(ddtx2[ind2_minu+1]-ddtx2[ind2_minu])\
							/(1+np.abs(ddx2[ind2_minu+1])/np.abs(ddx2[ind2_minu]))


	# Define identical pairs of two time axis
	[t1_dd_plus,t2_dd_plus] = find_closest(t_zeros1_plus,t_zeros2_plus)
	[t1_dd_minu,t2_dd_minu] = find_closest(t_zeros1_minu,t_zeros2_minu)
		

	t1_dd = np.vstack([t1_dd_plus,t1_dd_minu])
	t1_dd = t1_dd[t1_dd[:,0].argsort(),]

	t2_dd = np.vstack([t2_dd_plus,t2_dd_minu])
	t2_dd = t2_dd[t2_dd[:,0].argsort(),]
	
	return  (t1_d, t2_d, t1_dd, t2_dd)
#-----------------------------------------------------------------------------------------------------------------------
def find_closest(t1=None, t2=None):
	"""
	Finds pairs that are closest to each other in two timeseries

	Parameters :
		- t1        [ndarray]       Vector with time instants
		- t2        [ndarray]       Vector with time instants

	Returns :
		- t1new     [ndarray]       Identified time instants that are closest each other
		- t2new     [ndarray]       Identified time instants that are closest each other

	"""


	t1_orig = t1
	t2_orig = t2
	flag    = True


	while flag:
		flag_t1         = np.zeros(t1.shape)
		tckt1           = interpolate.interp1d(t1,np.arange(len(t1)),kind="nearest",fill_value="extrapolate")
		ind             = tckt1(t2)
		flag_t1[ind]    = 1
		tckt2           = interpolate.interp1d(t2,np.arange(len(t2)),kind="nearest",fill_value="extrapolate")
		ind             = tckt2(t1)
		flag_t2[ind]    = 1

		ind_zeros_t1 = np.where(flag_t1 == 0)[0]
		ind_zeros_t2 = np.where(flag_t2 == 0)[0]
		if ind_zeros_t1:
		  t1 = np.delete(t1,ind_zeros_t1)
		elif ind_zeros_t2:
		  t2 = np.delete(t2,ind_zeros_t2)
		else :
		  flag = False
		  break

	t1new = t1
	t2new = t2

	tckt1_orig  = interpolate.interp1d(t1_orig,np.arange(len(t1_orig)),kind="nearest")
	ind1new     = tckt1_orig(t1new)
	tckt2_orig  = interpolate.interp1d(t2_orig,np.arange(len(t2_orig)),kind="nearest")
	ind2new     = tckt2_orig(t2new)

	return (t1new,t2new)
#-----------------------------------------------------------------------------------------------------------------------
def pol2cart(phi, rho):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
def removerepeatpnts(tsdata):
	"""
	Remove repeated elements in TSERIES or structure data. Important when using DEFATT products. Must have a time variable.

	Parameters :
		- tsdata        [xarray/dict] Input time series

	Return :
		- newdata       [xarray/dict] Input time series
	
	"""
	threshold = 100 # Points separated in time by less than 100ns are treated as repeats

	if insinstance(tsdata,xr.DataArray):
		diffs = np.diff(tsdata.time.data.view("i8")*1e-9)

		norepeat = np.ones(len(tsdata))
		norepeat[diffs < threshold] = 0

		newtstime = tsdata.time.data[norepeat==1]
		newtsdata = tsdata.data[norepeat==1,:]

		if newtsdata.ndim == 1:
			newdata = ts_scalar(newtstime,newtsdata)
		elif newtsdata.ndim == 2:
			newdata = ts_vec_xyz(newtstime,newtsdata)
		elif newtsdata.ndim == 3:
			newdata = ts_vec_xyz(newtstime,newtsdata)
	elif isinstance(tsdata,dict) and ("time" in tsdata):
		if tsdata["time"].dtype == "<M8[ns]":
			diffs = np.diff(tsdata["time"].view("i8")*1e-9)
		else :
			diffs = np.diff(tsdata["time"])
		
		norepeat = np.ones(len(tsdata["time"]))
		norepeat[diffs < threshold] = 0
		
		varnames =  tsdata.keys()
		
		for varname in varnames:
			tsdata[varname] = tsdata[varname][norepeat == 1,:]
		
		newdata = tsdata
	else :
		newdata = tsdata # no change to input if it's not a TSERIES or structure

	return newdata

#-----------------------------------------------------------------------------------------------------------------------
def reduce(dist,dim,x,*args,**kwargs):
	
	# Check to what dimension the distribution is to be reduced   
	if dim == "1D" or dim == "2D":
		dim = int(dim[0]) # input dim can either be '1D' or '2D'
	else :
		raise ValueError("First input must be a string deciding projection type, either ''1D'' or ''2D''.")

	if dim == 1: # 1D: projection to line
		if isinstance(x,xr.DataArray):
			xphat_mat = norm(resample(x,dist.data)).data
		elif isinstance(x,list) and len(x) == 3:
			xphat_mat = np.tile(np.array(x),(len(dist),1))
		elif isinstance(x,np.ndarray) and x.shape == (len(dist),3):
			xphat_mat = x
			
		xphat_mat       = norm(resample(x,dist.data))
		xphat_amplitude = np.linalg.norm(xphat_mat, axis=1,keepdims=True)

		if np.abs(np.mean(xphat_amplitude)-1) < 1e-2 and np.std(xphat_amplitude) > 1e-2: # make sure x are unit vectors, 
			xphat_mat = xphat_mat/np.linalg.norm(xphat_mat,axis=1, keepdims=True)
			print("warning : |<x/|x|>-1| > 1e-2 or std(x/|x|) > 1e-2: x is recalculated as x = x/|x|.\n")

	elif dim == 2:
		if isinstance(x,xr.DataArray) and isinstance(args[0], xr.DataArray):
			y           = args[0]                               # assume other coordinate for perpendicular plane is given after and in same format
			args        = args[1:]
			xphat_mat   = norm(resample(x,dist.data)).data
			yphat_mat   = norm(resample(y,dist.data)).data
		elif isinstance(x,list) and len(x) == 3:
			y           = args[0]                               # assume other coordinate for perpendicular plane is given after and in same format
			args        = args[1:]
			xphat_mat   = np.tile(np.array(x),(len(dist),1))
			yphat_mat   = np.tile(np.array(y),(len(dist),1))
		elif isinstance(x,np.ndarray) and x.shape == (len(dist),3):
			y           = args[0]                               # assume other coordinate for perpendicular plane is given after and in same format
			args        = args[1:]
			xphat_mat   = x
			yphat_mat   = y
		
		else :
			print("Can''t recognize second vector for the projection plane, ''y'': PDist.reduce(''2D'',x,y,...)\n")

		# it's x and z that are used as input to irf_int_sph_dist
		# x and y are given, but might not be orthogonal
		# first make x and y unit vectors
		xphat_amplitude = np.linalg.norm(xphat_mat, axis=1)
		yphat_amplitude = np.linalg.norm(yphat_mat, axis=1)

		# These ifs are not really necessary, but could be there if one 
		# wants to add some output saying that they were not put in 
		# (inputted) as unit vectors. The definition of unit vectors is not
		# quite clear, due to tiny roundoff(?) errors
		if np.abs(np.mean(xphat_amplitude)-1) < 1e-2 and np.std(xphat_amplitude) > 1e-2: # make sure x are unit vectors, 
			xphat_mat = xphat_mat/xphat_amplitude[:,np.newaxis]
			print("warning |<x/|x|>-1| > 1e-2 or std(x/|x|) > 1e-2: x is recalculated as x = x/|x|.\n")

		if np.abs(np.mean(yphat_amplitude)-1) < 1e-2 and np.std(yphat_amplitude) > 1e-2: # make sure y are unit vectors, 
			yphat_mat = yphat_mat/yphat_amplitude[:,np.newaxis]
			print("warning |<y/|y|>-1| > 1e-2 or std(y/|y|) > 1e-2: y is recalculated as y = y/|y|.\n")

		# make z orthogonal to x and y
		zphat_mat       = np.cross(xphat_mat,yphat_mat,axis=1)
		zphat_amplitude = np.linalg.norm(zphat_mat,axis=1)
		zphat_mat       = zphat_mat/zphat_amplitude[:,np.newaxis]
		# make y orthogonal to z and x
		yphat_mat       = np.cross(zphat_mat,xphat_mat,axis=1)
		# check amplitude again, incase x and y were not orthogonal
		yphat_amplitude = np.linalg.norm(yphat_mat, axis=1)

		if np.abs(np.mean(yphat_amplitude)-1) < 1e-2 and np.std(yphat_amplitude) > 1e-2: # make sure y are unit vectors, 
			yphat_mat = yphat_mat/yphat_amplitude[:,np.newaxis]
			print("warning |<y/|y|>-1| > 1e-2 or std(y/|y|) > 1e-2: y is recalculated as y = y/|y|.\n")

		nargs = nargs - 1
		
		# Set default projection grid, can be overriden by given input 'phig'
		nAzg    = 32
		dPhig   = 2*np.pi/nAzg
		phig    = np.linspace(0,2*pi-dPhig,nAzg)+dPhig/2 # centers

	# make input distribution to SI units, s^3/m^6
	#dist = dist.convertto('s^3/m^6');
		   
	# Check for input flags
	# Default options and values
	doTint          = False
	doLowerElim     = False
	nMC             = 100               # number of Monte Carlo iterations
	vint            = [-np.inf,np.inf]
	aint            = [-180,180]        # azimuthal intherval
	vgInput         = 0
	vgInputEdges    = 0
	weight          = "none"      
	correct4scpot   = 0
	base            = "cart"            # coordinate base, cart or pol 

	if dist.attrs["species"] == "electrons":
		isDes = True
	else:
		isDes = False
		  
	ancillary_data = {}

	if "tint" in kwargs:
		tint    = kwargs["tint"]
		doTint  = True

	if "nmc" in kwargs:
		nMC                     = kwargs["nmc"]
		ancillary_data["nMC"]   = nMC

	if "vint" in kwargs:
		vint = kwargs["vint"]

	if "aint" in kwargs:
		aint = kwargs["aint"]

	if "phig" in kwargs:
		phig = kwargs["phig"]

	if "vg" in kwargs:
		vgInput = True
		vg      = kwargs["vg"]*1e3

	if "vg_edges" in kwargs:
		vgInputEdges    = True
		vg_edges        = kwargs["vg_edges"]*1e3

	if "weight" in kwargs:
		weight                      = kwargs["weight"]
		ancillary_data["weight"]    = weight

	if "scpot" in kwargs:
		scpot                       = kwargs["scpot"]
		ancillary_data["scpot"]     = scpot
		correct4scpot               = True

	if "lowerelim" in kwargs:
		lowerelim                   = kwargs["lowerelim"]
		ancillary_data["lowerelim"] = lowerelim
		doLowerElim                 = True
		if isinstance(lowerelim, xr.DataArray):
			lowerelim = resample(inp=lowerelim, ref=dist).data
		elif isinstance(lowerelim,(list,np.ndarray)) and len(lowerelim) == len(dist):
			lowerelim = lowerelim
		elif isinstance(lowerelim, float):
			lowerelim = np.tile(lowerelim,(len(dist),1))
		else :
			print("Can not recognize input for flag lowerelim")

	if "base" in kwargs:
		base = kwargs["base"]
			  
	# set vint ancillary data      
	ancillary_data["vint"] = vint;
	ancillary_data["vint_unit"] = "km/s"

	# Get angles and velocities for spherical instrument grid, set projection
	# grid and perform projection
	
	emat = dist.energy.data
	if doLowerElim:
		lowerelim_mat = np.tile(lowerelim, len(emat[0,:]))
		  
	if correct4scpot:
		scpot       = resample(tlim(scpot,dist),dist)
		scpot_mat   = np.tile(scpot.data, len(emat[0,:]))
		  
	if isDes:
		M = constants.m_e.value
	else:
		M = constants.m_p.value

	if doTint:                                          # get time indicies
		tck = interpolate.interp1(dist.time.data.view("i8")*1e-9,np.arange(len(dist.time)),kind='nearest')

		if len(tint) == 1:                                  # single time
			its = tck(Time(tint,format="isot").unix)
		else :                                              # time interval
			it1 = tck(Time(tint[0],format="isot").unix)
			it2 = tck(Time(tint[1],format="isot").unix)
			its = np.arange(it1,it2)
	else:                                               # use entire PDist
			its = np.arange(len(dist.data))
	  
	nt = len(its)
	if nt == 0:
		raise ValueError("Empty time array. Please verify the time(s) given.")

	# try to make initialization and scPot correction outside time-loop
	if not any([vgInput,vgInputEdges]):                 # prepare a single grid outside the time-loop
		emax            = emat[0,-1]+dist.attrs["delta_energy_plus"][1,-1].data
		vmax            = constants.c.value*np.sqrt(1-(emax*constants.e.value/(M*constants.c.value**2)-1)**2)
		nv              = 100
		vgcart_noinput  = np.linspace(-vmax,vmax,nv)
		print("warning : No velocity grid specified, using a default vg = linspace(-vmax,vmax,{:d}), with vmax = {:3.2f} km/s.".format(nv,vmax*1e-3))

	# loop to get projection
	
	all_vg = np.zeros((nt,100))
	all_vg_edges = np.zeros((nt,101))
	for i, it in enumerate(tqdm(its)):  # display progress
		if dim == 1:
			xphat = xphat_mat.data[i,:]
		elif dim == 2:
			xphat = xphat_mat.data[i,:] # corresponding to phi = 0 in 'phig'
			zphat = zphat_mat.data[i,:] # normal to the projection plane

		# 3d data matrix for time index it
		F3d     = np.squeeze(dist.data[it,...]) # s^3/m^6
		energy  = emat[it,:]

		if doLowerElim:
			remove_extra_ind    = 0 # for margin, remove extra energy channels
			ie_below_elim       = np.nonzero(np.abs(emat[it,:]-lowerelim_mat[it,:]) == np.min(np.abs(emat[it,:]-lowerelim_mat[it,:]))) # closest energy channel

			F3d[:(np.max(ie_below_elim) + remove_extra_ind),...] = 0
			
		if correct4scpot:
			if "delta_energy_minus" in dist.attrs: # remove all that satisfies E-Eminus<Vsc
				ie_below_scpot = np.nonzero(emat[it,:]-dist.attrs["delta_energy_minus"][it,:]-scpot_mat[it,0]<0,)[-1]
			else:
				ie_below_scpot = np.nonzero(np.abs(emat[it,:]-scpot_mat[it,:]) == np.min(np.abs(emat[it,:]-scpot_mat[it,:]))) # closest energy channel

			remove_extra_ind = 0 # for margin, remove extra energy channels
			
			F3d[1:(np.max(ie_below_scpot) + remove_extra_ind),...] = 0

			energy              = energy-scpot_mat[it,:]
			energy[energy<0]    = 0
		
		v = constants.c.value*np.sqrt(1-(energy*constants.e.value/(M*constants.c.value**2)-1)**2) # m/s  

		# azimuthal angle
		if dist.phi.ndim != 1:
			phi = dist.phi.data[it,:]   # in degrees
		else :# fast mode
			phi = dist.phi.data         # in degrees

		phi = phi-180
		phi = phi*np.pi/180             # in radians

		# elevation angle
		th = dist.theta.data            # polar angle in degrees
		th = th-90                      # elevation angle in degrees
		th = th*np.pi/180               # in radians

		# Set projection grid after the first distribution function
		# bin centers
		if vgInputEdges : # redefine vg (which is vg_center)
			vg = vg_edges[1:-1] + 0.5*np.diff(vg_edges)
		elif vgInput:
			vg = vg
		else : # define from instrument velocity bins
			if base == "cart":
				vg = vgcart_noinput # maybe just bypass this and go directly through input vg_edges?
			else :
				if dim == 1:
					vg = np.hstack((-np.flip(V),v))
				elif dim == 2:
					vg = v

		# initiate projected f
		if i == 0:
			if dim == 1:
				Fg  = np.zeros((nt,len(vg)))
				vel = np.zeros((nt,1))
			elif dim == 2 and base == "pol":
				Fg  = np.zeros((nt,len(phig),len(vg)))
				vel = np.zeros((nt,2))
			elif dim == 2 and base == "cart":
				Fg  = np.zeros((nt,len(vg),len(vg)))
				vel = np.zeros((nt,2))
			
			dens = np.zeros((nt,1))
			
		
		# perform projection
		if dim == 1 : # 1D plane
			# v, phi, th corresponds to the bins of F3d
			if vgInputEdges:
				tmpst = int_sph_dist(F3d,v,phi,th,vg,x=xphat,nmc=nMC,vzint=vint*1e3,aint=aint,weight=weight,vg_edges=vg_edges)
			else:

				tmpst = int_sph_dist(F3d,v,phi,th,vg,x=xphat,nmc=nMC,vzint=np.array(vint)*1e3,aint=aint,weight=weight)
				pdb.set_trace()
			all_vg[i,:]         = tmpst["v"] # normally vg, but if vg_edges is used, vg is overriden
			all_vg_edges[0,:]   = tmpst["v_edges"]
		elif dim == 2:
			# is 'vg_edges' implemented for 2d?
			tmpst = int_sph_dist(F3d,v,phi,th,vg,x=xphat,z=zphat,phig=phig,nmc=nMC,vzint=vint*1e3,weight=weight,base=base)

			all_vx[i,...]       = tmpst["vx"]
			all_vy[i,...]       = tmpst["vy"]
			all_vx_edges[i,...] = tmpst["vx_edges"]
			all_vy_edges[i,...] = tmpst["vy_edges"]
		
		# fix for special cases
		# dimension of projection, 1D if projection onto line, 2D if projection onto plane
		if dim == 1 or base == "cart":
			Fg[i,...] = tmpst["F"]
		elif dim == 2 :
			Fg[i,...] = tmpst["F_using_edges"]

		# set moments from reduced distribution (for debug)
		dens[i]     = tmpst["dens"]
		vel[i,:]    = tmpst["vel"]


	# Construct PDist objects with reduced distribution
	# vg is m/s, transform to km/s
	if dim == 1 :
		# Make output
		PD = xr.DataArray(data=Fg, coords=[dist.time[its],all_vg*1e-3], dims=["time","vg"])
		# attributes
		PD.attrs["vg_edges"] = all_vg_edges
	elif dim == 2 and base == "pol" :
		Fg_tmp              = Fg
		all_vx_tmp          = np.transose(all_vx[...,1:end-1],[0,1,2])*1e-3
		all_vy_tmp          = np.transose(all_vy[...,1:end-1],[0,1,2])*1e-3
		all_vx_edges_tmp    = np.transose(all_vx_edges,[0,1,2])*1e-3
		all_vy_edges_tmp    = np.transose(all_vy_edges,[0,1,2])*1e-3
		# Make output
		PD = xr.DataArray(data=Fg_tmp, coords=[dist.time[its],all_vx_tmp,all_vy_tmp], dims=["time","vx","vy"])
		# attributes
		PD.attrs["vx_edges"]    = all_vx_edges_tmp
		PD.attrs["vy_edges"]    = all_vy_edges_tmp
		PD.attrs["base"]        = "pol"
	elif dim == 2 and base == "cart" :
		# Make output
		PD = xr.DataArray(data=Fg_tmp, coords=[dist.time[its],all_vx*1e-3,all_vx*1e-3], dims=["time","vx","vx"])
		# attributes
		PD.attrs["vx_edges"]    = all_vx_edges*1e-3
		PD.attrs["vy_edges"]    = all_vx_edges*1e-3
		PD.attrs["base"]        = "cart"

	PD.attrs["species"] = dist.attrs["species"]
	#PD.userData = dist.userData;
	PD.attrs["v_units"] = "km/s"

	# set units and projection directions
	if dim == 1 :
		PD.attrs["units"]                   = 's/m^4';
		PD.attrs["projection_direction"]    = xphat_mat[its,:]
	elif dim == 2 :
		PD.attrs["units"]                   = 's^2/m^5';
		PD.attrs["projection_dir_1"]        = xphat_mat[its,:]
		PD.attrs["projection_dir_2"]        = yphat_mat[its,:]
		PD.attrs["projection_axis"]         = zphat_mat[its,:]

	if doLowerElim:
		PD.attrs["lowerelim"] = lowerelim_mat
#-----------------------------------------------------------------------------------------------------------------------
