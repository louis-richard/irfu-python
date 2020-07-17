import numpy as np
import xarray as xr

from .resample import resample
from .ts_tensor_xyz import ts_tensor_xyz




def rotate_tensor(*args):
	"""
	Rotates pressure or temperature tensor into another coordinate system
	
	Parameters :
		PeIJ/Peall : DataArray
			Time series of either separated terms of the tensor or the complete tensor. 
			If columns (PeXX,PeXY,PeXZ,PeYY,PeYZ,PeZZ)
		flag : str
			Flag of the target coordinates system : 
				Field-aligned coordinates "fac", requires background magnetic field Bback, optional 
				flag "pp" P_perp1 = P_perp2 or "qq" P_perp1 and P_perp2 are most unequal, sets P23 to zero.

				Arbitrary coordinate system "rot", requires new x-direction xnew, new y and z directions 
				ynew, znew (if not included y and z directions are orthogonal to xnew and closest to the 
				original y and z directions)
	 
				GSE coordinates "gse", requires MMS spacecraft number 1--4 MMSnum

	Returns : 
		Pe : DataArray
			Time series of the pressure or temperature tensor in field-aligned, user-defined, or GSE coordinates.
			For "fac" Pe = [Ppar P12 P13; P12 Pperp1 P23; P13 P23 Pperp2].
			For "rot" and "gse" Pe = [Pxx Pxy Pxz; Pxy Pyy Pyz; Pxz Pyz Pzz]
	 
	Examples :
		>>> #Rotate tensor into field-aligned coordinates
		>>> Pe = pyrf.rotate_tensor(PeXX,PeXY,PeXZ,PeYY,PeYZ,PeZZ,"fac",Bback)
		>>> Pe = pyrf.rotate_tensor(Peall,"fac",Bback)
		>>> Pe = pyrf.rotate_tensor(Peall,"fac",Bback,"pp")
		>>> 
		>>> #Rotate tensor into user-defined coordinate system
		>>> Pe = pyrf.rotate_tensor(Peall,"rot",xnew)
		>>> Pe = pyrf.rotate_tensor(Peall,"rot",xnew,ynew,znew)
		>>> 
		>>> # Rotate tensor from spacecraft coordinates into GSE coordinates
		>>> Pe = pyrf.rotate_tensor(Peall,"gse",MMSnum)

	"""

	nargin = len(args)

	rtntensor = 0;
	# Check input and load pressure/temperature terms
	if isinstance(args[1],str):
		rotflag = args[1]
		rotflagpos = 1
		Peall = args[0]
		Petimes = Peall.time.data
		if Peall.data.ndim == 3:
		  Ptensor = Peall.data
		  rtntensor = 1
		else :
		  Ptensor = np.zeros((len(Petimes),3,3))
		  Ptensor[:,0,0] = Peall.data[:,0]
		  Ptensor[:,1,0] = Peall.data[:,1]
		  Ptensor[:,2,0] = Peall.data[:,2]
		  Ptensor[:,0,1] = Peall.data[:,3]
		  Ptensor[:,1,1] = Peall.data[:,4]
		  Ptensor[:,2,1] = Peall.data[:,5]
		  Ptensor[:,0,2] = Peall.data[:,6]
		  Ptensor[:,1,2] = Peall.data[:,7]
		  Ptensor[:,2,2] = Peall.data[:,8]
	elif isnstance(args[6],str):
		rotflag = args[6]
		rotflagpos = 6
		Petimes = args[0].time.data
		Ptensor = np.zeros((len(args[0].time.data),3,3))
		Ptensor[:,0,0] = args[0].data
		Ptensor[:,1,0] = args[1].data
		Ptensor[:,2,0] = args[2].data
		Ptensor[:,0,1] = args[1].data
		Ptensor[:,1,1] = args[3].data
		Ptensor[:,2,1] = args[4].data
		Ptensor[:,0,2] = args[2].data
		Ptensor[:,1,2] = args[4].data
		Ptensor[:,2,2] = args[5].data
	else :
		raise SystemError("critical','Something is wrong with the input.")

	
	ppeq = 0
	qqeq = 0
	Rotmat = np.zeros((len(Petimes),3,3))

	if rotflag[0] == "f":
		print("notice Transforming tensor into field-aligned coordinates.")
		if nargin == rotflagpos:
			raise ValueError("B TSeries is missing.")
		
		Bback = args[rotflagpos+1]
		Bback = resample(Bback,Peall)
		if nargin == 4:
			if isinstance(args[3],str) and args[3][0] == "p":
				ppeq = 1
			elif isinstance(args[3],str) and args[3][0] == "q":
				qqeq = 1
			else :
				raise ValueError("Flag not recognized no additional rotations applied.")
		
		if nargin == 9 :
			if isinstance(args[8],str) and args[8][0] == "p":
				ppeq = 1
			elif isinstance(args[8],str) and args[8][0] == "q":
				qqeq = 1
			else :
				raise ValueError("Flag not recognized no additional rotations applied.")
		
		Bvec = Bback/np.linalg.norm(Bback,axis=1,keepdims=True)
		Rx = Bvec.data
		Ry = np.array([1,0,0])
		Rz = np.cross(Rx,Ry);
		Rmag = np.linalg.norm(Rz,axis=1,keepdims=True)
		Rz = Rz/Rmag
		Ry = np.cross(Rz, Rx);
		Rmag = np.linalg.norm(Ry,axis=1,keepdims=True)
		Ry = Ry/Rmag
		Rotmat[:,0,:] = Rx
		Rotmat[:,1,:] = Ry
		Rotmat[:,2,:] = Rz
	elif rotflag[0] == "r":
		print("notice : Transforming tensor into user defined coordinate system.")
		if nargin == rotflagpos:
			raise ValueError("Vector(s) is(are) missing.")
		
		vectors = list(args[rotflagpos+1:])
		if len(vectors) == 1:
			Rx = vectors[0]
			if len(Rx) != 3:
				raise TypeError("Vector format not recognized.")
			
			Rx = Rx/np.linalg.norm(Rx,keepdims=True)
			Ry = np.array([0,1,0])
			Rz = np.cross(Rx,Ry)
			Rz = Rz/np.linalg.norm(Rz,keepdims=True)
			Ry = np.cross(Rz,Rx)
			Ry = Ry/np.linalg.norm(Ry,keepdims=True)
		elif len(vectors) == 3:
			Rx = vectors[0]
			Ry = vectors[1]
			Rz = vectors[2]
			Rx = Rx/np.linalg.norm(Rx,keepdims=True)
			Ry = Ry/np.linalg.norm(Ry,keepdims=True)
			Rz = Rz/np.linalg.norm(Rz,keepdims=True)
			# TO DO: add check that vectors are orthogonal
		else :
			raise TypeError("Vector format not recognized.")
			
		
		Rotmat[:,0,:] = np.ones((len(Petimes),1))*Rx
		Rotmat[:,1,:] = np.ones((len(Petimes),1))*Ry
		Rotmat[:,2,:] = np.ones((len(Petimes),1))*Rz
	elif rotflag[0] == "g":
		"""
		print("notice : Transforming tensor into GSE coordinates.")
		SCnum = args[rotflagpos+1]
		Tint = irf.tint(Petimes.start.utc,Petimes.stop.utc); %#ok<NASGU>
		c_eval('defatt = mms.db_get_variable(''mms?_ancillary_defatt'',''zra'',Tint);',SCnum);
		c_eval('defatt.zdec = mms.db_get_variable(''mms?_ancillary_defatt'',''zdec'',Tint).zdec;',SCnum);
		defatt = mms_removerepeatpnts(defatt); %#ok<NODEF>
		
		% Development of transformation matrix follows modified version of mms_dsl2gse.m
		ttDefatt = EpochTT(defatt.time);
		zra = irf.ts_scalar(ttDefatt,defatt.zra);
		zdec = irf.ts_scalar(ttDefatt,defatt.zdec);
		zra = zra.resample(Petimes);
		zdec = zdec.resample(Petimes);
		[x,y,z] = sph2cart(zra.data*pi/180,zdec.data*pi/180,1);
		saxTmp = irf.geocentric_coordinate_transformation([Petimes.epochUnix x y z],'gei>gse');
		spin_axis = saxTmp(:,2:4);
		Rx = spin_axis(:,1); 
		Ry = spin_axis(:,2); 
		Rz = spin_axis(:,3);
		a = 1./sqrt(Ry.^2+Rz.^2);
		Rotmat(:,1,:) = [a.*(Ry.^2+Rz.^2) -a.*Rx.*Ry -a.*Rx.*Rz];
		Rotmat(:,2,:) = [0*a a.*Rz  -a.*Ry];
		Rotmat(:,3,:) = [Rx Ry Rz];
		"""
	else :
		raise ValueError("Flag is not recognized.")


	Ptensorp = np.zeros((len(Petimes),3,3))
	for ii in range(len(Petimes)):

		rottemp = np.squeeze(Rotmat[ii,:,:])
		Ptensorp[ii,:,:] = np.matmul(np.matmul(rottemp,np.squeeze(Ptensor[ii,:,:])),np.transpose(rottemp))


	if ppeq:
		print("notice : Rotating tensor so perpendicular diagonal components are equal.")
		thetas = 0.5*np.arctan((Ptensorp[:,2,2]-Ptensorp[:,1,1])/(2*Ptensorp[:,1,2]))
		
		for ii, theta in enumerate(thetas):
			if np.isnan(theta):
				theta = 0

			rottemp = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
			Ptensorp[ii,:,:] = np.matmul(np.matmul(rottemp,np.squeeze(Ptensorp[ii,:,:])),np.transpose(rottemp))


	if qqeq:
		print("notice : Rotating tensor so perpendicular diagonal components are most unequal.")
		thetas = 0.5*np.arctan((2*Ptensorp[:,1,2])/(Ptensorp[:,2,2]-Ptensorp[:,1,1]))

		for ii, theta in enumerate(thetas):
			rottemp = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
			Ptensorp[ii,:,:] = np.matmul(np.matmul(rottemp,np.squeeze(Ptensorp[ii,:,:])),np.transpose(rottemp))


	# Construct output
	Pe = ts_tensor_xyz(Petimes,Ptensorp)


	try :
		Pe.attrs["units"] = args[0].attrs["units"]
	except KeyError:
		pass

	return Pe