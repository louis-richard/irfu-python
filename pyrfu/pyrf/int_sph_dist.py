#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
int_sph_dist.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
import psychopy.tools.coordinatetools as ct



def int_sph_dist(F,v,phi,th,vg,**kwargs):
	"""
	Calculates the integrated distribution function of a 3D distribution function in a spherical grid. F is a 3D 
	matrix containing values of the flux (index order of F is [velocity,phi,th]), phi is the azimuthal and th the 
	elevation angles in radians, and vg is the .

	Parameters :
		F : array
			3D skymap of the distribution function
		
		v : array
			Velocity

		ph : array
			Azimuthal angle

		th : array
			Elevation angle

		vg : array
			Bin centers of the velocity of the projection grid

	Options:
		phig : array
			Bin centers of the azimuthal angle of the projection in radians in the span [0,2*pi]. If this input is 
			given, the projection will be 2D. If it is omitted, the projection will be 1D.c
		x : list/array
			Axis that is not integrated along in 1D and x-axis (phig = 0) in 2D. x = [1,0,0] if omitted.

		z : list/array
			Axis that is integrated along in 2D. z = [0,0,1] if omitted.

		nMC : int
			Average number of Monte Carlo iterations used per bin for integration, default is 10. Number of iterations 
			can be weighted data value in each bin.

		weight : str
			How the number of MC iterations per bin is weighted, can be None (default), "lin" or "log"

		vzint : list of float
			Set limits on the out-of-plane velocity interval in 2D and "transverse" velocity in 1D.

		aint : list of float
			Angular limit in degrees, can be combined with vzlim
		
		base : str
			coordinate base, 'pol' for polar or 'cart' for Cartesian, does not matter in 1D case
		
		ve : list of float
			Velocity edges of instrument (from delta_energy) with same units as v and one element longer than v

	Returns :
		out : DataArray
			Time series of the projected distribution function

	Notes :
		The function goes through all instrument bins and finds the best match on the projection bin. The value in 
		the projection bin, F, is updated as F = F+f*dTau/dA, where f is the instrument value in the bin, dTau is the 
		volume element in velocity space of the instrument bin and dA is the area or line element in the projection 
		bin. An instrument bin can actually cover several projection bin and the value should be added to all those 
		bins, scaled to the area the instrument bin covers of a given projection bin. This area is calculated with a 
		Monte Carlo integration scheme, where many "particles" are generated somewhere inside the instrument bin, each 
		assigned a fraction of f. The "particles" are then projected onto the line or plane.

	See also: 
		mms.plot_int_projection

	TODO : 
		Optimization

	"""
	
	# Check for flags in input
	# Set default values
	xphat           = np.array([1,0,0])             # axes projection is done against in 1D, x-axis in 2D
	zphat           = np.array([0,0,1])             # integrate along this axes in 2D, has no use in 1D
	nMC             = 10                            # number of Monte Carlo iterations
	vzint           = np.array([-np.inf,np.inf])    # limit on out-of-plane velocity
	aint            = np.array([-180,180])          # limit on out-of-plane velocity
	projDim         = 1                             # number of dimensions of the projection
	weight          = "none"                        # how number of MC points is weighted to data
	base            = "pol"                         # If 1D then this does not matter
	veInput         = False                         # input energy differences
	veInputEdges    = False                         # 

	# read options
	if "x" in kwargs :
		xphat = kwargs["x"]

	if "z" in kwargs :
		zphat = kwargs["z"]
		# If this flag is given, it assumes 2D projection
		projDim = 2

	if "phig" in kwargs :
		phig = kwargs["phig"]

	if "nmc" in kwargs :
		nMC = kwargs["nmc"]

	if "vzint" in kwargs :
		vzint = kwargs["vzint"]

	if "aint" in kwargs :
		aint = kwargs["aint"]

	if "weight" in kwargs :
		weight = kwargs["weight"]

	if "base" in kwargs :
		base = kwargs["base"]

	if "ve" in kwargs :
		ve 		= kwargs["ve"]
		veInput = True

	if "vg_edges" in kwargs :
		vg_edges 		= kwargs["vg_edges"]
		veInputEdges 	= True


	# Initiate initiate various things

	# complete RH system
	xphat = xphat/np.linalg.norm(xphat, keepdims=True)
	yphat = np.cross(zphat,xphat) # zphat define as default [0 0 1] or read in as optional input above
	yphat = yphat/np.linalg.norm(yphat, keepdims=True)
	zphat = np.cross(xphat,yphat) # z = cross(x,cross(z,x)) % enforce z to be orthogonal to x
	zphat = zphat/np.linalg.norm(zphat, keepdims=True)

	# diffs of instrument bins
	# velocity
	if veInput :
		dVm = v-ve[:-1]
		dVp = ve[1:]-v  # minus and plus velocity from center
		dV  = dVm+dVp   # total difference
	else :
		dV  = np.diff(v)
		dV  = np.hstack([dV[0],dV])     # quick and dirty
		dVm = np.diff(v)/2              # minus velocity from center
		#dVp = np.diff(v)/2              # plus velocity from center

	dPhi    = np.abs(np.median(np.diff(phi)))   # constant
	dTh     = np.abs(np.median(np.diff(th)))    # constant

	# primed (grid) diffs
	# dVg = diff(vg); dVg = [dVg(1),dVg]; % quick and dirty
	if projDim == 2 :
		dPhig = np.median(np.diff(phig))    # constant
	else :
		dPhig = 1                           # unity for 1D

	# Number of projection bins
	nVg = len(vg);
	if projDim == 2 :
		nAzg = len(phig);
	else :
		nAzg = 0    # for 1D

	# Number of instrument bins
	nAz     = len(phi);
	nEle    = len(th);
	nV      = len(v);

	
	# bin edges
	if veInputEdges : # get vg from vg_edges
		vg_edges    = vg_edges
		vg          = vg_edges[:-1]+0.5*np.diff(vg_edges)
		nVg         = vg.size
	else : # get vg_edges from vg
		vg_edges        = np.zeros(len(vg)+1)
		vg_edges[0]   	= vg[0]-np.diff(vg[:2])/2
		vg_edges[1:-1]  = vg[:-1]+np.diff(vg)/2
		vg_edges[-1]  	= vg[-1]+np.diff(vg[-2:])/2

	
	if base == "pol":
		if projDim == 2:
			phig_edges = np.hstack([phig-dPhig/2,phig[-1]+dPhig/2])

		# primed (grid) diffs
		dVg = np.diff(vg_edges)

		# convert to cartesian mesh, only for output
		if projDim == 2:
			[phiMesh,vMesh] = np.meshgrid(phig_edges+dPhig/2,vg)    # Creates the mesh, center of bins, phi has one extra bin at the end
			[vxMesh,vyMesh] = ct.pol2cart(phiMesh-pi/nAzg,vMesh)  # Converts to cartesian

			[phiMesh_edges,vMesh_edges] = np.meshgrid(phig_edges,vg_edges)          # Creates the mesh, edges of bins
			[vxMesh_edges,vyMesh_edges] = ct.pol2cart(phiMesh_edges,vMesh_edges)    # Converts to cartesian, edges

	elif base == "cart" :
		# for cartesian grid, the velocity bins must all be equal
		# a linearly spaced grid can have small roundoff differences in step
		# with std, there could potentially be some outlier? i dont know
		meandiff = np.mean(np.diff(vg))
		errtol = 1e-2 # 1%

		if not all((np.diff(vg)/meandiff-1)<errtol):
			raise ValueError("For a cartesian grid (default), all velocity bins diff(vg) must be equal.")

		dVg = np.diff(vg[:2])


	# 3D matrices for instrumental bin centers

	TH  = np.tile(th,(nV,nAz,1))        # [v,phi,th]
	PHI = np.tile(phi,(nV,nEle,1))      # [v,th,phi]
	PHI = np.transpose(PHI,[0,2,1])     # [v,phi,th]
	VEL = np.tile(v,(nAz,nEle,1))       # [phi,th,v]
	VEL = np.transpose(VEL,[2,0,1])     # [v,phi,th]
	DV  = np.tile(dV,(nAz,nEle,1))      # [phi,th,v]
	DV  = np.transpose(DV,[2,0,1])      # [v,phi,th]

	# Weighting of number of Monte Carlo particles
	Nsum = int(nMC*len(np.nonzero(F.data)[0]))   # total number of Monte Carlo particles

	if weight == "none":
		# 3D matrix with values of nMC for each bin
		NMC         = np.zeros(F.shape,dtype=int) # no points when data is 0
		NMC[F!=0]   = int(nMC)
	elif weight == "lin":
		NMC = np.ceil(Nsum/np.sum(np.sum(np.sum(F,axis=1),axis=1),axis=1)*F).astype(int)
	elif weight == "log":
		NMC = np.ceil(Nsum/np.sum(np.sum(np.sum(np.log10(F+1),axis=1),axis=1),axis=1)*np.log10(F+1)).astype(int)

	# "Volume" element in velocity space of an instrument bin
	#pdb.set_trace()
	dtau = np.cos(TH)*DV*dPhi*dTh*VEL**2
	# set grid data matrix and grid "area" element
	if base == "pol":
		# init Fp
		Fg  = np.zeros((nAzg+1,nVg))
		Fg_ = np.zeros((nAzg,nVg)) # use this one with 'edges bins'
		# Area or line element (primed)
		dAg = dVg*dPhig*vg**(projDim-1)
	elif base == "cart":
		Fg  = zeros((nVg,nVg))
		dAg = dVg**2

	# Perform projection
	# Loop through all instrument bins
	for i in range(nV):     # velocity (energy)
		for j in range(nAz):    # phi
			for k in range(nEle):   # theta
				
				# generate MC points
				nMCt = NMC[i,j,k] # temporary number
				# Ignore bin if value of F is zero to save computations
				if F[i,j,k] == 0:
					continue

				# Construct Monte Carlo particles within particle bins
				# first is not random
				
				dV_MC   = np.vstack([0,-np.random.rand(nMCt-1,1)*dV[i]-dVm[0]]) # velocity within [-dVm,+dVp]
				dPHI_MC = np.vstack([0,(np.random.rand(nMCt-1,1)-.5)*dPhi])
				dTH_MC  = np.vstack([0,(np.random.rand(nMCt-1,1)-.5)*dTh])

				# convert instrument bin to cartesian velocity
				[vx,vy,vz] = sfs.util.sph2cart(PHI[i,j,k]+dPHI_MC,TH[i,j,k]+dTH_MC,VEL[i,j,k]+dV_MC)

				# Get velocities in primed coordinate system
				vxp = np.dot(np.hstack([vx,vy,vz]),xphat.data)    # all MC points
				vyp = np.dot(np.hstack([vx,vy,vz]),yphat.data)
				vzp = np.dot(np.hstack([vx,vy,vz]),zphat.data)    # all MC points
				vabsp = np.sqrt(vxp**2+vyp**2+vzp**2)

				if projDim == 1 :   # get transverse velocity sqrt(vy^2+vz^2)
					vzp = np.sqrt(vyp**2+vzp**2)    # call it vzp

				alpha = pvlib.tools.asind(vzp/vabsp)

				# If "particle" is outside allowed interval, don't use point
				
				usePoint    = ((vzp >= vzint[0])*(vzp <= vzint[1])*(alpha >= aint[0])*(alpha <= aint[1]))

				if projDim == 1:
					vp = vxp
				elif base == "pol" :
					# convert to polar coordinates (phip could become negative)
					[phip,vp] = ct.cart2pol(vxp,vyp);
					# fix if negative
					phip[phip<0] = 2*np.pi+phip[phip<0]
				
				# different procedure for 1D or polar OR cartesian
				if base == "pol" or projDim == 1 :
					# ------ 1D AND POLAR CASE ------
					# get indicies for all MC points
					iVg = np.digitize(vp,vg_edges[:])
					# fixes bug that exists on some systems, may influence
					# performance
					if iVg[iVg==0].size != 0:
						iVg[iVg==0] = 999999


					if projDim == 2 :
						iAzg = np.digitize(phip,phig_edges);
					else:
						iAzg = np.zeros((1,nMCt),dtype=int)


					# Loop through MC points and add value of instrument bin to the
					# appropriate projection bin
					t = time.time()
					for l in range(nMCt):
						
						# add value to appropriate projection bin
						if (usePoint[l] and (iAzg[0,l].size != 0) and (iVg[l].size != 0) and (iAzg[0,l]<nAzg+1 or iAzg[0,l]==1) and (iVg[l]<nVg)):
							
							Fg[iAzg[0,l],iVg[l]] = Fg[iAzg[0,l],iVg[l]]+F[i,j,k]*dtau[i,j,k]/(dAg[iVg[l]]*nMCt)
							if projDim == 2 :
								Fg_[iAzg[0,l],iVg[l]] = Fg[iAzg[0,l],iVg[l]]+F[i,j,k]*dtau[i,j,k]/(dAg[iVg[l]]*nMCt)

					enlapsed = time.time()-t
					print(enlapsed)
				elif base == "cart" :
					# ------ CARTESIAN CASE ------
					# get indicies for all MC points
					iVxg = np.digitize(vxp,vg_edges);
					iVyg = np.digitize(vyp,vg_edges);
					# fixes bug that exists on some systems, may influence
					# performance
					iVxg[iVxg==0] = np.nan
					iVyg[iVyg==0] = np.nan

					# Loop through MC points and add value of instrument bin to the
					# appropriate projection bin
					
					for l in range(nMCt):
						if ((usePoint[l]) and (vxp[l]>np.min(vg_edges)) and (vxp[l]<np.max(vg_edges)) and (vyp[l]>np.min(vg_edges)) and (vyp[l]<np.max(vg_edges))):
							Fg[iVxg[l],iVyg[l]] = Fg[iVxg[l],iVyg[l]]+F[i,j,k]*dtau[i,j,k]/(dAg*nMCt)



	pdb.set_trace()
	# Output
	if ((projDim == 2) and (base == "pol")):
		# fix for interp shading, otherwise the last row can be whatever
		Fg[-1,:] = np.mean(np.vstack([Fg[-2,:],Fg[0,:]]))

	# Calculate density
	if projDim == 1:
		dens    = np.sum(Fg*dAg)
	elif base == "pol":
		dAG     = np.tile(dAg,(nAzg,1))
		dens    = np.sum(np.sum(Fg[1:-1,:]*dAG))
	elif base == "cart":
		dens    = np.sum(np.sum(Fg*dAg))

	# Calculate velocity moment
	if projDim == 1:
		vel = np.nansum(Fg*dAg*vg);
	elif base == "pol":
		VG          = np.tile(vg.T,(1,nAzg))
		PHIG        = np.tile(phig,(nVg,1))
		[VXG,VYG]   = ct.pol2cart(PHIG,VG)

		vel = [0,0]
		for l in range(nVg):
			for m in range(nAzg):
				vel = vel+[VXG[l,m],VYG[l,m]]*Fg[m,l]*dAg[l]
	elif base == "cart":
		vel = [0,0] # whatever

	vel = vel/dens


	# output
	pstdict = {}
	pstdict["F"] = Fg
	if projDim == 1:
		pstdict["v"]                = vg
		pstdict["v_edges"]          = vg_edges
	elif base == "pol":
		pstdict["vx"]               = vxMesh
		pstdict["vy"]               = vyMesh
		pstdict["F_using_edges"]    = Fg_
		pstdict["vx_edges"]         = vxMesh_edges
		pstdict["vy_edges"]         = vyMesh_edges
	elif base == "cart":
		pstdict["vx"]               = vg
		pstdict["vy"]               = vg
		pstdict["F_using_edges"]    = np.vstack([np.hstack([Fg,np.zeros((nVg,1))]),np.zeros((1,nVg+1))])
		pstdict["vx_edges"]         = vg_edges
		pstdict["vy_edges"]         = vg_edges
		
	pstdict["dens"] = dens
	pstdict["vel"]  = vel

	return pstdict
#---------------------------------------------------------------------------------------------------------------------
# End main function
#---------------------------------------------------------------------------------------------------------------------

