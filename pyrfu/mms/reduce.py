#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reduce.py

@author : Louis RICHARD
"""


import numpy as np
import xarray as xr

from astropy import constants
from astropy.time import Time

from ..pyrf import resample, norm, int_sph_dist


def reduce(dist, dim, x, *args, **kwargs):
    # Check to what dimension the distribution is to be reduced
    if dim == "1D" or dim == "2D":
        dim = int(dim[0])  # input dim can either be '1D' or '2D'
    else:
        raise ValueError("First input must be a string deciding projection type, either ''1D'' or ''2D''.")

    if dim == 1:  # 1D: projection to line
        if isinstance(x, xr.DataArray):
            xphat_mat = norm(resample(x, dist.data)).data
        elif isinstance(x, list) and len(x) == 3:
            xphat_mat = np.tile(np.array(x), (len(dist), 1))
        elif isinstance(x, np.ndarray) and x.shape == (len(dist), 3):
            xphat_mat = x
        else:
            raise TypeError("Invalid type for x")

        xphat_mat = norm(resample(x, dist.data))
        xphat_amplitude = np.linalg.norm(xphat_mat, axis=1, keepdims=True)

        if np.abs(np.mean(xphat_amplitude) - 1) < 1e-2 < np.std(xphat_amplitude):  # make sure x are unit vectors
            xphat_mat = xphat_mat / np.linalg.norm(xphat_mat, axis=1, keepdims=True)
            print("warning : |<x/|x|>-1| > 1e-2 or std(x/|x|) > 1e-2: x is recalculated as x = x/|x|.\n")

    elif dim == 2:
        if isinstance(x, xr.DataArray) and isinstance(args[0], xr.DataArray):
            y = args[0]  # assume other coordinate for perpendicular plane is given after and in same format
            args = args[1:]
            xphat_mat = norm(resample(x, dist.data)).data
            yphat_mat = norm(resample(y, dist.data)).data
        elif isinstance(x, list) and len(x) == 3:
            y = args[0]  # assume other coordinate for perpendicular plane is given after and in same format
            args = args[1:]
            xphat_mat = np.tile(np.array(x), (len(dist), 1))
            yphat_mat = np.tile(np.array(y), (len(dist), 1))
        elif isinstance(x, np.ndarray) and x.shape == (len(dist), 3):
            y = args[0]  # assume other coordinate for perpendicular plane is given after and in same format
            args = args[1:]
            xphat_mat = x
            yphat_mat = y

        else:
            raise ValueError("Can''t recognize second vector for the projection plane, ''y'': PDist.reduce(''2D'',x,y,...)\n")

        # it's x and z that are used as input to irf_int_sph_dist
        # x and y are given, but might not be orthogonal
        # first make x and y unit vectors
        xphat_amplitude = np.linalg.norm(xphat_mat, axis=1)
        yphat_amplitude = np.linalg.norm(yphat_mat, axis=1)

        # These ifs are not really necessary, but could be there if one
        # wants to add some output saying that they were not put in
        # (inputted) as unit vectors. The definition of unit vectors is not
        # quite clear, due to tiny roundoff(?) errors
        if np.abs(np.mean(xphat_amplitude) - 1) < 1e-2 < np.std(xphat_amplitude):  # make sure x are unit vectors,
            xphat_mat = xphat_mat / xphat_amplitude[:, np.newaxis]
            print("warning |<x/|x|>-1| > 1e-2 or std(x/|x|) > 1e-2: x is recalculated as x = x/|x|.\n")

        if np.abs(np.mean(yphat_amplitude) - 1) < 1e-2 < np.std(yphat_amplitude):  # make sure y are unit vectors,
            yphat_mat = yphat_mat / yphat_amplitude[:, np.newaxis]
            print("warning |<y/|y|>-1| > 1e-2 or std(y/|y|) > 1e-2: y is recalculated as y = y/|y|.\n")

        # make z orthogonal to x and y
        zphat_mat = np.cross(xphat_mat, yphat_mat, axis=1)
        zphat_amplitude = np.linalg.norm(zphat_mat, axis=1)
        zphat_mat = zphat_mat / zphat_amplitude[:, np.newaxis]
        # make y orthogonal to z and x
        yphat_mat = np.cross(zphat_mat, xphat_mat, axis=1)
        # check amplitude again, incase x and y were not orthogonal
        yphat_amplitude = np.linalg.norm(yphat_mat, axis=1)

        if np.abs(np.mean(yphat_amplitude) - 1) < 1e-2 < np.std(yphat_amplitude):  # make sure y are unit vectors,
            yphat_mat = yphat_mat / yphat_amplitude[:, np.newaxis]
            print("warning |<y/|y|>-1| > 1e-2 or std(y/|y|) > 1e-2: y is recalculated as y = y/|y|.\n")

        nargs = nargs - 1

        # Set default projection grid, can be overriden by given input 'phig'
        nAzg = 32
        dPhig = 2 * np.pi / nAzg
        phig = np.linspace(0, 2 * pi - dPhig, nAzg) + dPhig / 2  # centers

    # make input distribution to SI units, s^3/m^6
    # dist = dist.convertto('s^3/m^6');

    # Check for input flags
    # Default options and values
    doTint = False
    doLowerElim = False
    nMC = 100  # number of Monte Carlo iterations
    vint = [-np.inf, np.inf]
    aint = [-180, 180]  # azimuthal intherval
    vgInput = 0
    vgInputEdges = 0
    weight = "none"
    correct4scpot = 0
    base = "cart"  # coordinate base, cart or pol

    if dist.attrs["species"] == "electrons":
        isDes = True
    else:
        isDes = False

    ancillary_data = {}

    if "tint" in kwargs:
        tint = kwargs["tint"]
        doTint = True

    if "nmc" in kwargs:
        nMC = kwargs["nmc"]
        ancillary_data["nMC"] = nMC

    if "vint" in kwargs:
        vint = kwargs["vint"]

    if "aint" in kwargs:
        aint = kwargs["aint"]

    if "phig" in kwargs:
        phig = kwargs["phig"]

    if "vg" in kwargs:
        vgInput = True
        vg = kwargs["vg"] * 1e3

    if "vg_edges" in kwargs:
        vgInputEdges = True
        vg_edges = kwargs["vg_edges"] * 1e3

    if "weight" in kwargs:
        weight = kwargs["weight"]
        ancillary_data["weight"] = weight

    if "scpot" in kwargs:
        scpot = kwargs["scpot"]
        ancillary_data["scpot"] = scpot
        correct4scpot = True

    if "lowerelim" in kwargs:
        lowerelim = kwargs["lowerelim"]
        ancillary_data["lowerelim"] = lowerelim
        doLowerElim = True
        if isinstance(lowerelim, xr.DataArray):
            lowerelim = resample(inp=lowerelim, ref=dist).data
        elif isinstance(lowerelim, (list, np.ndarray)) and len(lowerelim) == len(dist):
            lowerelim = lowerelim
        elif isinstance(lowerelim, float):
            lowerelim = np.tile(lowerelim, (len(dist), 1))
        else:
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
        lowerelim_mat = np.tile(lowerelim, len(emat[0, :]))

    if correct4scpot:
        scpot = resample(tlim(scpot, dist), dist)
        scpot_mat = np.tile(scpot.data, len(emat[0, :]))

    if isDes:
        M = constants.m_e.value
    else:
        M = constants.m_p.value

    if doTint:  # get time indicies
        tck = interpolate.interp1(dist.time.data.view("i8") * 1e-9, np.arange(len(dist.time)), kind='nearest')

        if len(tint) == 1:  # single time
            its = tck(Time(tint, format="isot").unix)
        else:  # time interval
            it1 = tck(Time(tint[0], format="isot").unix)
            it2 = tck(Time(tint[1], format="isot").unix)
            its = np.arange(it1, it2)
    else:  # use entire PDist
        its = np.arange(len(dist.data))

    nt = len(its)
    if nt == 0:
        raise ValueError("Empty time array. Please verify the time(s) given.")

    # try to make initialization and scPot correction outside time-loop
    if not any([vgInput, vgInputEdges]):  # prepare a single grid outside the time-loop
        emax = emat[0, -1] + dist.attrs["delta_energy_plus"][1, -1].data
        vmax = constants.c.value * np.sqrt(1 - (emax * constants.e.value / (M * constants.c.value ** 2) - 1) ** 2)
        nv = 100
        vgcart_noinput = np.linspace(-vmax, vmax, nv)
        print(
            "warning : No velocity grid specified, using a default vg = linspace(-vmax,vmax,{:d}), with vmax = {:3.2f} km/s.".format(
                nv, vmax * 1e-3))

    # loop to get projection

    all_vg = np.zeros((nt, 100))
    all_vg_edges = np.zeros((nt, 101))
    for i, it in enumerate(tqdm(its)):  # display progress
        if dim == 1:
            xphat = xphat_mat.data[i, :]
        elif dim == 2:
            xphat = xphat_mat.data[i, :]  # corresponding to phi = 0 in 'phig'
            zphat = zphat_mat.data[i, :]  # normal to the projection plane

        # 3d data matrix for time index it
        F3d = np.squeeze(dist.data[it, ...])  # s^3/m^6
        energy = emat[it, :]

        if doLowerElim:
            remove_extra_ind = 0  # for margin, remove extra energy channels
            ie_below_elim = np.nonzero(np.abs(emat[it, :] - lowerelim_mat[it, :]) == np.min(
                np.abs(emat[it, :] - lowerelim_mat[it, :])))  # closest energy channel

            F3d[:(np.max(ie_below_elim) + remove_extra_ind), ...] = 0

        if correct4scpot:
            if "delta_energy_minus" in dist.attrs:  # remove all that satisfies E-Eminus<Vsc
                ie_below_scpot = \
                np.nonzero(emat[it, :] - dist.attrs["delta_energy_minus"][it, :] - scpot_mat[it, 0] < 0, )[-1]
            else:
                ie_below_scpot = np.nonzero(np.abs(emat[it, :] - scpot_mat[it, :]) == np.min(
                    np.abs(emat[it, :] - scpot_mat[it, :])))  # closest energy channel

            remove_extra_ind = 0  # for margin, remove extra energy channels

            F3d[1:(np.max(ie_below_scpot) + remove_extra_ind), ...] = 0

            energy = energy - scpot_mat[it, :]
            energy[energy < 0] = 0

        v = constants.c.value * np.sqrt(
            1 - (energy * constants.e.value / (M * constants.c.value ** 2) - 1) ** 2)  # m/s

        # azimuthal angle
        if dist.phi.ndim != 1:
            phi = dist.phi.data[it, :]  # in degrees
        else:  # fast mode
            phi = dist.phi.data  # in degrees

        phi = phi - 180
        phi = phi * np.pi / 180  # in radians

        # elevation angle
        th = dist.theta.data  # polar angle in degrees
        th = th - 90  # elevation angle in degrees
        th = th * np.pi / 180  # in radians

        # Set projection grid after the first distribution function
        # bin centers
        if vgInputEdges:  # redefine vg (which is vg_center)
            vg = vg_edges[1:-1] + 0.5 * np.diff(vg_edges)
        elif vgInput:
            vg = vg
        else:  # define from instrument velocity bins
            if base == "cart":
                vg = vgcart_noinput  # maybe just bypass this and go directly through input vg_edges?
            else:
                if dim == 1:
                    vg = np.hstack((-np.flip(V), v))
                elif dim == 2:
                    vg = v

        # initiate projected f
        if i == 0:
            if dim == 1:
                Fg = np.zeros((nt, len(vg)))
                vel = np.zeros((nt, 1))
            elif dim == 2 and base == "pol":
                Fg = np.zeros((nt, len(phig), len(vg)))
                vel = np.zeros((nt, 2))
            elif dim == 2 and base == "cart":
                Fg = np.zeros((nt, len(vg), len(vg)))
                vel = np.zeros((nt, 2))

            dens = np.zeros((nt, 1))

        # perform projection
        if dim == 1:  # 1D plane
            # v, phi, th corresponds to the bins of F3d
            if vgInputEdges:
                tmpst = int_sph_dist(F3d, v, phi, th, vg, x=xphat, nmc=nMC, vzint=vint * 1e3, aint=aint, weight=weight,
                                     vg_edges=vg_edges)
            else:

                tmpst = int_sph_dist(F3d, v, phi, th, vg, x=xphat, nmc=nMC, vzint=np.array(vint) * 1e3, aint=aint,
                                     weight=weight)
                pdb.set_trace()
            all_vg[i, :] = tmpst["v"]  # normally vg, but if vg_edges is used, vg is overriden
            all_vg_edges[0, :] = tmpst["v_edges"]
        elif dim == 2:
            # is 'vg_edges' implemented for 2d?
            tmpst = int_sph_dist(F3d, v, phi, th, vg, x=xphat, z=zphat, phig=phig, nmc=nMC, vzint=vint * 1e3,
                                 weight=weight, base=base)

            all_vx[i, ...] = tmpst["vx"]
            all_vy[i, ...] = tmpst["vy"]
            all_vx_edges[i, ...] = tmpst["vx_edges"]
            all_vy_edges[i, ...] = tmpst["vy_edges"]

        # fix for special cases
        # dimension of projection, 1D if projection onto line, 2D if projection onto plane
        if dim == 1 or base == "cart":
            Fg[i, ...] = tmpst["F"]
        elif dim == 2:
            Fg[i, ...] = tmpst["F_using_edges"]

        # set moments from reduced distribution (for debug)
        dens[i] = tmpst["dens"]
        vel[i, :] = tmpst["vel"]

    # Construct PDist objects with reduced distribution
    # vg is m/s, transform to km/s
    if dim == 1:
        # Make output
        PD = xr.DataArray(data=Fg, coords=[dist.time[its], all_vg * 1e-3], dims=["time", "vg"])
        # attributes
        PD.attrs["vg_edges"] = all_vg_edges
    elif dim == 2 and base == "pol":
        Fg_tmp = Fg
        all_vx_tmp = np.transose(all_vx[..., 1:end - 1], [0, 1, 2]) * 1e-3
        all_vy_tmp = np.transose(all_vy[..., 1:end - 1], [0, 1, 2]) * 1e-3
        all_vx_edges_tmp = np.transose(all_vx_edges, [0, 1, 2]) * 1e-3
        all_vy_edges_tmp = np.transose(all_vy_edges, [0, 1, 2]) * 1e-3
        # Make output
        PD = xr.DataArray(data=Fg_tmp, coords=[dist.time[its], all_vx_tmp, all_vy_tmp], dims=["time", "vx", "vy"])
        # attributes
        PD.attrs["vx_edges"] = all_vx_edges_tmp
        PD.attrs["vy_edges"] = all_vy_edges_tmp
        PD.attrs["base"] = "pol"
    elif dim == 2 and base == "cart":
        # Make output
        PD = xr.DataArray(data=Fg_tmp, coords=[dist.time[its], all_vx * 1e-3, all_vx * 1e-3], dims=["time", "vx", "vx"])
        # attributes
        PD.attrs["vx_edges"] = all_vx_edges * 1e-3
        PD.attrs["vy_edges"] = all_vx_edges * 1e-3
        PD.attrs["base"] = "cart"

    PD.attrs["species"] = dist.attrs["species"]
    # PD.userData = dist.userData;
    PD.attrs["v_units"] = "km/s"

    # set units and projection directions
    if dim == 1:
        PD.attrs["units"] = 's/m^4';
        PD.attrs["projection_direction"] = xphat_mat[its, :]
    elif dim == 2:
        PD.attrs["units"] = 's^2/m^5';
        PD.attrs["projection_dir_1"] = xphat_mat[its, :]
        PD.attrs["projection_dir_2"] = yphat_mat[its, :]
        PD.attrs["projection_axis"] = zphat_mat[its, :]

    if doLowerElim:
        PD.attrs["lowerelim"] = lowerelim_mat