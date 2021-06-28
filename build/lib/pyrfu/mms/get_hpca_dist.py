#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

mass_and_charge = {"hydrogen+": [1.04535e-2, 1], "helium+": [4.18138e-2, 1],
                   "helium++": [4.18138e-2, 2], "oxygen+": [0.167255, 1],
                   "oxygen++": [0.167255, 2]}


def _get_energy(inp, dim):
    energy_len = len(inp.ccomp.data)
    energy_reform = np.reshape(inp.ccomp.data, [energy_len, 1, 1])
    # repeated across theta
    energy_rebin1 = np.repeat(energy_reform, dim[1], axis=2)
    out_energy = np.repeat(energy_rebin1, dim[2], axis=1)

    return out_energy


def _get_theta(inp, azimuth_dim):
    energy_len = len(inp.ccomp.data)

    # elevations are constant across time
    # convert colat -> lat
    theta_len = len(inp.rcomp.data)
    theta_reform = 90. - np.reshape(inp.rcomp.data, [1, 1, theta_len])

    # in the IDL code, we use reform to repeat the vector above
    # here, we'll do the same thing with np.repeat
    # repeated across phi
    theta_rebin1 = np.repeat(theta_reform, azimuth_dim[1], axis=1)
    # repeated across energy
    out_theta = np.repeat(theta_rebin1, energy_len, axis=0)

    return out_theta


def _get_phi(azimuth, full):
    # get azimuth
    # -shift from time-azi.-elev.-en. to time-en.-azi.-elev.
    out_phi = azimuth.data[full, :, :, :]

    if out_phi.ndim == 4:
        out_phi = out_phi.transpose([0, 3, 1, 2])
    elif out_phi.ndim == 3:
        out_phi = out_phi.transpose([2, 0, 1])

    return out_phi


def _get_dphi(azimuth, full, out_phi, inp):
    energy_len = len(inp.ccomp.data)
    theta_len = len(inp.rcomp.data)
    phi_len = azimuth.shape[1]
    # get dphi
    #  - use median distance between subsequent phi measurements within each
    #  distribution (median is used to discard large differences across 0=360)
    #  - preserve dimensionality in case differences arise across energy or
    #  elevation

    if out_phi.ndim == 4:
        out_dphi = np.median(np.diff(azimuth.data[full, ...], axis=1), axis=1)
        out_dphi = np.transpose(out_dphi, [0, 2, 1])
        dphi_reform = np.reshape(out_dphi,
                                 [full.size, energy_len, theta_len, 1])
        out_dphi = np.repeat(dphi_reform, phi_len, axis=3)
    elif out_phi.ndim == 3:
        out_dphi = np.median(np.diff(azimuth.data[full, ...], axis=1), axis=0)
        out_dphi = out_dphi.transpose([1, 0])
        dphi_reform = np.reshape(out_dphi, [energy_len, theta_len, 1])
        out_dphi = np.repeat(dphi_reform, phi_len, axis=2)
    else:
        raise TypeError("Invalid shape of out_phi")

    return out_dphi


def get_hpca_dist(inp, azimuth):
    r"""Returns pseudo-3D particle data structures containing mms hpca data
    for use with spd_slice2d.

    Paramters
    ---------
    inp : xarray.DataArray
        HPCA ion spec
    azimuth : xarray.DataArray
        Azimuthal angles.

    Returns
    -------
    out : dict
        HPCA distribution


    """
    # check if the time series is monotonic to avoid doing incorrect
    # calculations when there's a problem with the CDF files
    time_data = azimuth.time.data

    n_mono = np.where(time_data[1:] <= time_data[:-1])
    assert len(n_mono[0]) == 0, "non-monotonic data found in the Epoch_Angles"

    # find azimuth times with complete 1/2 spins of particle data this is
    # used to determine the number of 3D distributions that will be created
    # and where their corresponding data is located in the particle data
    # structure
    n_times = len(azimuth.data[0, 0, :, 0])
    data_idx = np.searchsorted(inp.time.data, time_data) - 1
    full = np.argwhere((data_idx[1:] - data_idx[:-1]) == n_times)
    assert len(full) != 0, "Azimuth data does not cover current time range"

    # filter times when azimuth data is all zero
    #   -just check the first energy & elevation
    #   -assume azimuth values are positive
    n_expected = azimuth.data[full, 0, 0, :].size
    n_valid_az = np.count_nonzero(azimuth.data[full, 0, 0, :])
    assert n_valid_az == n_expected, "zeroes found in the azimuth array"

    full = np.squeeze(full)

    data_idx = data_idx[full].flatten()

    # Initialize energies, angles, and support data
    # final dimensions for a single distribution (energy-azimuth-elevation)
    azimuth_dim = azimuth.data.shape
    dim = (azimuth_dim[3], azimuth_dim[2], azimuth_dim[1])

    specie = inp.attrs["FIELDNAM"].split(" ")[0].lower()

    mass, charge = mass_and_charge[specie]

    out_bins = np.ones(dim) + 1
    out_denergy = np.zeros(dim)

    energy_len = len(inp.ccomp.data)
    theta_len = len(inp.rcomp.data)
    phi_len = azimuth_dim[1]

    # energy bins are constant
    out_energy = _get_energy(inp, dim)

    out_theta = _get_theta(inp, azimuth_dim)

    out_dtheta = np.zeros([energy_len, theta_len, azimuth_dim[2]]) + 22.5

    out_phi = _get_phi(azimuth, full)

    out_dphi = _get_dphi(azimuth, full, out_phi, inp)

    out_data = np.zeros((full.size, dim[0], dim[1], dim[2]))

    # copy particle data
    for i in range(full.size):
        # need to extract the data from the center of the half-spin
        if data_idx[i] - n_times / 2.0 < 0:
            # start_idx = 0
            continue
        else:
            start_idx = int(data_idx[i] - n_times / 2.)

        if data_idx[i] + n_times / 2. - 1 >= len(inp.time):
            # end_idx = len(inp.time)
            continue
        else:
            end_idx = int(data_idx[i] + n_times / 2.)

        out_data[i, ...] = np.transpose(inp.data[start_idx:end_idx, :, :],
                                        [2, 0, 1])

    out = {'data': out_data, 'bins': out_bins, 'theta': out_theta,
           'phi': out_phi, 'energy': out_energy, 'dtheta': out_dtheta,
           'dphi': out_dphi, 'denergy': out_denergy, 'n_energy': energy_len,
           'n_theta': theta_len, 'n_phi': phi_len, 'n_times': full.size,
           'project_name': 'MMS', 'species': specie, 'charge': charge,
           'units_name': 'df_cm', 'mass': mass}

    return out
