#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import special, constants

# Local imports
from..pyrf import resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _thermal_velocity(en, specie):
    if specie.lower() == "ions":
        mass = constants.proton_mass
    elif specie.lower() == "electrons":
        mass = constants.electron_mass
    else:
        raise ValueError("Invalid specie")

    return np.sqrt(2 * en * constants.electron_volt / mass)


def make_model_kappa(vdf, n, v_xyz, t, kappa: float = 7.):
    r"""Make a general isottropic kappa distribution function based on
    particle moment data in the same format as vdf.

    Parameters
    ----------
    vdf : xarray.Dataset
        Particle distribution (skymap).
    n : xarray.DataArray
        Time series of the number density.
    v_xyz : xarray.DataArray
        Time series of the bulk velocity.
    t : xarray.DataArray
        Time series of the scalar temperature.
    kappa : float. Optional
        Kappa index, Default is 7.

    Returns
    -------
    out : xarray.Dataset
        Distribution function in the same format as vdf.

    See also
    --------
    pyrfu.mms.make_model_vdf
    """

    # Unpack azimuthal and elevation angles from the skymap distribution

    phi = vdf.phi.data[0, :]
    theta = vdf.theta.data

    # Resample number density, bulk velocity and temperature to skymap sampling
    n = resample(n, vdf.time)
    v_xyz = resample(v_xyz, vdf.time)
    t = resample(t, vdf.time)

    # Compute thermal velocity
    v_th = _thermal_velocity(t.data, vdf.attrs["species"])

    # Initialize output to zeros
    out_data = np.zeros([vdf.data.shape[0], vdf.data.shape[3],
                         vdf.data.shape[1], vdf.data.shape[2]])

    for i in range(len(vdf.time.data)):
        energies = vdf.energy.data[i, :]

        # Construct velocities matrix
        en_mat, theta_mat, phi_mat = np.meshgrid(energies, theta, phi)
        v_x_hat = np.sin(np.deg2rad(theta_mat)) * np.cos(np.deg2rad(phi_mat))
        v_y_hat = np.sin(np.deg2rad(theta_mat)) * np.sin(np.deg2rad(phi_mat))
        v_z_hat = np.cos(np.deg2rad(theta_mat))
        v_mag = _thermal_velocity(en_mat, vdf.attrs["species"])
        v_mat = np.stack([v_mag * v_x_hat, v_mag * v_y_hat, v_mag * v_z_hat])

        # Unpack thermal and bulk velocities
        v_thk = (kappa - 3 / 2) * v_th[i] ** 2
        v_bulk = v_xyz.data[i, ...]

        # Shift velocities with lbulk velocity
        v_squ = np.sum((v_mat - v_bulk[:, None, None, None]) ** 2, axis=0)

        # Compute kappa distribution
        coef1 = 1 / (np.pi * v_thk) ** (3 / 2)
        coef2 = special.gamma(kappa + 1) / special.gamma(kappa - 1 / 2)
        out_data[i, ...] = (1 + v_squ / v_thk) ** -(kappa + 1)
        out_data[i, ...] *= coef1 * coef2

    # Scale with number density
    out_data *= n.data[:, None, None, None]

    # Convert to differential particule flux
    # out_data *= np.tile(en_mat, (n_t, 1, 1, 1)) * 1e15 / 0.53707

    # Transform to usual shape
    out_data = np.transpose(out_data, [0, 2, 3, 1])

    # Creates Dataset
    out_dict = {f"idx{i}": range(k) for i, k in enumerate(out_data.shape[1:])}
    out_dict["data"] = (["time", "idx0", "idx1", "idx2"], out_data)
    out_dict["energy"] = (["time", "idx0"], vdf.energy.data)
    out_dict["phi"] = (["time", "idx1"], vdf.phi.data)
    out_dict["theta"] = (["idx2"], vdf.theta.data)
    out_dict["time"] = vdf.time.data

    out = xr.Dataset(out_dict)

    out.attrs["UNITS"] = "s^3/m^6"
    out.attrs["species"] = vdf.attrs.get("species", None)
    out.attrs["energy_dminus"] = vdf.attrs.get("energy_dminus", None)
    out.attrs["energy_dplus"] = vdf.attrs.get("energy_dplus", None)

    return out
