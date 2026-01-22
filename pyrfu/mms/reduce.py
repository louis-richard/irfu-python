#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third party imports
import numpy as np

# Built-in imports
import tqdm
import xarray as xr
from scipy.constants import electron_mass, electron_volt, proton_mass, speed_of_light

# Local imports
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.int_sph_dist import int_sph_dist
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.time_clip import time_clip
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def reduce(vdf, xyz, dim: str = "1d", base: str = "pol", **kwargs):
    r"""Reduces (integrates) 3D distribution to 1D (line) or 2D (plane).
    Draft do not use!!

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution function.
    xyz : xarray.DataArray or numpy.ndarray
        Transformation matrix from instrument frame to desired frame.
    base : str, Optional
        Base for the 2D projection either cartesian 'cart' (default) or
        polar 'pol'.
    **kwargs
        Keyword arguments

    Returns
    -------
    out : xarray.DataArray
        Time series of reduced velocity distribution function.

    """

    # Make sure the projection dimension amd base are correct
    assert base.lower() in ["cart", "pol"], "Invalid projection base!!"
    assert dim.lower() in ["1d", "2d", "3d"], "Invalid projection dimension!!"
    ndim = int(dim[0])

    # Clip the distribution. If no time interval provided use the entire
    # time series.
    # Note: irfu-matlab does the rotation matrix first and then time clip but here
    # we do time clip first to save computation time if the time interval is
    # smaller than the VDF time series.
    vdf_time = vdf.time
    tint = kwargs.get("tint", list(datetime642iso8601(vdf_time.data[[0, -1]])))
    vdf_time = time_clip(vdf_time, tint)
    vdf_energy = time_clip(vdf.energy, tint).copy()
    vdf_phi = time_clip(vdf.phi, tint).copy()
    vdf_theta = vdf.theta.copy()
    vdf_data = time_clip(vdf.data, tint).copy()

    # Lower energy bound of instrument bins. If not provided set to 0 eV.
    if "delta_energy_minus" in vdf.attrs:
        delta_energy_minu = xr.DataArray(
            vdf.attrs["delta_energy_minus"],
            coords=[vdf.time.data, vdf.idx0.data],
            dims=["time", "idx0"],
        )
    else:
        delta_energy_minu = xr.DataArray(
            np.zeros((len(vdf.time.data), len(vdf.idx0.data))),
            coords=[vdf.time.data, vdf.idx0.data],
            dims=["time", "idx0"],
        )

    delta_energy_minu = time_clip(delta_energy_minu, tint)

    # make input distribution to SI units, s^3/m^6
    if vdf.data.attrs["UNITS"].lower() == "s^3/cm^6":
        vdf_data *= 1e12
    elif vdf.data.attrs["UNITS"].lower() == "s^3/m^6":
        vdf_data *= 1e0
    elif vdf.data.attrs["UNITS"].lower() == "s^3/km^6":
        vdf_data *= 1e-18
    else:
        raise ValueError("Invalid units!!")

    # Get VDF dimension (time, energy, phi, theta)
    # n_t, _, n_ph, _ = vdf_data.shape
    n_t, n_en, _, _ = vdf_data.shape

    # Construct or resample the time series of transformation matrix
    if isinstance(xyz, xr.DataArray):
        # If time series, clip to time interval and resample to VDF's time line
        xyz = time_clip(xyz, tint)
        xyz = resample(xyz, vdf_time).data
    elif isinstance(xyz, np.ndarray) and xyz.ndim == 2:
        assert xyz.shape == (3, 3), "xyz must be a transformation matrix!!"
        # If matrix, tile to VDF timeline
        xyz = np.tile(xyz, (n_t, 1, 1))
    else:
        raise TypeError("Invalid type for xyz")

    # Check that the transformation matrices are orthonormal direct
    x_phat_ts, y_phat_ts, z_phat_ts = [xyz[:, :, i] for i in range(3)]
    x_phat_ts /= np.linalg.norm(x_phat_ts, axis=1, keepdims=True)
    y_phat_ts /= np.linalg.norm(y_phat_ts, axis=1, keepdims=True)
    z_phat_ts = np.cross(x_phat_ts, y_phat_ts)
    z_phat_ts /= np.linalg.norm(z_phat_ts, axis=1, keepdims=True)
    y_phat_ts = np.cross(z_phat_ts, x_phat_ts)

    # Construct the time series of transformation matrix
    xyz_ts = np.transpose(
        np.stack([x_phat_ts, y_phat_ts, z_phat_ts]),
        [1, 2, 0],
    )

    # This is now done in int_sph_dist
    # Set azimuthal angle projection grid.
    # d_phi = kwargs.get("d_phi", 2 * np.pi / n_ph)
    # phi_grid = np.linspace(0, 2 * np.pi, n_ph) + d_phi / 2  # centers

    # Read the keyword arguments
    # Velocity projection grid and edges
    velocity_grid = kwargs.get("vg", None)  # TODO : check that for no input!!
    velocity_grid_edges = kwargs.get("vg_edges", None)

    # azimuthal angle of projection plane
    n_phi_grid = len(vdf_phi)
    d_phi_g = 2 * np.pi / n_phi_grid
    phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, n_phi_grid) + d_phi_g / 2
    phi_grid = kwargs.get("phig", phi_grid)

    # Number of Monte Carlo iterations and weights
    n_mc = kwargs.get("n_mc", 100)  # OK
    weight = kwargs.get("weight", None)  # OK

    # Velocity and azimuthal angle integration intervals
    v_int = kwargs.get("v_int", [-np.inf, np.inf])  # OK
    a_int = kwargs.get("a_int", [-180.0, 180.0])  # OK

    # Spacecraft potential to account for photoelectrons. If not provided set
    # to 0 V.
    sc_pot = kwargs.get("sc_pot", ts_scalar(vdf_time.data, np.zeros(n_t)))
    sc_pot = resample(sc_pot, vdf_time).data

    # Threshold energy (e.g., to remove background noise). If not provided set
    # to 0 eV.
    lower_e_lim = kwargs.get("lower_e_lim", 0.0)

    if isinstance(lower_e_lim, xr.DataArray):
        # If time series, clip to time interval and resample to VDF's time line
        lower_e_lim = resample(lower_e_lim, vdf_time).data
    elif isinstance(lower_e_lim, float):
        # If float, tile to VDF's timeline
        lower_e_lim = np.tile(lower_e_lim, n_t)
    else:
        raise TypeError("Invalid lower_e_lim!!")

    # Set particle specie mass from VDF's attribute "species"
    if vdf.species.lower() == "electrons":
        m_p = electron_mass
    elif vdf.species.lower() == "ions":
        m_p = proton_mass
    else:
        raise ValueError("Invalid species!!")

    # Convert maximum energy from instrument to velocity (relativistically
    # correct)
    e_max = vdf_energy.data[0, -1] + vdf.attrs["delta_energy_plus"][0, -1]
    gamma_max = 1 + electron_volt * e_max / (m_p * speed_of_light**2)
    v_max = speed_of_light * np.sqrt(1 - 1 / gamma_max**2)  # m/s

    # initiate projected f
    if velocity_grid_edges is not None:
        n_vg = len(velocity_grid_edges) - 1
        velocity_grid = velocity_grid_edges[:-1] + 0.5 * np.diff(velocity_grid_edges)
    elif velocity_grid is not None:
        n_vg = len(velocity_grid)
    else:
        n_vg = 100
        if base == "cart":
            n_vg = 100
        elif base == "pol":
            if ndim == 1:
                n_vg = 2 * n_en
            else:
                n_vg = n_en

    # Initialize output arrays
    f_g = np.zeros([n_t, *[n_vg] * ndim])
    all_v = {f"v{chr(120 + i)}": np.zeros((n_t, n_vg)) for i in range(ndim)}

    for i_t in tqdm.tqdm(range(n_t), ncols=60):  # display progress
        # 3d data matrix for time index
        f_3d = np.squeeze(vdf_data.data[i_t, ...])  # s^3/m^6
        f_3d = f_3d.astype(np.float64).copy()  # Convert to float64

        # Energies
        energy = vdf_energy.data[i_t, :]
        energy = energy.astype(np.float64)  # Convert to float64

        # Assign zeros to phase-space density values corresponding to energies
        # below the threshold energy or below the spacecraft potential if
        # provided.
        thresh_e = np.max([lower_e_lim[i_t], sc_pot[i_t]])
        e_min_idx = np.where(energy - delta_energy_minu[i_t, :] > thresh_e)[0][0]
        f_3d[:e_min_idx] = 0.0

        # Correct energy shift due to spacecraft potential
        energy -= sc_pot[i_t]
        energy[energy < 0] = 0.0

        # Convert energy to velocity (relativistically correct)
        gamma = 1 + electron_volt * energy / (m_p * speed_of_light**2)
        velocity = speed_of_light * np.sqrt(1 - 1 / gamma**2)  # m/s

        # azimuthal angle
        if vdf_phi.ndim == 2:
            phi = vdf_phi.data[i_t, :].astype(np.float64)  # in degrees
        else:
            # Fast mode
            phi = vdf_phi.data.astype(np.float64)  # in degrees

        # Remove 180 deg to convert from instrument to physical frame
        phi = np.deg2rad(phi - 180.0)  # in radians

        # elevation angle and remove 90 deg to convert from instrument to
        # physical frame.
        theta = vdf_theta.data.astype(np.float64)  # in degrees
        theta = np.deg2rad(theta - 90.0)  # in radians

        # TODO: add delta_phi and delta_theta if needed

        # Set velocity projection grid.
        if velocity_grid is None:
            if base == "cart":
                velocity_grid = np.linspace(-v_max, v_max, 100, endpoint=True)
            elif base == "pol":
                if ndim == 1:
                    velocity_grid = np.hstack((-np.flip(velocity), velocity))
                else:
                    velocity_grid = velocity

        else:
            pass

        options = {
            "xyz": xyz_ts[i_t, ...],
            "n_mc": n_mc,
            "weight": weight,
            "v_int": v_int,
            "a_int": a_int,
            "projection_dim": dim,
            "projection_base": base,
            "velocity_grid_edges": velocity_grid_edges,
        }

        tmpst = int_sph_dist(
            f_3d, velocity, phi, theta, velocity_grid, phi_grid, **options
        )

        f_g[i_t, ...] = tmpst["f"]

        for i in range(ndim):
            all_v[f"v{chr(120 + i)}"][i_t, :] = tmpst[f"v{chr(120 + i)}"] / 1e3  # km/s

    # Build output as a time series with dimensions:
    #   - (time x vx) for 1D reduced distribution
    #   - (time x vx x vy) for 2D reduced distribution
    coords = [
        vdf_time.data,
        *[all_v[f"v{chr(120 + i)}"][0, :] for i in range(ndim)],
    ]
    dims = ["time", *[f"v{chr(120 + i)}" for i in range(ndim)]]
    out = xr.DataArray(f_g, coords=coords, dims=dims)

    return out
