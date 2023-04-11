#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third party imports
import tqdm
import pdb

import numpy as np
import xarray as xr

from scipy import constants

from ..pyrf import resample, time_clip, datetime642iso8601, ts_scalar, int_sph_dist

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"


def reduce(vdf, xyz: np.ndarray, dim: str = "1d", base: str = "pol", **kwargs):
    r"""draft do not use!!

    Args:
        vdf:
        dim:
        xyz:
        **kwargs:

    Returns:

    """
    # Make sure the projection dimension amd base are correct
    assert dim.lower() in ["1d", "2d", "3d"], "Invalid projection dimension!!"
    assert base.lower() in ["cart", "pol"], "Invalid projection base!!"

    # Get time line
    vdf_time = vdf.time.copy()
    n_t, n_e, n_ph, n_th = vdf.data.shape

    # Lower energy bound of instrument bins
    delta_energy_minu = xr.DataArray(
        vdf.attrs["delta_energy_minus"],
        coords=[vdf_time.data, np.arange(n_e)],
        dims=["time", "idx0"],
    )

    # Clip the distribution. If no time interval provided use the entire time series.
    tint = kwargs.get("tint", list(datetime642iso8601(vdf_time[[0, -1]])))
    vdf_time = time_clip(vdf_time, tint).copy()
    vdf_energy = time_clip(vdf.energy, tint).copy()
    vdf_phi = time_clip(vdf.phi, tint).copy()
    vdf_theta = vdf.theta.copy()
    vdf_data = time_clip(vdf.data, tint).copy()

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
    n_t, n_e, n_ph, n_th = vdf_data.shape

    # Construct or resample the time series of transformation matrix
    if isinstance(xyz, xr.DataArray):
        xyz = resample(xyz, vdf_time).data
    elif isinstance(xyz, np.ndarray) and xyz.ndim == 2:
        assert xyz.shape == (3, 3), "xyz must be a transformation matrix!!"
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
    xyz_ts = np.transpose(np.stack([x_phat_ts, y_phat_ts, z_phat_ts]), [1, 2, 0])

    # Set azimuthal angle projection grid
    d_phi = kwargs.get("d_phi", 2 * np.pi / n_ph)
    phi_grid = np.linspace(0, 2 * np.pi, n_ph) + d_phi / 2  # centers

    # Number of Monte Carlo iterations and weights
    n_mc = kwargs.get("n_mc", 100)
    weight = kwargs.get("weight", None)

    v_lim = kwargs.get("v_lim", [-np.inf, np.inf])  # Velocity grid limits
    a_lim = kwargs.get("a_lim", [-180.0, 180.0])  # Azimuthal angle limits

    # Spacecraft potential to account for photoelectrons. If not provided set to 0 V.
    sc_pot = kwargs.get("sc_pot", ts_scalar(vdf_time.data, np.zeros(n_t)))
    sc_pot = resample(sc_pot, vdf_time).data

    # Threshold energy (e.g., to remove background noise). If not provided set to 0 eV.
    lower_e_lim = kwargs.get("lower_e_lim", 0.0)

    if isinstance(lower_e_lim, xr.DataArray):
        lower_e_lim = resample(lower_e_lim, vdf_time).data
    elif isinstance(lower_e_lim, float):
        lower_e_lim = np.tile(lower_e_lim, n_t)
    else:
        raise TypeError("Invalid lower_e_lim!!")

    if vdf.species.lower() in ["electron", "electrons"]:
        m_p = constants.electron_mass
    elif vdf.species.lower() in ["ion", "ions"]:
        m_p = constants.proton_mass
    else:
        raise ValueError("Invalid species!!")

    e_max = vdf_energy.data[0, -1] + vdf.attrs["delta_energy_plus"][0, -1]
    v_max = constants.speed_of_light * np.sqrt(
        1
        - (e_max * constants.electron_volt / (m_p * constants.speed_of_light**2) + 1)
        ** -2
    )  # m/s
    speed_grid_cart = np.linspace(-v_max, v_max, endpoint=True)

    speed_grid = kwargs.get("vg", None)
    speed_grid_edges = kwargs.get("vg_edges", None)

    # initiate projected f
    if dim == "1d":
        f_g = np.zeros((n_t, len(speed_grid)))
        all_v = {"v": np.zeros((n_t, len(speed_grid)))}
    elif dim == "2d" and base == "cart":
        f_g = np.zeros((n_t, len(speed_grid), len(speed_grid)))
        all_v = {
            "vx": np.zeros((n_t, len(speed_grid))),
            "vy": np.zeros((n_t, len(speed_grid))),
        }
    elif dim == "3d" and base == "cart":
        f_g = np.zeros((n_t, len(speed_grid), len(speed_grid), len(speed_grid)))
        all_v = {
            "vx": np.zeros((n_t, len(speed_grid))),
            "vy": np.zeros((n_t, len(speed_grid))),
            "vz": np.zeros((n_t, len(speed_grid))),
        }
    else:
        raise ValueError("Invalid projection dimansion and base!!")

    for i_t in tqdm.tqdm(range(n_t)):  # display progress
        # 3d data matrix for time index
        f_3d = np.squeeze(vdf_data.data[i_t, ...])  # s^3/m^6
        f_3d = f_3d.astype(np.float64).copy()  # convert to C contiguous float64

        # Energies
        energy = vdf_energy.data[i_t, :]
        energy = energy.astype(np.float64)  # Convert to float64

        # Assign zeros to phase-space density values corresponding to energies below
        # the threshold energy or below the spacecraft potential if provided.
        thresh_e = np.max([lower_e_lim[i_t], sc_pot[i_t]])
        e_min_idx = np.where(energy - delta_energy_minu[i_t, :] > thresh_e)[0][0]
        f_3d[:e_min_idx] = 0.0

        # Correct energy shift due to spacecraft potential
        energy -= sc_pot[i_t]
        energy[energy < 0] = 0.0

        # Convert energy to velocity (relativistically correct)
        speed = constants.speed_of_light * np.sqrt(
            1
            - 1
            / (
                energy * constants.electron_volt / (m_p * constants.speed_of_light**2)
                + 1
            )
            ** 2
        )  # m/s

        # azimuthal angle
        phi = vdf_phi.data[i_t, :].astype(np.float64)  # in degrees
        phi = np.deg2rad(phi - 180.0)  # in radians

        # elevation angle
        theta = vdf_theta.data.astype(np.float64)  # in degrees
        theta = np.deg2rad(theta - 90.0)  # in radians

        # Set velocity projection grid.
        if speed_grid_edges is not None:
            speed_grid = speed_grid_edges[:-1] + 0.5 * np.diff(speed_grid_edges)
        elif speed_grid is not None:
            speed_grid = speed_grid
        else:
            if base == "pol":
                if dim == "1d":
                    speed_grid = np.hstack((-np.flip(speed), speed))
                elif dim in ["2d", "3d"]:
                    speed_grid = speed
                else:
                    raise ValueError("Invalid projection dimension!!")
            elif base == "cart":
                speed_grid = speed_grid_cart
            else:
                raise ValueError("Invalid base!!")

        options = dict(
            xyz=xyz_ts[i_t, ...],
            n_mc=n_mc,
            weight=weight,
            v_lim=v_lim,
            a_lim=a_lim,
            projection_dim=dim,
            projection_base=base,
            speed_grid_edges=speed_grid_edges,
        )

        tmpst = int_sph_dist(f_3d, speed, phi, theta, speed_grid, **options)

        if dim == "1d":
            f_g[i_t, :] = tmpst["f"]
            all_v["v"][i_t, :] = tmpst["v"]
        elif dim == "2d" and base == "cart":
            f_g[i_t, :, :] = tmpst["f"]
            all_v["vx"][i_t, :] = tmpst["vx"]
            all_v["vy"][i_t, :] = tmpst["vy"]
        elif dim == "3d" and base == "cart":
            f_g[i_t, ...] = tmpst["f"]
            all_v["vx"][i_t, :] = tmpst["vx"]
            all_v["vy"][i_t, :] = tmpst["vy"]
            all_v["vz"][i_t, :] = tmpst["vz"]
        else:
            raise NotImplementedError("Invalid dimension")

    return all_v, f_g