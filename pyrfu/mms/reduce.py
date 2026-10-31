#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import numbers

# Third party imports
import numpy as np
import tqdm
import xarray as xr
from scipy.constants import electron_mass, electron_volt, proton_mass, speed_of_light

# Local imports
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.int_sph_dist import _speed_bin_edges, int_sph_dist
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.time_clip import time_clip
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _energy2speed(energy, mass):
    r"""Convert energy (eV) to speed (m/s), relativistically correct."""
    gamma = 1 + electron_volt * energy / (mass * speed_of_light**2)
    return speed_of_light * np.sqrt(1 - 1 / gamma**2)


def _angle_widths(vdf, theta, tint):
    r"""Polar bands and azimuthal widths of the bins from the delta_theta_* and
    delta_phi_* attributes (degrees), or None if the distribution has none."""
    keys = ["delta_theta_minus", "delta_theta_plus"]
    keys += ["delta_phi_minus", "delta_phi_plus"]

    if not all(k in vdf.attrs for k in keys):
        return None

    # polar bands, as the (physical) elevation of the particle velocities
    colat_1 = theta - vdf.attrs["delta_theta_minus"]
    colat_2 = theta + vdf.attrs["delta_theta_plus"]
    elev_c = np.deg2rad((colat_1 + colat_2) / 2.0 - 90.0)
    d_theta = np.deg2rad(colat_2 - colat_1)
    solid = np.sin(np.deg2rad(colat_2 - 90.0)) - np.sin(np.deg2rad(colat_1 - 90.0))

    # azimuthal widths and shifts of the bin centres (time, phi)
    in_tint = np.isin(vdf.time.data, time_clip(vdf.time, tint).time.data)
    phi_m = np.asarray(vdf.attrs["delta_phi_minus"], dtype=np.float64)[in_tint]
    phi_p = np.asarray(vdf.attrs["delta_phi_plus"], dtype=np.float64)[in_tint]

    return {
        "theta": elev_c,
        "d_theta": d_theta,
        "scale": solid / (np.cos(elev_c) * d_theta),
        "phi_shift": np.deg2rad(phi_p - phi_m) / 2.0,
        "d_phi": np.deg2rad(phi_p + phi_m),
    }


def reduce(vdf, xyz, dim: str = "1d", base: str = "pol", **kwargs):
    r"""Reduces (integrates) 3D distribution to 1D (line) or 2D (plane).

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution function.
    xyz : xarray.DataArray or numpy.ndarray
        Transformation matrix from instrument frame to desired frame.
    dim : {"1d", "2d", "3d"}, Optional
        Dimension of the reduced distribution: a line ("1d", default), a plane
        ("2d") or a box ("3d").
    base : {"pol", "cart"}, Optional
        Base of the projection: polar "pol" (default, for "1d") or cartesian
        "cart" (for "2d" and "3d"). dim and base are case-insensitive.
    **kwargs
        Keyword arguments, among which:

        * sc_pot : xarray.DataArray, the spacecraft potential (V), 0 V by
          default.
        * lower_e_lim : float or xarray.DataArray, the threshold energy
          (eV), 0 eV by default. The channels below max(sc_pot,
          lower_e_lim) are set to zero; a time step without any channel
          above it is NaN.
        * verbose : bool, show a progress bar (default True).

    Returns
    -------
    out : xarray.DataArray
        Time series of reduced velocity distribution function, in SI units
        (``UNITS`` s/m^4 for 1d, s^2/m^5 for 2d), on velocities in km/s.

    Notes
    -----
    The energy widths of the channels (``delta_energy_minus`` and
    ``delta_energy_plus`` attributes of `vdf`) are optional. Without both,
    the speed bins are edged at the geometric means of neighbouring channel
    speeds (see `pyrfu.pyrf.int_sph_dist`); without ``delta_energy_plus``,
    the default Cartesian grid extends to the upper edge of these bins.

    The Monte-Carlo options (``n_mc``, ``weight``, ...) are those of
    `pyrfu.pyrf.int_sph_dist`. ``weight="log"`` weights the particles by
    1 + ln(f / min f); in irfu-matlab, and before pyrfu 2.5.2, it was the
    same as ``"lin"`` for phase-space densities in SI units.

    """

    # Make sure the projection dimension and base are correct
    base, dim = base.lower(), dim.lower()

    if base not in ["cart", "pol"]:
        raise ValueError(f'base must be "pol" or "cart", not {base!r}')

    if dim not in ["1d", "2d", "3d"]:
        raise ValueError(f'dim must be "1d", "2d" or "3d", not {dim!r}')

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

    # Upper energy bound of instrument bins. The energy widths define the speed
    # bins used for the Monte-Carlo integration; if either is missing, the
    # speed bin edges are inferred from the channel speeds in int_sph_dist.
    has_energy_widths = all(
        k in vdf.attrs for k in ["delta_energy_minus", "delta_energy_plus"]
    )

    if has_energy_widths:
        delta_energy_plus = xr.DataArray(
            vdf.attrs["delta_energy_plus"],
            coords=[vdf.time.data, vdf.idx0.data],
            dims=["time", "idx0"],
        )
        delta_energy_plus = time_clip(delta_energy_plus, tint)

    # Angular widths of the bins (degrees), e.g. the FEEPS skymaps. The
    # Monte-Carlo particles are drawn over the band between the edges of each
    # polar bin, and the data are scaled so that each bin has its exact solid
    # angle (int_sph_dist uses cos(theta) d_theta at the centre of the band).
    # Without them, the widths are the spacing of the grids (FPI, EIS).
    angle_widths = _angle_widths(vdf, vdf_theta.data, tint)

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

    # azimuthal angle of projection plane (phi is (time, phi) in burst mode)
    n_phi_grid = vdf_phi.shape[-1]
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
    elif isinstance(lower_e_lim, numbers.Real):
        # If a number (int or float), tile to VDF's timeline
        lower_e_lim = np.tile(float(lower_e_lim), n_t)
    else:
        raise TypeError("Invalid lower_e_lim!!")

    # Set particle specie mass from VDF's attribute "species"
    if vdf.species.lower() == "electrons":
        m_p = electron_mass
    elif vdf.species.lower() == "ions":
        m_p = proton_mass
    else:
        raise ValueError("Invalid species!!")

    # Maximum speed of the instrument (relativistically correct): the upper
    # energy edge of the top channel, or without delta_energy_plus the upper
    # edge of the speed bins inferred in int_sph_dist
    if "delta_energy_plus" in vdf.attrs:
        e_max = vdf_energy.data[0, -1] + vdf.attrs["delta_energy_plus"][0, -1]
        v_max = _energy2speed(e_max, m_p)  # m/s
    else:
        energy = vdf_energy.data[0, :].astype(np.float64)
        v_max = _speed_bin_edges(_energy2speed(energy, m_p))[-1]  # m/s

    # initiate projected f
    if velocity_grid is not None:
        n_vg = len(velocity_grid)
    elif velocity_grid is None and velocity_grid_edges is not None:
        n_vg = len(velocity_grid_edges) - 1
        velocity_grid = velocity_grid_edges[:-1] + 0.5 * np.diff(velocity_grid_edges)
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

    # Time steps computed (those without any channel above the threshold
    # energy are NaN)
    computed = np.zeros(n_t, dtype=bool)

    for i_t in tqdm.tqdm(range(n_t), ncols=60, disable=not kwargs.get("verbose", True)):
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
        above = np.where(energy - delta_energy_minu[i_t, :] > thresh_e)[0]

        if len(above) == 0:
            f_g[i_t, ...] = np.nan
            continue

        f_3d[: above[0]] = 0.0

        # Correct energy shift due to spacecraft potential
        energy -= sc_pot[i_t]
        energy[energy < 0] = 0.0

        # Convert energy to velocity (relativistically correct)
        velocity = _energy2speed(energy, m_p)  # m/s

        # Speed widths of the instrument bins from the energy bounds of the
        # channels (corrected for the spacecraft potential)
        if has_energy_widths:
            e_low = vdf_energy.data[i_t, :] - delta_energy_minu.data[i_t, :]
            e_upp = vdf_energy.data[i_t, :] + delta_energy_plus.data[i_t, :]
            e_low = np.clip(e_low.astype(np.float64) - sc_pot[i_t], 0.0, None)
            e_upp = np.clip(e_upp.astype(np.float64) - sc_pot[i_t], 0.0, None)
            speed_widths = {
                "d_v_m": velocity - _energy2speed(e_low, m_p),
                "d_v_p": _energy2speed(e_upp, m_p) - velocity,
            }
        else:
            speed_widths = {}

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

        angle_options = {}

        if angle_widths is not None:
            theta = angle_widths["theta"]
            f_3d *= angle_widths["scale"][np.newaxis, np.newaxis, :]
            phi += angle_widths["phi_shift"][i_t]
            angle_options = {
                "d_theta": angle_widths["d_theta"],
                "d_phi": angle_widths["d_phi"][i_t],
            }

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
            **speed_widths,
            **angle_options,
        }

        tmpst = int_sph_dist(
            f_3d, velocity, phi, theta, velocity_grid, phi_grid, **options
        )

        f_g[i_t, ...] = tmpst["f"]

        for i in range(ndim):
            all_v[f"v{chr(120 + i)}"][i_t, :] = tmpst[f"v{chr(120 + i)}"] / 1e3  # km/s

        computed[i_t] = True

    if not np.any(computed):
        raise ValueError(
            "no energy channel above the spacecraft potential and lower_e_lim at "
            "any time"
        )

    # Build output as a time series with dimensions:
    #   - (time x vx) for 1D reduced distribution
    #   - (time x vx x vy) for 2D reduced distribution
    i_grid = np.argmax(computed)  # the grid of the first computed step
    coords = [
        vdf_time.data,
        *[all_v[f"v{chr(120 + i)}"][i_grid, :] for i in range(ndim)],
    ]
    dims = ["time", *[f"v{chr(120 + i)}" for i in range(ndim)]]
    # SI units, s^3/m^6 integrated over 3 - ndim velocity components in m/s;
    # velocities in km/s (none)
    out = xr.DataArray(
        f_g,
        coords=coords,
        dims=dims,
        attrs={"UNITS": {1: "s/m^4", 2: "s^2/m^5", 3: "s^3/m^6"}[ndim]},
    )

    for v_dim in dims[1:]:
        out[v_dim].attrs["UNITS"] = "km/s"

    return out
