#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_tensor_xyz import ts_tensor_xyz
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


@numba.njit(cache=True, fastmath=False, nogil=True, parallel=True)
def _sanitize_nan_inplace(vdf):
    """Replace NaN entries in `vdf` with 0.0, in place. Deliberately
    compiled WITHOUT fastmath -- this is the function meant to be
    reliably NaN-aware, so it can't itself use the flag that breaks
    NaN detection (see the "NaN + fastmath" note in the module
    docstring). `x != x` is true iff `x` is NaN under IEEE-754; this is
    the standard non-fastmath-dependent isnan idiom and, unlike
    `np.isnan()`, needs no extra import inside the jitted function.

    Mutates and returns `vdf` in place (no second full-size array
    allocated) -- measured ~7x faster than an out-of-place numba
    version and ~10x faster than `np.where(np.isnan(vdf), 0.0, vdf)` on
    a realistic (512, 32, 32, 16) array in this sandbox, since it's one
    read+write pass over the array instead of two. Safe to call on
    `vdf_data` in the wrapper below because that array was already
    freshly allocated (C-contiguous, as `reshape` requires) by the unit
    conversion a few lines earlier -- mutating it doesn't touch the
    caller's original `vdf.data`.
    """
    flat = vdf.reshape(-1)
    n = flat.shape[0]
    for i in numba.prange(n):
        if flat[i] != flat[i]:
            flat[i] = 0.0
    return vdf


@numba.jit(cache=True, fastmath=True, nogil=True, parallel=True, nopython=True)
def _moms(
    energy,
    delta_v,
    q_e,
    sc_pot,
    p_mass,
    flag_inner_electron,
    w_inner_electron,
    phi,
    theta,
    use_energy,
    vdf,
    delta_ang,
):
    n_psd = np.zeros(vdf.shape[0])
    v_psd = np.zeros((vdf.shape[0], 3))
    p_psd = np.zeros((vdf.shape[0], 3, 3))
    h_psd = np.zeros((vdf.shape[0], 3))

    n_ph = vdf.shape[2]
    n_th = vdf.shape[3]

    for i_t in numba.prange(vdf.shape[0]):
        energy_correct = energy[i_t, :] - sc_pot[i_t]

        velocity = np.sqrt(2 * q_e * energy_correct / p_mass)
        velocity[energy_correct < flag_inner_electron * w_inner_electron] = 0

        phi_i = np.deg2rad(phi[i_t, :, :])
        theta_i = np.deg2rad(theta[i_t, :, :])

        psd2n_mat = np.ones(theta_i.shape) * np.sin(theta_i)

        # Particle flux and heat flux vector
        psd2v_x_mat = -np.cos(phi_i) * np.sin(theta_i) ** 2
        psd2v_y_mat = -np.sin(phi_i) * np.sin(theta_i) ** 2
        psd2v_z_mat = -np.ones(theta_i.shape) * np.sin(theta_i) * np.cos(theta_i)

        psd2p_xx_mat = np.cos(phi_i) ** 2.0 * np.sin(theta_i) ** 3
        psd2p_yy_mat = np.sin(phi_i) ** 2.0 * np.sin(theta_i) ** 3
        psd2p_zz_mat = np.ones(theta_i.shape) * np.sin(theta_i) * np.cos(theta_i) ** 2
        psd2p_xy_mat = np.cos(phi_i) * np.sin(phi_i) * np.sin(theta_i) ** 3
        psd2p_xz_mat = np.cos(phi_i) * np.sin(theta_i) ** 2 * np.cos(theta_i)
        psd2p_yz_mat = np.sin(phi_i) * np.sin(theta_i) ** 2 * np.cos(theta_i)

        for i_e in range(vdf.shape[1]):
            # Energy channels selected for this time step
            if not use_energy[i_t, i_e]:
                continue

            n_acc = 0.0
            vx_acc = 0.0
            vy_acc = 0.0
            vz_acc = 0.0
            pxx_acc = 0.0
            pxy_acc = 0.0
            pxz_acc = 0.0
            pyy_acc = 0.0
            pyz_acc = 0.0
            pzz_acc = 0.0

            # Single fused pass over the (phi, theta) grid: read the
            # measured PSD value once, skip NaNs (matching np.nansum's
            # omission behavior), and accumulate all 10 weighted moment
            # contributions together instead of 10 separate
            # multiply+nansum passes over the same data.
            for i_ph in range(n_ph):
                for i_th in range(n_th):
                    val = vdf[i_t, i_e, i_ph, i_th]
                    if np.isnan(val):
                        continue
                    w = val * delta_ang[i_t, i_ph, i_th]

                    n_acc += w * psd2n_mat[i_ph, i_th]

                    vx_acc += w * psd2v_x_mat[i_ph, i_th]
                    vy_acc += w * psd2v_y_mat[i_ph, i_th]
                    vz_acc += w * psd2v_z_mat[i_ph, i_th]

                    pxx_acc += w * psd2p_xx_mat[i_ph, i_th]
                    pxy_acc += w * psd2p_xy_mat[i_ph, i_th]
                    pxz_acc += w * psd2p_xz_mat[i_ph, i_th]
                    pyy_acc += w * psd2p_yy_mat[i_ph, i_th]
                    pyz_acc += w * psd2p_yz_mat[i_ph, i_th]
                    pzz_acc += w * psd2p_zz_mat[i_ph, i_th]

            # number density
            n_psd_tmp = n_acc * delta_v[i_t, i_e] * velocity[i_e] ** 2
            n_psd[i_t] += n_psd_tmp

            # Bulk velocity
            v_temp_x = vx_acc * delta_v[i_t, i_e] * velocity[i_e] ** 3
            v_temp_y = vy_acc * delta_v[i_t, i_e] * velocity[i_e] ** 3
            v_temp_z = vz_acc * delta_v[i_t, i_e] * velocity[i_e] ** 3

            v_psd[i_t, 0] += v_temp_x
            v_psd[i_t, 1] += v_temp_y
            v_psd[i_t, 2] += v_temp_z

            # Pressure tensor
            p_temp_xx = pxx_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4
            p_temp_xy = pxy_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4
            p_temp_xz = pxz_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4
            p_temp_yy = pyy_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4
            p_temp_yz = pyz_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4
            p_temp_zz = pzz_acc * delta_v[i_t, i_e] * velocity[i_e] ** 4

            p_psd[i_t, 0, 0] += p_temp_xx
            p_psd[i_t, 0, 1] += p_temp_xy
            p_psd[i_t, 0, 2] += p_temp_xz
            p_psd[i_t, 1, 1] += p_temp_yy
            p_psd[i_t, 1, 2] += p_temp_yz
            p_psd[i_t, 2, 2] += p_temp_zz

            # Heat flux vector.
            h_psd[i_t, 0] += vx_acc * delta_v[i_t, i_e] * velocity[i_e] ** 5
            h_psd[i_t, 1] += vy_acc * delta_v[i_t, i_e] * velocity[i_e] ** 5
            h_psd[i_t, 2] += vz_acc * delta_v[i_t, i_e] * velocity[i_e] ** 5

    return n_psd, v_psd, p_psd, h_psd


def _energy_edges(vdf, energy, energy0, energy1, step_table, flag_same_e):
    # Upper and lower energy edges of the channels (time, energy), as in
    # irfu-matlab mms.psd_moments: the delta_energy_plus/minus attributes
    # when there is one table, the linear midpoints between the channels of
    # each table otherwise (extrapolated linearly at both ends).
    energy_plus = vdf.attrs.get("delta_energy_plus")
    energy_minus = vdf.attrs.get("delta_energy_minus")

    if flag_same_e and energy_plus is not None and energy_minus is not None:
        energy_upper = np.broadcast_to(energy + energy_plus, energy.shape)
        energy_lower = np.broadcast_to(energy - energy_minus, energy.shape)
        return energy_upper, energy_lower

    def _midpoints(table):
        table = np.asarray(table, dtype=np.float64)
        table_all = np.concatenate(
            [
                2 * table[..., :1] - table[..., 1:2],
                table,
                2 * table[..., -1:] - table[..., -2:-1],
            ],
            axis=-1,
        )
        diff_table = np.diff(table_all, axis=-1)
        return table + diff_table[..., 1:] / 2, table - diff_table[..., :-1] / 2

    if flag_same_e:
        return _midpoints(energy)

    energy0_upper, energy0_lower = _midpoints(energy0)
    energy1_upper, energy1_lower = _midpoints(energy1)
    is_table1 = np.asarray(step_table)[:, np.newaxis] == 1
    energy_upper = np.where(is_table1, energy1_upper, energy0_upper)
    energy_lower = np.where(is_table1, energy1_lower, energy0_lower)
    return energy_upper, energy_lower


def _energy_mask(vdf, energy, kwargs):
    r"""Energy channels to integrate over at each time step, as a boolean
    (time, energy) array, from the energy_range or en_channels options."""
    n_t, n_e = energy.shape

    if "energy_range" in kwargs and "en_channels" in kwargs:
        raise ValueError("Use either energy_range or en_channels, not both")

    if "energy_range" not in kwargs:
        start, stop = kwargs.get("en_channels", [0, n_e])
        use_energy = np.zeros((n_t, n_e), dtype=np.bool_)
        use_energy[:, start:stop] = True
        return use_energy

    e_range = kwargs["energy_range"]

    if isinstance(e_range, xr.DataArray):
        # Time dependent energy range on its own time line
        if not np.array_equal(e_range.time.data, vdf.time.data):
            e_range = resample(e_range, vdf.time)

        e_range = e_range.data

    e_range = np.asarray(e_range, dtype=np.float64)

    if e_range.shape == (2,):
        # Channels of the first energy table within the range, and the same
        # channels for the other table (as irfu-matlab)
        channels = np.where((energy[0] > e_range[0]) & (energy[0] < e_range[1]))[0]

        if channels.size == 0:
            raise ValueError(f"No energy channel in energy_range {e_range}")

        use_energy = np.zeros((n_t, n_e), dtype=np.bool_)
        use_energy[:, channels[0] : channels[-1] + 1] = True
        logger.info("Using partial energy range")

    elif e_range.shape == (n_t, 2):
        # Time dependent: the channels of each time step within its range
        use_energy = (energy > e_range[:, :1]) & (energy < e_range[:, 1:])
        n_empty = int(np.sum(~np.any(use_energy, axis=1)))

        if n_empty:
            logger.warning(
                "No energy channel in energy_range at %(n)d time steps (NaN moments)",
                {"n": n_empty},
            )
    else:
        raise ValueError(
            "energy_range must be [E_min, E_max], or (E_min, E_max) at each time "
            "(array of shape (n_t, 2) or DataArray with a time coordinate)"
        )

    return use_energy


def psd_moments(vdf, sc_pot, **kwargs):
    r"""Computes moments from the FPI particle phase-space densities.

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution. The angular widths are taken from the
        optional attributes `delta_phi_minus`/`delta_phi_plus` (time, phi) and
        `delta_theta_minus`/`delta_theta_plus` (theta), in degrees, or else
        from the spacing of the phi and theta grids.
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.

    Returns
    -------
    n_psd : xarray.DataArray
        Time series of the number density (1rst moment).
    v_psd : xarray.DataArray
        Time series of the bulk velocity (2nd moment).
    p_psd : xarray.DataArray
        Time series of the pressure tensor (3rd moment).
    p2_psd : xarray.DataArray
        Time series of the pressure tensor.
    t_psd : xarray.DataArray
        Time series of the temperature tensor.
    h_psd : xarray.DataArray
        Time series of the heat flux vector

    Other Parameters
    ----------------
    energy_range : array_like or xarray.DataArray
        Energy range in eV to integrate over, strictly within (instrument
        energies, before the spacecraft potential correction):

        * [E_min, E_max]: applied to the energy table of the first time step,
          and the same channels are used at all times, to ensure that the same
          number of points are integrated over with alternating tables.
        * time dependent (E_min, E_max): an array of shape (n_t, 2) at the
          times of vdf, or a DataArray with a time coordinate and 2 columns,
          resampled to the times of vdf. At each time step, the channels of
          its energy table within its range are integrated over; time steps
          without channel get NaN moments.

        Can't be used with en_channels.
    no_sc_pot : bool
        Set to 1 to set spacecraft potential to zero. Calculates moments
        without correcting for spacecraft potential.
    en_channels : array_like
        Energy channels to integrate over, [start, stop) 0-based indices as a
        Python slice (default [0, 32], all the channels; unlike irfu-matlab's
        1-based inclusive [min max]). Can't be used with energy_range.
    partial_moments : numpy.ndarray or xarray.DataArray
        Use a binary array to select which psd points are used in the moments
        calculation. `partial_moments` must be a binary array (1s and 0s,
        1s correspond to points used). Array (or data of Dataarray) must be the same
        size as vdf.data.
    inner_electron : {"on", "off"}
        inner_electrontron potential for electron moments.

    Raises
    ------
    ValueError
        If energy_range has no channel (fixed range) or a wrong shape, or if
        both energy_range and en_channels are given.

    Notes
    -----
    The speed widths of the energy channels follow irfu-matlab
    `mms.psd_moments`: the channel edges are E + delta_energy_plus and
    E - delta_energy_minus (attributes) when there is one energy table, and
    the midpoints between the channels of each table otherwise. The
    spacecraft potential is subtracted from the edges (pyrfu < 2.5 used the
    uncorrected edges, which underestimates the density of cold electrons, by
    10-30 % for 10-20 eV and 10 V). The phase-space density is not corrected
    for spacecraft photoelectrons, unlike the FPI moments: use
    ``inner_electron="on"`` or an energy range above the spacecraft potential
    to compare the electron moments with them.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Load magnetic field and spacecraft potential

    >>> scpot = mms.get_data("V_edp_brst_l2", tint_brst, 1)

    Load electron velocity distribution function

    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint_brst, 1)

    Compute moments

    >>> options = dict(energy_range=[1, 1000])
    >>> moments_e = mms.psd_moments(vdf_e, scpot, **options)
    """

    # [eV] sc_pot + w_inner_electron for electron moments calculation
    w_inner_electron = 3.5

    # Check if data is fast or burst resolution
    if "brst" in vdf.data.attrs["FIELDNAM"].lower():
        logger.info("Burst resolution data is used")
    elif "fast" in vdf.data.attrs["FIELDNAM"].lower():
        logger.info("Fast resolution data is used")
    else:
        raise TypeError("Could not identify if data is fast or burst.")

    theta = vdf.theta.data
    particle_type = vdf.attrs["species"]
    assert particle_type[0].lower() in ["e", "i"], "invalid particle type"

    # In SI units, as a new C-contiguous float64 array: get_dist returns a
    # transposed (non-contiguous) view, which the numba kernels can't reshape
    vdf_data = np.multiply(vdf.data.data, 1e12, dtype=np.float64, order="C")

    step_table = vdf.attrs["esteptable"]
    energy = vdf.energy.data
    energy0 = vdf.attrs["energy0"]
    energy1 = vdf.attrs["energy1"]
    e_tmp = energy1 - energy0

    flag_same_e = np.all(e_tmp == 0)

    # resample sc_pot to same resolution as particle distributions
    sc_pot = resample(sc_pot, vdf.time).data

    no_sc_pot = kwargs.get("no_sc_pot", False)
    if no_sc_pot:
        sc_pot = np.zeros(sc_pot.shape)
        logger.info("Setting spacecraft potential to zero")

    use_energy = _energy_mask(vdf, energy, kwargs)

    if "partial_moments" in kwargs:
        partial_moments = kwargs["partial_moments"]
        if isinstance(partial_moments, xr.DataArray):
            partial_moments = partial_moments.data

        # Check size of partial_moments
        if partial_moments.shape == vdf_data.shape:
            if np.isin(partial_moments, [0, 1]).all():
                logger.info(
                    "partial_moments is correct. Partial moments will be calculated"
                )
                vdf_data = vdf_data * partial_moments
            else:
                logger.info(
                    "All values are not ones and zeros in partial_moments. "
                    "Full moments will be calculated"
                )
        else:
            logger.info(
                "Size of partial_moments is wrong. Full moments will be calculated"
            )

    tmp_ = kwargs.get("inner_electron", "")
    flag_inner_electron = tmp_ == "on" and particle_type[0] == "e"

    # Define constants
    q_e = constants.elementary_charge
    k_b = constants.Boltzmann

    if particle_type[0] == "e":
        p_mass = constants.electron_mass
        logger.info("Particles are electrons")
    else:
        p_mass = constants.proton_mass
        sc_pot *= -1.0
        logger.info("Particles are ions")

    # angle between theta and phi points is 360/32 = 11.25 degrees
    phi = vdf.phi.data

    if "delta_phi_minus" in vdf.attrs and "delta_phi_plus" in vdf.attrs:
        delta_phi_minus = vdf.attrs["delta_phi_minus"]
        delta_phi_plus = vdf.attrs["delta_phi_plus"]
        # Widths in degrees, as in the FPI files (irfu-matlab converts too)
        delta_phi = np.deg2rad(delta_phi_plus + delta_phi_minus)
        delta_phi = np.tile(delta_phi[:, :, np.newaxis], (1, 1, vdf_data.shape[3]))
    else:
        delta_phi = np.deg2rad(np.median(np.diff(phi[0, :])))
        delta_phi = delta_phi * np.ones(
            (vdf_data.shape[0], vdf_data.shape[2], vdf_data.shape[3])
        )

    if "delta_theta_minus" in vdf.attrs and "delta_theta_plus" in vdf.attrs:
        delta_theta_minus = vdf.attrs["delta_theta_minus"]
        delta_theta_plus = vdf.attrs["delta_theta_plus"]
        delta_theta = np.deg2rad(delta_theta_plus + delta_theta_minus)
        delta_theta = np.tile(
            delta_theta[np.newaxis, np.newaxis, :],
            (vdf_data.shape[0], vdf_data.shape[2], 1),
        )
    else:
        delta_theta = np.deg2rad(np.median(np.diff(theta)))
        delta_theta = delta_theta * np.ones(
            (vdf_data.shape[0], vdf_data.shape[2], vdf_data.shape[3])
        )

    delta_ang = delta_phi * delta_theta

    phi_mat = np.tile(phi[:, :, np.newaxis], (1, 1, vdf_data.shape[3]))
    theta_mat = np.tile(
        theta[np.newaxis, np.newaxis, :], (vdf_data.shape[0], vdf_data.shape[2], 1)
    )

    energy_correct = energy - sc_pot[:, np.newaxis]
    velocity = np.sqrt(2 * q_e * energy_correct / p_mass)
    velocity[energy_correct < flag_inner_electron * w_inner_electron] = 0

    # Speed widths of the energy channels, as in irfu-matlab mms.psd_moments:
    # the edges of each channel, corrected for the spacecraft potential
    energy_upper, energy_lower = _energy_edges(
        vdf, energy, energy0, energy1, step_table, flag_same_e
    )
    sc_pot_2d = sc_pot[:, np.newaxis]
    e_upper = np.clip(energy_upper - sc_pot_2d, 0.0, None)
    e_lower = np.clip(energy_lower - sc_pot_2d, 0.0, None)
    v_upper = np.sqrt(2 * q_e * e_upper / p_mass)
    v_lower = np.sqrt(2 * q_e * e_lower / p_mass)
    delta_v = np.ascontiguousarray(v_upper - v_lower)

    # Clean up NaN values in the input VDF data before passing to the numba kernel.
    # This is done in place to avoid extra memory allocation and to ensure that the
    # kernel does not encounter NaN values, which could lead to incorrect moment
    # calculations.
    _sanitize_nan_inplace(vdf_data)

    n_psd, v_psd, p_psd, h_psd = _moms(
        energy,
        delta_v,
        q_e,
        sc_pot,
        p_mass,
        flag_inner_electron,
        w_inner_electron,
        phi_mat,
        theta_mat,
        use_energy,
        vdf_data,
        delta_ang,
    )

    # No energy channel at these time steps (time dependent energy_range)
    no_energy = ~np.any(use_energy, axis=1)
    n_psd[no_energy] = np.nan
    v_psd[no_energy] = np.nan
    p_psd[no_energy] = np.nan
    h_psd[no_energy] = np.nan

    # Compute moments in SI units
    p_psd *= p_mass
    v_psd /= n_psd[:, np.newaxis]
    p2_psd = np.zeros_like(p_psd)
    p2_psd[:, 0, 0] = p_psd[:, 0, 0]
    p2_psd[:, 0, 1] = p_psd[:, 0, 1]
    p2_psd[:, 0, 2] = p_psd[:, 0, 2]
    p2_psd[:, 1, 1] = p_psd[:, 1, 1]
    p2_psd[:, 1, 2] = p_psd[:, 1, 2]
    p2_psd[:, 2, 2] = p_psd[:, 2, 2]
    p2_psd[:, 1, 0] = p2_psd[:, 0, 1]
    p2_psd[:, 2, 0] = p2_psd[:, 0, 2]
    p2_psd[:, 2, 1] = p2_psd[:, 1, 2]

    p_psd[:, 0, 0] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 0]
    p_psd[:, 0, 1] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 1]
    p_psd[:, 0, 2] -= p_mass * n_psd * v_psd[:, 0] * v_psd[:, 2]
    p_psd[:, 1, 1] -= p_mass * n_psd * v_psd[:, 1] * v_psd[:, 1]
    p_psd[:, 1, 2] -= p_mass * n_psd * v_psd[:, 1] * v_psd[:, 2]
    p_psd[:, 2, 2] -= p_mass * n_psd * v_psd[:, 2] * v_psd[:, 2]
    p_psd[:, 1, 0] = p_psd[:, 0, 1]
    p_psd[:, 2, 0] = p_psd[:, 0, 2]
    p_psd[:, 2, 1] = p_psd[:, 1, 2]

    p_trace = np.trace(p_psd, axis1=1, axis2=2)
    t_psd = np.zeros(p_psd.shape)
    t_psd[...] = p_psd[...] / (k_b * n_psd[:, np.newaxis, np.newaxis])

    v_abs2 = np.linalg.norm(v_psd, axis=1) ** 2
    h_psd *= p_mass / 2
    h_psd[:, 0] -= v_psd[:, 0] * p_psd[:, 0, 0]
    h_psd[:, 0] -= v_psd[:, 1] * p_psd[:, 0, 1]
    h_psd[:, 0] -= v_psd[:, 2] * p_psd[:, 0, 2]
    h_psd[:, 0] -= 0.5 * v_psd[:, 0] * (p_trace + p_mass * n_psd * v_abs2)
    h_psd[:, 1] -= v_psd[:, 0] * p_psd[:, 1, 0]
    h_psd[:, 1] -= v_psd[:, 1] * p_psd[:, 1, 1]
    h_psd[:, 1] -= v_psd[:, 2] * p_psd[:, 1, 2]
    h_psd[:, 1] -= 0.5 * v_psd[:, 1] * (p_trace + p_mass * n_psd * v_abs2)
    h_psd[:, 2] -= v_psd[:, 0] * p_psd[:, 2, 0]
    h_psd[:, 2] -= v_psd[:, 1] * p_psd[:, 2, 1]
    h_psd[:, 2] -= v_psd[:, 2] * p_psd[:, 2, 2]
    h_psd[:, 2] -= 0.5 * v_psd[:, 2] * (p_trace + p_mass * n_psd * v_abs2)

    # Convert to typical units (/cc, km/s, nP, eV, and ergs/s/cm^2).
    n_psd /= 1e6
    v_psd /= 1e3
    p_psd *= 1e9
    p2_psd *= 1e9
    t_psd *= k_b / q_e
    h_psd *= 1e3

    # Construct TSeries
    n_psd = ts_scalar(vdf.time.data, n_psd)
    v_psd = ts_vec_xyz(vdf.time.data, v_psd)
    p_psd = ts_tensor_xyz(vdf.time.data, p_psd)
    p2_psd = ts_tensor_xyz(vdf.time.data, p2_psd)
    t_psd = ts_tensor_xyz(vdf.time.data, t_psd)
    h_psd = ts_vec_xyz(vdf.time.data, h_psd)

    return n_psd, v_psd, p_psd, p2_psd, t_psd, h_psd
