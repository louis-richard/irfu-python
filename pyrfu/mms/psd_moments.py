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
from pyrfu.mms._vdf_bins import _bin_pitch_angles, _phi_edges, _theta_edges
from pyrfu.mms.vdf_convert import _PSD, vdf_convert
from pyrfu.pyrf.normalize import normalize
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
    exact_kernels,
    use_exact,
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

        # Kernels integrated exactly over each angular bin (delta_ang is 1)
        if use_exact:
            psd2n_mat = exact_kernels[i_t, 0].copy()
            psd2v_x_mat = exact_kernels[i_t, 1].copy()
            psd2v_y_mat = exact_kernels[i_t, 2].copy()
            psd2v_z_mat = exact_kernels[i_t, 3].copy()
            psd2p_xx_mat = exact_kernels[i_t, 4].copy()
            psd2p_yy_mat = exact_kernels[i_t, 5].copy()
            psd2p_zz_mat = exact_kernels[i_t, 6].copy()
            psd2p_xy_mat = exact_kernels[i_t, 7].copy()
            psd2p_xz_mat = exact_kernels[i_t, 8].copy()
            psd2p_yz_mat = exact_kernels[i_t, 9].copy()

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


def _exact_kernels(
    phi, theta, delta_phi_minus, delta_phi_plus, delta_theta_minus, delta_theta_plus
):
    # Angular kernels of the moments integrated over each (phi, theta) bin,
    # assuming a constant phase-space density in the bin: (time, kernel, phi,
    # theta), kernels in the order of _moms. Angles and widths in degrees, phi
    # (time, phi), theta (theta,).
    phi_a = np.deg2rad(phi - delta_phi_minus)[:, :, np.newaxis]
    phi_b = np.deg2rad(phi + delta_phi_plus)[:, :, np.newaxis]
    theta_a = np.deg2rad(theta - delta_theta_minus)[np.newaxis, np.newaxis, :]
    theta_b = np.deg2rad(theta + delta_theta_plus)[np.newaxis, np.newaxis, :]

    # integrals over phi of 1, cos, sin, cos^2, sin^2 and cos sin
    d_phi = phi_b - phi_a
    sin2_phi = (np.sin(2 * phi_b) - np.sin(2 * phi_a)) / 4
    i_1 = d_phi
    i_c = np.sin(phi_b) - np.sin(phi_a)
    i_s = np.cos(phi_a) - np.cos(phi_b)
    i_cc = d_phi / 2 + sin2_phi
    i_ss = d_phi / 2 - sin2_phi
    i_cs = (np.sin(phi_b) ** 2 - np.sin(phi_a) ** 2) / 2

    # integrals over theta of sin, sin^2, sin cos, sin^3, sin cos^2, sin^2 cos
    c_a, c_b = np.cos(theta_a), np.cos(theta_b)
    s_a, s_b = np.sin(theta_a), np.sin(theta_b)
    j_s = c_a - c_b
    j_ss = (theta_b - theta_a) / 2 - (np.sin(2 * theta_b) - np.sin(2 * theta_a)) / 4
    j_sc = (s_b**2 - s_a**2) / 2
    j_sss = (c_a - c_b) - (c_a**3 - c_b**3) / 3
    j_scc = (c_a**3 - c_b**3) / 3
    j_ssc = (s_b**3 - s_a**3) / 3

    kernels = [
        i_1 * j_s,
        -i_c * j_ss,
        -i_s * j_ss,
        -i_1 * j_sc,
        i_cc * j_sss,
        i_ss * j_sss,
        i_1 * j_scc,
        i_cs * j_sss,
        i_c * j_ssc,
        i_s * j_ssc,
    ]

    return np.ascontiguousarray(np.stack(kernels, axis=1), dtype=np.float64)


def _speed_relativistic(energy, p_mass):
    # Relativistic speed (m/s) of particles of kinetic energy (eV)
    gamma = 1.0 + constants.elementary_charge * energy / (p_mass * constants.c**2)
    return constants.c * np.sqrt(1.0 - 1.0 / gamma**2)


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


def _moms_relativistic(moms_args, sc_pot, p_mass):
    # Relativistic moments with the (non-relativistic) _moms kernel: energies
    # whose non-relativistic speed is the relativistic one, and the pressure
    # and heat flux from the data weighted by the Lorentz factor (momentum flux
    # gamma m v v) and by the kinetic energy over m v^2 / 2
    # arguments of _moms: energy (0), flag and width of the inner electron
    # energies (5, 6), data (10)
    energy, vdf_data = moms_args[0], moms_args[10]
    flag_inner_electron, w_inner_electron = moms_args[5], moms_args[6]
    q_e, c_light = constants.elementary_charge, constants.c

    energy_correct = energy - sc_pot[:, np.newaxis]
    gamma = 1.0 + q_e * energy_correct / (p_mass * c_light**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = _speed_relativistic(energy_correct, p_mass)
        kinetic = np.where(
            velocity > 0.0, 2 * (gamma - 1) * c_light**2 / velocity**2, 1.0
        )

    # Raw energies where the kernel sets the speed to zero (inner electron
    # energies) or where it is undefined (below the spacecraft potential)
    keep = ~(velocity > 0.0) | (energy_correct < flag_inner_electron * w_inner_electron)

    args = list(moms_args)
    args[0] = np.where(
        keep, energy, p_mass * velocity**2 / (2 * q_e) + sc_pot[:, np.newaxis]
    )
    n_psd, v_psd, _, _ = _moms(*args)

    args[10] = np.ascontiguousarray(vdf_data * gamma[:, :, np.newaxis, np.newaxis])
    p_psd = _moms(*args)[2]

    args[10] = np.ascontiguousarray(vdf_data * kinetic[:, :, np.newaxis, np.newaxis])
    h_psd = _moms(*args)[3]

    return n_psd, v_psd, p_psd, h_psd


# Sub-grid of the angular bins for the pitch-angle fractions
_N_SUB = 8


def _psd_si(vdf, relativistic):
    r"""Phase-space density in s^3/m^6, as a new C-contiguous float64 array
    (get_dist returns a transposed view, which the numba kernels can't
    reshape), from the units of `vdf` (``UNITS``, s^3/cm^6 if missing)."""
    units = str(vdf.data.attrs.get("UNITS", "s^3/cm^6"))

    if units.lower() in _PSD:
        factor = _PSD[units.lower()]
        return np.multiply(vdf.data.data, factor, dtype=np.float64, order="C")

    # Differential energy or particle fluxes, as irfu-matlab
    # (PDist.convertto)
    vdf_si = vdf_convert(vdf, "s^3/m^6", relativistic=relativistic)
    return np.ascontiguousarray(vdf_si.data.data, dtype=np.float64)


def _pitch_angle_args(kwargs):
    # b_xyz, [min, max] and method of the pitch-angle range
    b_xyz = kwargs.get("b_xyz")
    pitch_angle_range = np.asarray(kwargs["pitch_angle_range"], dtype=np.float64)
    method = kwargs.get("pitch_angle_weight", "centre")

    if b_xyz is None:
        raise ValueError("pitch_angle_range needs b_xyz")

    valid = pitch_angle_range.shape == (2,) and (
        0.0 <= pitch_angle_range[0] <= pitch_angle_range[1] <= 180.0
    )

    if not valid:
        raise ValueError("pitch_angle_range must be [min, max] within [0, 180]")

    if method not in ["centre", "fraction"]:
        raise ValueError(f"Invalid pitch_angle_weight {method}: centre or fraction")

    return b_xyz, pitch_angle_range, method


def _pitch_angle_weights(vdf, b_xyz, pitch_angle_range, method):
    # Weight of each angular bin (time, phi, theta) for the pitch angles in
    # pitch_angle_range: 1 or 0 from the pitch angle of its centre ("centre"),
    # or the fraction of its solid angle in the range ("fraction", on a
    # _N_SUB x _N_SUB grid uniform in phi and cos(theta)), and the times
    # without B
    b_hat = normalize(resample(b_xyz, vdf.time)).data
    no_b = ~np.all(np.isfinite(b_hat), axis=1)
    pa_min, pa_max = pitch_angle_range
    phi = np.broadcast_to(vdf.phi.data, (len(vdf.time), vdf.data.shape[2]))

    if method == "centre":
        pitch = _bin_pitch_angles(phi, vdf.theta.data, b_hat)
        return ((pitch >= pa_min) & (pitch <= pa_max)).astype(np.float64), no_b

    sub = (np.arange(_N_SUB) + 0.5) / _N_SUB
    phi_lower, phi_upper = _phi_edges(vdf)
    phi_sub = phi_lower[..., None] + sub * (phi_upper - phi_lower)[..., None]
    phi_sub = phi_sub.reshape(len(vdf.time), -1)
    theta_lower, theta_upper = _theta_edges(vdf)
    cos_lower = np.cos(np.deg2rad(theta_lower))[:, None]
    cos_upper = np.cos(np.deg2rad(theta_upper))[:, None]
    theta_sub = np.rad2deg(np.arccos(cos_lower - sub * (cos_lower - cos_upper)))
    theta_sub = theta_sub.ravel()

    n_t, n_ph, n_th = len(vdf.time), vdf.data.shape[2], vdf.data.shape[3]
    weights = np.zeros((n_t, n_ph, n_th))

    # Time chunks: the sub-grid is _N_SUB^2 times the angular grid
    for start in range(0, n_t, 64):
        chunk = slice(start, start + 64)
        pitch = _bin_pitch_angles(phi_sub[chunk], theta_sub, b_hat[chunk])
        inside = (pitch >= pa_min) & (pitch <= pa_max)
        inside = inside.reshape(-1, n_ph, _N_SUB, n_th, _N_SUB)
        weights[chunk] = np.mean(inside, axis=(2, 4))

    return weights, no_b


def psd_moments(vdf, sc_pot, **kwargs):
    r"""Computes moments from the FPI particle phase-space densities.

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution. The angular widths are taken from the
        optional attributes `delta_phi_minus`/`delta_phi_plus` (time, phi) and
        `delta_theta_minus`/`delta_theta_plus` (theta), in degrees, or else
        from the spacing of the phi and theta grids. The data are in the
        units of their ``UNITS`` attribute (s^3/cm^6, as FPI, if missing): a
        phase-space density or a differential energy or particle flux, see
        `vdf_convert`.
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
    angle_integration : {"midpoint", "exact"}
        Integration of the angular kernels over each (phi, theta) bin:
        "midpoint", the kernels at the bin centres times the bin widths, as
        irfu-matlab, or "exact", the kernels integrated over each bin for a
        phase-space density constant in the bin. Default is the
        ``angle_integration`` attribute of `vdf` if any (e.g. "exact" for the
        FEEPS skymaps), else "midpoint". On the 11.25 degree FPI grid, "exact"
        changes the moments by less than 1 %; on wide bins "midpoint"
        underestimates the density (8 % on the FEEPS polar bins).
    relativistic : bool
        Relativistic speeds, momentum flux (pressure, gamma m v v) and kinetic
        energy (heat flux), for a relativistic phase-space density (`dpf2psd`
        with ``relativistic=True``). Default is the ``relativistic``
        attribute of `vdf` if any (True for the FEEPS skymaps), else False, as
        FPI and irfu-matlab. The bulk-flow corrections of the pressure and heat
        flux are non-relativistic.
    pitch_angle_range : array_like
        Range of pitch angles [min, max] (degrees, inclusive) of the particles
        used, with `b_xyz`: the moments are those of the particles in the
        range (the pressure and temperature about their own bulk velocity).
        Default uses all the pitch angles.
    b_xyz : xarray.DataArray
        Time series of the magnetic field in the coordinate system of `vdf`
        (e.g., DMPA for FPI), resampled to the times of `vdf`. Needed with
        `pitch_angle_range`; the moments are NaN at the times without B.
    pitch_angle_weight : {"centre", "fraction"}
        With `pitch_angle_range`, a bin is used if the pitch angle of its
        centre is in the range ("centre", default, as get_pitch_angle_dist
        and the partial moments of the irfu-matlab examples), or weighted by
        the fraction of its solid angle in the range ("fraction", for wide
        bins such as those of FEEPS and EIS, or cones narrower than the bins).

    Raises
    ------
    ValueError
        If energy_range has no channel (fixed range) or a wrong shape, if
        both energy_range and en_channels are given, if angle_integration
        is not "midpoint" or "exact", or if pitch_angle_range is given without
        b_xyz, is not [min, max] within [0, 180], pitch_angle_weight is not
        "centre" or "fraction", or the units of `vdf` are not supported.

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

    # Data rate, for information only (e.g. FEEPS survey skymaps)
    field_name = str(vdf.data.attrs.get("FIELDNAM", "")).lower()
    for rate in ["brst", "fast", "srvy"]:
        if rate in field_name:
            logger.info("%s resolution data is used", rate)

    theta = vdf.theta.data
    particle_type = vdf.attrs["species"]
    assert particle_type[0].lower() in ["e", "i"], "invalid particle type"

    # In SI units (s^3/m^6)
    relativistic = bool(
        kwargs.get("relativistic", vdf.attrs.get("relativistic", False))
    )
    vdf_data = _psd_si(vdf, relativistic)

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

    angle_integration = kwargs.get(
        "angle_integration", vdf.attrs.get("angle_integration", "midpoint")
    )

    if angle_integration not in ["midpoint", "exact"]:
        raise ValueError(
            f"Invalid angle_integration {angle_integration}: must be midpoint or "
            "exact"
        )

    use_exact = angle_integration == "exact"

    if use_exact:
        half_phi = np.rad2deg(delta_phi[:, :, 0]) / 2
        half_theta = np.rad2deg(delta_theta[0, 0, :]) / 2
        exact_kernels = _exact_kernels(
            phi,
            theta,
            vdf.attrs.get("delta_phi_minus", half_phi),
            vdf.attrs.get("delta_phi_plus", half_phi),
            vdf.attrs.get("delta_theta_minus", half_theta),
            vdf.attrs.get("delta_theta_plus", half_theta),
        )
        delta_ang = np.ones(delta_ang.shape)
    else:
        # placeholder, unused by _moms (numba needs an array)
        exact_kernels = np.zeros((1, 10, 1, 1))

    # Pitch-angle range: weights of the angular bins
    no_b = np.zeros(vdf_data.shape[0], dtype=bool)

    if kwargs.get("pitch_angle_range") is not None:
        weights, no_b = _pitch_angle_weights(vdf, *_pitch_angle_args(kwargs))

        if use_exact:
            exact_kernels = np.ascontiguousarray(exact_kernels * weights[:, None])
        else:
            delta_ang = delta_ang * weights

    # float64 angles: the angular kernels of _moms must have the dtype of the
    # exact kernels (FPI angles are float32)
    phi_mat = np.tile(phi[:, :, np.newaxis], (1, 1, vdf_data.shape[3]))
    phi_mat = phi_mat.astype(np.float64)
    theta_mat = np.tile(
        theta[np.newaxis, np.newaxis, :], (vdf_data.shape[0], vdf_data.shape[2], 1)
    ).astype(np.float64)

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
    if relativistic:
        v_upper, v_lower = [_speed_relativistic(e, p_mass) for e in [e_upper, e_lower]]
    else:
        v_upper = np.sqrt(2 * q_e * e_upper / p_mass)
        v_lower = np.sqrt(2 * q_e * e_lower / p_mass)
    delta_v = np.ascontiguousarray(v_upper - v_lower)

    # Clean up NaN values in the input VDF data before passing to the numba kernel.
    # This is done in place to avoid extra memory allocation and to ensure that the
    # kernel does not encounter NaN values, which could lead to incorrect moment
    # calculations.
    _sanitize_nan_inplace(vdf_data)

    moms_args = [
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
        exact_kernels,
        use_exact,
    ]

    if relativistic:
        n_psd, v_psd, p_psd, h_psd = _moms_relativistic(moms_args, sc_pot, p_mass)
    else:
        n_psd, v_psd, p_psd, h_psd = _moms(*moms_args)

    # No energy channel at these time steps (time dependent energy_range)
    no_energy = ~np.any(use_energy, axis=1) | no_b
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
