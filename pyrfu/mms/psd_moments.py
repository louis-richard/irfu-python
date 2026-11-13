#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from pyrfu.mms._internal.vdf_bins import (
    _bin_pitch_angles,
    _charge,
    _charge_sign,
    _mass,
    _phi_edges,
    _speed,
    _theta_edges,
)
from pyrfu.mms.vdf_convert import _relativistic_psd
from pyrfu.pyrf.normalize import normalize
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_tensor_xyz import ts_tensor_xyz
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

logger = logging.getLogger(__name__)


def _midpoint_kernels(phi, theta):
    # Angular kernels of the moments at the centre of each (phi, theta) bin,
    # (time, kernel, phi, theta), in the order of _exact_kernels: density,
    # flux (x, y, z) and momentum flux (xx, yy, zz, xy, xz, yz). Angles in
    # degrees, phi (time, phi), theta (theta,).
    phi = np.deg2rad(np.asarray(phi, dtype=np.float64))[:, :, np.newaxis]
    theta = np.deg2rad(np.asarray(theta, dtype=np.float64))[np.newaxis, np.newaxis, :]
    c_p, s_p, c_t, s_t = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    kernels = [
        s_t,
        -c_p * s_t**2,
        -s_p * s_t**2,
        -s_t * c_t,
        c_p**2 * s_t**3,
        s_p**2 * s_t**3,
        s_t * c_t**2,
        c_p * s_p * s_t**3,
        c_p * s_t**2 * c_t,
        s_p * s_t**2 * c_t,
    ]
    shape = (phi.shape[0], phi.shape[1], theta.shape[2])
    return np.stack([np.broadcast_to(k_, shape) for k_ in kernels], axis=1)


def _moments(angular_sums, energy_correct, delta_v, used, p_mass):
    r"""Relativistic moment integrals: density (time,), particle flux (time,
    3), momentum flux (time, 6: xx, yy, zz, xy, xz, yz) and energy flux
    (time, 3), from the angular integrals of the data `angular_sums` (time,
    energy, 10, see _midpoint_kernels) and the energy sums over the `used`
    channels with the relativistic speeds, the Lorentz factor (momentum flux
    gamma m v v) and the kinetic energy over m v^2 / 2 (energy flux), which
    depend on (time, energy) only (a numba kernel, three times, on energies
    whose non-relativistic speed was the relativistic one)."""
    eps = constants.elementary_charge * energy_correct / (p_mass * constants.c**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.where(used, _speed(energy_correct, p_mass), 0.0)
        kinetic = np.where(velocity > 0.0, 2 * eps * constants.c**2 / velocity**2, 1.0)

    d_v = np.where(used, delta_v, 0.0)
    weights = [
        d_v * velocity**2,
        d_v * velocity**3,
        (1.0 + eps) * d_v * velocity**4,
        kinetic * d_v * velocity**5,
    ]

    n_psd = np.sum(angular_sums[..., 0] * weights[0], axis=1)
    v_psd, p_psd, h_psd = [
        np.sum(angular_sums[..., kernels] * w_[..., np.newaxis], axis=1)
        for kernels, w_ in zip([slice(1, 4), slice(4, 10), slice(1, 4)], weights[1:])
    ]

    return n_psd, v_psd, p_psd, h_psd


def _exact_kernels(
    phi, theta, delta_phi_minus, delta_phi_plus, delta_theta_minus, delta_theta_plus
):
    # Angular kernels of the moments integrated over each (phi, theta) bin,
    # assuming a constant phase-space density in the bin: (time, kernel, phi,
    # theta), kernels in the order of _midpoint_kernels. Angles and widths in
    # degrees, phi (time, phi), theta (theta,).
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


def _first_channel_mask(vdf, n_e, first_channel):
    r"""Channels from a time series of first channels (e.g. the recommended
    energy index of the FPI partial moments) to the last, as a boolean (time,
    energy) array."""
    if not np.array_equal(first_channel.time.data, vdf.time.data):
        # Nearest sample: the indices are integers, not to be interpolated
        first_channel = first_channel.sel(time=vdf.time.data, method="nearest")

    start = np.asarray(first_channel.data)

    if start.shape != (len(vdf.time),) or np.any((start < 0) | (start >= n_e)):
        raise ValueError(
            f"en_channels as a time series must be a scalar time series of "
            f"channel indices in [0, {n_e})"
        )

    return np.arange(n_e)[np.newaxis, :] >= start.astype(np.int64)[:, np.newaxis]


def _energy_mask(vdf, energy, energy_range, en_channels):
    r"""Energy channels to integrate over at each time step, as a boolean
    (time, energy) array, from the energy_range or en_channels options."""
    n_t, n_e = energy.shape

    if energy_range is not None and en_channels is not None:
        raise ValueError("Use either energy_range or en_channels, not both")

    if energy_range is None:
        en_channels = [0, n_e] if en_channels is None else en_channels

        if isinstance(en_channels, xr.DataArray):
            return _first_channel_mask(vdf, n_e, en_channels)

        start, stop = en_channels
        use_energy = np.zeros((n_t, n_e), dtype=np.bool_)
        use_energy[:, start:stop] = True
        return use_energy

    e_range = energy_range

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


# Sub-grid of the angular bins for the pitch-angle fractions
_N_SUB = 8


def _psd_si(vdf):
    r"""Relativistic phase-space density in s^3/m^6, as a new C-contiguous
    float64 array (get_dist returns a transposed view; the NaN are set to 0
    in place), from the units of `vdf` (``UNITS``, s^3/cm^6 if
    missing; differential fluxes as irfu-matlab PDist.convertto): that of the
    FPI, HPCA and EIS files times 2 gamma^5 / (gamma + 1)."""
    if "UNITS" not in vdf.data.attrs:
        vdf = vdf.copy()
        vdf["data"] = vdf.data.copy()
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

    return np.array(_relativistic_psd(vdf), dtype=np.float64, order="C")


def _partial_moments_mask(vdf, mask, shape):
    r"""The 0/1 (or boolean) array `partial_moments` of the bins used, of the
    shape of the data and, for a DataArray, at the times of `vdf`. A wrong
    mask raises ValueError: it was only logged (INFO) and the full moments
    were returned, as irfu-matlab."""
    if isinstance(mask, xr.DataArray):
        if "time" in mask.coords and not np.array_equal(mask.time.data, vdf.time.data):
            raise ValueError("partial_moments must be at the times of vdf")

        mask = mask.data

    mask = np.asarray(mask)

    if mask.shape != shape:
        raise ValueError(
            f"partial_moments must have the shape of vdf.data {shape}, "
            f"not {mask.shape}"
        )

    if not np.isin(mask, [0, 1]).all():
        raise ValueError("partial_moments must be a binary array (1s and 0s)")

    return mask


def _pitch_angle_args(b_xyz, pitch_angle_range, method):
    # b_xyz, [min, max] and method of the pitch-angle range, checked
    pitch_angle_range = np.asarray(pitch_angle_range, dtype=np.float64)

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


def psd_moments(
    vdf,
    sc_pot,
    *,
    energy_range=None,
    en_channels=None,
    no_sc_pot=False,
    partial_moments=None,
    inner_electron: Literal["on", "off"] = "off",
    angle_integration: Literal["midpoint", "exact"] | None = None,
    pitch_angle_range=None,
    b_xyz=None,
    pitch_angle_weight: Literal["centre", "fraction"] = "centre",
):
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
        `vdf_convert`. The ``species`` attribute gives the mass (e.g. "ions",
        "electrons", "oxygen+", "helium++") and the optional ``charge``
        attribute the charge of the ions (1 by default, e.g. 2 for the HPCA
        He++ skymaps).
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential. The ions gain q V and the
        electrons lose e V (the energies at infinity, see Notes).

    Returns
    -------
    n_psd : xarray.DataArray
        Time series of the number density (1rst moment).
    v_psd : xarray.DataArray
        Time series of the bulk velocity (2nd moment). NaN where the density
        is zero, as the pressure, temperature and heat flux.
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
    en_channels : array_like or xarray.DataArray
        Energy channels to integrate over:

        * [start, stop): 0-based indices as a Python slice (default [0, 32],
          all the channels; unlike irfu-matlab's 1-based inclusive
          [min max]);
        * a time series of first channels, e.g. FPI's recommended energy
          index ``partidxi_fpi_brst_l2``: at each time step, the channels
          from it to the last. Its nearest sample is used at each time of
          vdf.

        Can't be used with energy_range.
    partial_moments : numpy.ndarray or xarray.DataArray
        Use a binary array to select which psd points are used in the moments
        calculation. `partial_moments` must be a binary array (1s and 0s, or
        booleans, 1s correspond to points used). Array (or data of DataArray)
        must be the same size as vdf.data, and a DataArray at the times of
        vdf.
    inner_electron : {"on", "off"}
        "on" leaves out the electron channels below 3.5 eV above the
        spacecraft potential energy (the inner electrons). Default "off".
    angle_integration : {"midpoint", "exact"}
        Integration of the angular kernels over each (phi, theta) bin:
        "midpoint", the kernels at the bin centres times the bin widths, as
        irfu-matlab, or "exact", the kernels integrated over each bin for a
        phase-space density constant in the bin. Default is the
        ``angle_integration`` attribute of `vdf` if any (e.g. "exact" for the
        FEEPS skymaps), else "midpoint". On the 11.25 degree FPI grid, "exact"
        changes the moments by less than 1 %; on wide bins "midpoint"
        underestimates the density (8 % on the FEEPS polar bins).
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
        "centre" or "fraction", the units of `vdf` are not supported,
        en_channels as a time series is not of channel indices, or
        partial_moments is not a binary array of the shape of vdf.data (at
        the times of vdf for a DataArray).
    TypeError
        If ``relativistic`` is given (the argument was removed).

    Notes
    -----
    The moments are relativistic for all the species: relativistic speeds,
    momentum flux (pressure, gamma m v v) and kinetic energy (heat flux), with
    the relativistic phase-space density (that of the FPI, HPCA and EIS
    files, non-relativistic, times 2 gamma^5 / (gamma + 1); that of the FEEPS
    skymaps as it is). They were non-relativistic without the
    ``relativistic`` attribute or argument: for FPI electrons below 30 keV,
    the density changes by 0.25 % at most and the ion moments by less than
    0.001 %. The bulk-flow corrections of the pressure and heat flux are
    non-relativistic.

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

    The energies are corrected for the spacecraft potential V as the energies
    at infinity, E + q V (energy conservation): the electrons lose e V and
    the ions of charge q gain q V. For a species attracted by the spacecraft
    (the electrons for V > 0), the channels below its potential energy hold
    photo- and secondary electrons of the spacecraft and are left out; a
    channel whose centre energy is below the cut is left out whole, as in
    irfu-matlab, although its upper part is above it. For a species repelled
    by the spacecraft (the ions for V > 0), all the channels are ambient
    particles, but those below q V never reach the instrument: the moments
    are those of the ambient particles above q V, nothing being assumed
    below. The measured phase-space density is used at the corrected speeds,
    as in irfu-matlab: exact non-relativistically, whereas relativistically
    the invariant along the trajectories is the density in momentum space,
    so that the density in velocity space at infinity is the measured one
    times (gamma_inf / gamma)^5, about 1 + 5 q V / (m c^2) (-4e-4 for
    electrons at 38 V, negligible for ions).

    The FPI ion moments start at the recommended energy index of the
    partial-moments files (``partidxi_fpi_brst_l2``, constant within each
    file): 0, all the channels, in dense plasma, but up to 11 (100 eV) in the
    tenuous plasma sheet, where the lowest channels can hold most of the
    density (3.8 times the FPI density on 2017-07-11, MMS1). Use
    ``en_channels=mms.get_data("partidxi_fpi_brst_l2", tint, mms_id)`` to
    compare the ion moments with them, also across files with different
    indices (within 4 % on the magnetopause, magnetosheath, bow shock, solar
    wind and plasma sheet intervals tested). The penetrating-radiation
    background (`remove_idist_background`) is another correction, which is
    small there.

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

    # Mass and charge of the species (ValueError if not supported). All the
    # ions were protons and the species not starting with "e" or "i" raised
    # AssertionError.
    p_mass, is_electron = _mass(vdf), _charge_sign(vdf) < 0.0
    charge = 1.0 if is_electron else _charge(vdf)

    # Relativistic speeds, momentum flux and kinetic energy, for all the
    # species: the relativistic phase-space density in SI units (s^3/m^6)
    vdf_data = _psd_si(vdf)

    step_table = vdf.attrs["esteptable"]
    energy = vdf.energy.data
    energy0 = vdf.attrs["energy0"]
    energy1 = vdf.attrs["energy1"]
    e_tmp = energy1 - energy0

    flag_same_e = np.all(e_tmp == 0)

    # resample sc_pot to same resolution as particle distributions
    sc_pot = resample(sc_pot, vdf.time).data

    if no_sc_pot:
        sc_pot = np.zeros(sc_pot.shape)
        logger.info("Setting spacecraft potential to zero")

    use_energy = _energy_mask(vdf, energy, energy_range, en_channels)

    # The bins of partial_moments (a wrong mask gave the full moments)
    if partial_moments is not None:
        mask = _partial_moments_mask(vdf, partial_moments, vdf_data.shape)
        vdf_data = vdf_data * mask

    flag_inner_electron = inner_electron == "on" and is_electron

    # Define constants
    q_e = constants.elementary_charge
    k_b = constants.Boltzmann

    if is_electron:
        logger.info("Particles are electrons")
    else:
        # Ions of charge q gain q V (eV)
        sc_pot *= -charge
        logger.info("Particles are %s", vdf.attrs["species"])

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

    if angle_integration is None:
        angle_integration = vdf.attrs.get("angle_integration", "midpoint")

    if angle_integration not in ["midpoint", "exact"]:
        raise ValueError(
            f"Invalid angle_integration {angle_integration}: must be midpoint or "
            "exact"
        )

    use_exact = angle_integration == "exact"

    # Pitch-angle range: weights of the angular bins
    no_b = np.zeros(vdf_data.shape[0], dtype=bool)
    weights = None

    if pitch_angle_range is not None:
        weights, no_b = _pitch_angle_weights(
            vdf, *_pitch_angle_args(b_xyz, pitch_angle_range, pitch_angle_weight)
        )

    if use_exact:
        half_phi = np.rad2deg(delta_phi[:, :, 0]) / 2
        half_theta = np.rad2deg(delta_theta[0, 0, :]) / 2
        exact_widths = [
            np.broadcast_to(vdf.attrs.get(key, half_phi), phi.shape)
            for key in ["delta_phi_minus", "delta_phi_plus"]
        ]
        exact_widths += [
            vdf.attrs.get(key, half_theta)
            for key in ["delta_theta_minus", "delta_theta_plus"]
        ]

    # Speed widths of the energy channels, as in irfu-matlab mms.psd_moments:
    # the edges of each channel, corrected for the spacecraft potential
    energy_upper, energy_lower = _energy_edges(
        vdf, energy, energy0, energy1, step_table, flag_same_e
    )
    sc_pot_2d = sc_pot[:, np.newaxis]
    e_upper = np.clip(energy_upper - sc_pot_2d, 0.0, None)
    e_lower = np.clip(energy_lower - sc_pot_2d, 0.0, None)
    v_upper, v_lower = [_speed(e, p_mass) for e in [e_upper, e_lower]]
    delta_v = v_upper - v_lower

    # The data without NaN (as np.nansum), in place: vdf_data is a new array
    vdf_data[np.isnan(vdf_data)] = 0.0

    # Angular integrals of the data (time, energy, kernel): the kernels at the
    # centre of each bin times its solid angle, or integrated exactly over
    # the bin, times the pitch-angle weights, by blocks of times (the kernels
    # are (time, 10, phi, theta))
    n_t, n_e = vdf_data.shape[:2]
    angular_sums = np.empty((n_t, n_e, 10))

    for start in range(0, n_t, 256):
        block = slice(start, start + 256)

        if use_exact:
            kernels = _exact_kernels(
                phi[block],
                theta,
                *[width[block] for width in exact_widths[:2]],
                *exact_widths[2:],
            )
        else:
            kernels = _midpoint_kernels(phi[block], theta)
            kernels = kernels * delta_ang[block, np.newaxis]

        if weights is not None:
            kernels = kernels * weights[block, np.newaxis]

        # (numpy 2.0 with Apple Accelerate raises spurious floating-point
        # warnings in large matmul products of finite arrays, e.g. "divide
        # by zero encountered in matmul")
        n_b = kernels.shape[0]

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            angular_sums[block] = np.matmul(
                vdf_data[block].reshape(n_b, n_e, -1),
                kernels.reshape(n_b, 10, -1).transpose(0, 2, 1),
            )

    # The channels used, those of an attracted species above its potential
    # energy (plus w_inner_electron for the inner electrons)
    energy_correct = energy - sc_pot_2d
    used = use_energy & (energy_correct >= flag_inner_electron * w_inner_electron)

    n_psd, v_psd, p_moms, h_psd = _moments(
        angular_sums, energy_correct, delta_v, used, p_mass
    )

    # Momentum flux tensor (upper triangle, completed below)
    p_psd = np.zeros((n_t, 3, 3))
    for k_, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        p_psd[:, i, j] = p_moms[:, k_]

    # No energy channel at these time steps (time dependent energy_range)
    no_energy = ~np.any(use_energy, axis=1) | no_b
    n_psd[no_energy] = np.nan
    v_psd[no_energy] = np.nan
    p_psd[no_energy] = np.nan
    h_psd[no_energy] = np.nan

    # Compute moments in SI units. Without particles (n = 0), the bulk
    # velocity, pressure, temperature and heat flux are NaN (0 / 0, silently)
    p_psd *= p_mass

    with np.errstate(divide="ignore", invalid="ignore"):
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

    with np.errstate(divide="ignore", invalid="ignore"):
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

    # Time series, with their units (they had none)
    time = vdf.time.data
    return (
        ts_scalar(time, n_psd, {"UNITS": "cm^-3"}),
        ts_vec_xyz(time, v_psd, {"UNITS": "km/s"}),
        ts_tensor_xyz(time, p_psd, {"UNITS": "nPa"}),
        ts_tensor_xyz(time, p2_psd, {"UNITS": "nPa"}),
        ts_tensor_xyz(time, t_psd, {"UNITS": "eV"}),
        ts_vec_xyz(time, h_psd, {"UNITS": "mW/m^2"}),
    )
