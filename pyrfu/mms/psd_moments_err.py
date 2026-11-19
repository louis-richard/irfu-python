#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from pyrfu.mms._internal.vdf_bins import _charge, _charge_sign, _mass, _speed
from pyrfu.mms.psd_moments import (
    _energy_edges,
    _exact_kernels,
    _midpoint_kernels,
    _psd_si,
)
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_tensor_xyz import ts_tensor_xyz
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

logger = logging.getLogger(__name__)

# The 13 raw moments, in this order: density (n), particle flux (nVx, nVy,
# nVz), momentum flux (Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) and energy flux (Qx, Qy,
# Qz). Each one is the sum over the bins of the phase-space density times an
# angular kernel (index in the output of _midpoint_kernels) times an energy
# weight (index in the output of _energy_weights).
_KERNEL = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3])
_WEIGHT = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3])


def _angular_weights(vdf):
    # Angular kernels of the raw moments integrated over each (phi, theta)
    # bin, (time, 10, phi, theta), as psd_moments with its default options:
    # the kernels at the centre of the bin times its solid angle, or
    # integrated exactly over the bin when the skymap has the attribute
    # angle_integration="exact" (FEEPS), with the widths of the bins from
    # the attributes delta_phi_minus/plus and delta_theta_minus/plus
    # (degrees, as HPCA and FEEPS set them) or from the grid spacing.
    phi = vdf.phi.data
    theta = vdf.theta.data

    if "delta_phi_minus" in vdf.attrs and "delta_phi_plus" in vdf.attrs:
        delta_phi = vdf.attrs["delta_phi_plus"] + vdf.attrs["delta_phi_minus"]
        delta_phi = np.broadcast_to(np.deg2rad(delta_phi), phi.shape)
    else:
        delta_phi = np.deg2rad(np.median(np.diff(phi[0, :]))) * np.ones(phi.shape)

    if "delta_theta_minus" in vdf.attrs and "delta_theta_plus" in vdf.attrs:
        delta_theta = vdf.attrs["delta_theta_plus"] + vdf.attrs["delta_theta_minus"]
        delta_theta = np.deg2rad(delta_theta)
    else:
        delta_theta = np.deg2rad(np.median(np.diff(theta))) * np.ones(theta.shape)

    if vdf.attrs.get("angle_integration", "midpoint") == "exact":
        half_phi = np.rad2deg(delta_phi) / 2
        half_theta = np.rad2deg(delta_theta) / 2
        widths = [
            vdf.attrs.get(k, half_phi) for k in ["delta_phi_minus", "delta_phi_plus"]
        ]
        widths += [
            vdf.attrs.get(k, half_theta)
            for k in ["delta_theta_minus", "delta_theta_plus"]
        ]
        return _exact_kernels(phi, theta, *widths)

    delta_ang = delta_phi[:, :, np.newaxis] * delta_theta[np.newaxis, np.newaxis, :]

    return _midpoint_kernels(phi, theta) * delta_ang[:, np.newaxis]


def _energy_weights(vdf, sc_pot, p_mass):
    # Energy weights of the raw moments, (time, 4, energy): density, particle
    # flux, momentum flux and energy flux, as psd_moments._moments (SI units,
    # with the particle mass). `sc_pot` (time,) is the potential energy in eV
    # removed from the energies of the channels.
    energy = vdf.energy.data
    energy0 = vdf.attrs["energy0"]
    energy1 = vdf.attrs["energy1"]
    flag_same_e = np.all(energy1 - energy0 == 0)

    # Speed width of each channel, from its energy edges
    energy_upper, energy_lower = _energy_edges(
        vdf, energy, energy0, energy1, vdf.attrs["esteptable"], flag_same_e
    )
    sc_pot = sc_pot[:, np.newaxis]
    v_upper = _speed(np.clip(energy_upper - sc_pot, 0.0, None), p_mass)
    v_lower = _speed(np.clip(energy_lower - sc_pot, 0.0, None), p_mass)
    delta_v = v_upper - v_lower

    # Channels used: those above the potential energy
    energy_correct = energy - sc_pot
    used = energy_correct >= 0.0

    # Relativistic speed, Lorentz factor - 1 (eps) and kinetic energy over
    # m v^2 / 2 (kinetic)
    eps = constants.elementary_charge * energy_correct / (p_mass * constants.c**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.where(used, _speed(energy_correct, p_mass), 0.0)
        kinetic = np.where(velocity > 0.0, 2 * eps * constants.c**2 / velocity**2, 1.0)

    delta_v = np.where(used, delta_v, 0.0)

    weights = [
        delta_v * velocity**2,
        delta_v * velocity**3,
        p_mass * (1.0 + eps) * delta_v * velocity**4,
        0.5 * p_mass * kinetic * delta_v * velocity**5,
    ]

    return np.stack(weights, axis=1)


def _bin_weights(energy, angular, i_t):
    # Weights of the 13 raw moments in each bin at the time i_t, (13, energy
    # x phi x theta): the energy weight times the angular kernel of each
    # moment.
    weights = (
        energy[i_t, _WEIGHT][:, :, np.newaxis, np.newaxis]
        * angular[i_t, _KERNEL][:, np.newaxis, :, :]
    )

    return weights.reshape(13, -1)


def _moments_cov(vdf, vdf_err, sc_pot):
    r"""Raw moments (time, 13) and covariance matrix (time, 19, 19) of the
    derived moments in SI units, for independent bins:

    .. math::

        M_a = \sum_s w^a_s f_s, \qquad
        \mathrm{Cov}(D_i, D_j) = \sum_s W^i_s W^j_s \sigma_{f, s}^2, \qquad
        W^i_s = \sum_a J_{ia} w^a_s,

    where :math:`s` runs over the (energy, phi, theta) bins, :math:`w^a_s` is
    the angular weight times the energy weight of the raw moment :math:`a`,
    :math:`f_s` the phase-space density, :math:`\sigma_{f, s}` its error
    and :math:`J` the Jacobian of the derived moments :math:`D` with respect
    to the raw moments. :math:`W^i_s` are the weights of the linearised
    derived moments, so that the covariance is summed bin by bin: it is
    :math:`J \Sigma J^T` with :math:`\Sigma` the covariance of the raw
    moments, but without its cancellation of large terms (for a cold beam,
    :math:`J \Sigma J^T` can give negative variances).
    """

    p_mass = _mass(vdf)

    # Phase-space density and its error in s^3/m^6, (time, energy, phi,
    # theta), without NaN. _psd_si is linear in the data, so that the error
    # is converted as the phase-space density.
    psd = _psd_si(vdf)
    psd_err = _psd_si(vdf_err)

    # A NaN error of a measured bin would be propagated as a zero variance
    n_missing = int(np.sum(np.isnan(psd_err) & ~np.isnan(psd)))

    if n_missing:
        logger.warning(
            "%d bins have a NaN error and a finite phase-space density: "
            "their variance is taken as zero",
            n_missing,
        )

    psd[np.isnan(psd)] = 0.0
    psd_err[np.isnan(psd_err)] = 0.0
    psd_var = psd_err**2

    angular = _angular_weights(vdf)
    energy = _energy_weights(vdf, sc_pot, p_mass)

    n_t = psd.shape[0]
    moments = np.zeros((n_t, 13))

    # (numpy 2.0 to 2.2 with Apple Accelerate raises spurious floating-point
    # warnings in matmul products of finite arrays, as in psd_moments)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i_t in range(n_t):
            moments[i_t] = _bin_weights(energy, angular, i_t) @ psd[i_t].ravel()

    # Without particles (n = 0) the errors of the derived moments are NaN
    # (0 / 0, silently, as the moments of psd_moments)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, jacobian = _derived_moments(moments, p_mass)

    covariance = np.zeros((n_t, 19, 19))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i_t in range(n_t):
            # Weights of the 19 linearised derived moments in each bin
            weights = jacobian[i_t] @ _bin_weights(energy, angular, i_t)
            covariance[i_t] = (weights * psd_var[i_t].ravel()) @ weights.T

    return moments, covariance


def _check_error_skymap(vdf, vdf_err):
    # Raises if vdf and vdf_err are not skymaps (xarray.Dataset) with the
    # same times, shape and units.
    for inp, label in [(vdf, "vdf"), (vdf_err, "vdf_err")]:
        if not isinstance(inp, xr.Dataset):
            raise TypeError(f"{label} must be a xarray.Dataset")

    if vdf_err.data.shape != vdf.data.shape:
        raise ValueError(
            f"vdf_err must have the shape of vdf {vdf.data.shape}, "
            f"not {vdf_err.data.shape}"
        )

    if not np.array_equal(vdf_err.time.data, vdf.time.data):
        raise ValueError("vdf_err must have the times of vdf")

    units_vdf, units_err = (inp.data.attrs.get("UNITS") for inp in [vdf, vdf_err])

    if units_vdf != units_err:
        raise ValueError(f"vdf_err must have the units of vdf ({units_vdf!r})")


def _sc_pot_energy(vdf, sc_pot):
    # Potential energy (eV) removed from the energies of the channels, at the
    # times of vdf: the electrons lose e V and the ions of charge q gain q V,
    # as psd_moments.
    sc_pot = resample(sc_pot, vdf.time).data

    if _charge_sign(vdf) < 0.0:
        return sc_pot

    return -_charge(vdf) * sc_pot


# The 19 derived moments, in this order: density (n), bulk velocity (Vx, Vy,
# Vz), pressure (Pxx, Pyy, Pzz, Pxy, Pxz, Pyz), temperature (same order) and
# heat flux (Qx, Qy, Qz). The six tensor components are the pairs below, in
# the order of the raw momentum flux.
_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
_DERIVED = (
    ["n"]
    + [f"V{c}" for c in "xyz"]
    + [f"P{'xyz'[i]}{'xyz'[j]}" for i, j in _PAIRS]
    + [f"T{'xyz'[i]}{'xyz'[j]}" for i, j in _PAIRS]
    + [f"Q{c}" for c in "xyz"]
)


def _derived_moments(moments, p_mass):
    r"""Derived moments (time, 19) and their Jacobian with respect to the raw
    moments (time, 19, 13), in SI units, as the end of psd_moments:

    .. math::

        V_i = F_i / n, \qquad
        P_{ij} = \Pi_{ij} - m F_i F_j / n, \qquad
        T_{ij} = P_{ij} / (k_B n),

        Q_i = E_i - \sum_j F_j \Pi_{ij} / n + m F_i |F|^2 / n^2
        - F_i \,\mathrm{tr}\,\Pi / (2 n),

    with :math:`n` the density, :math:`F` the particle flux, :math:`\Pi`
    the momentum flux and :math:`E` the energy flux (the raw moments). The
    heat flux is written with the raw moments only (the velocity, pressure
    and trace of psd_moments substituted), which is what the derivatives
    use.
    """

    k_b = constants.Boltzmann

    n_t = moments.shape[0]
    n = moments[:, 0]
    flux = moments[:, 1:4]
    pi_raw = moments[:, 4:10]
    e_flux = moments[:, 10:13]

    # The momentum flux as a (time, 3, 3) tensor and its trace
    pi_tensor = np.zeros((n_t, 3, 3))

    for k_, (i, j) in enumerate(_PAIRS):
        pi_tensor[:, i, j] = pi_raw[:, k_]
        pi_tensor[:, j, i] = pi_raw[:, k_]

    trace = np.trace(pi_tensor, axis1=1, axis2=2)

    derived = np.zeros((n_t, 19))
    jacobian = np.zeros((n_t, 19, 13))

    # Density
    derived[:, 0] = n
    jacobian[:, 0, 0] = 1.0

    # Bulk velocity V_i = F_i / n
    velocity = flux / n[:, np.newaxis]
    derived[:, 1:4] = velocity

    for i in range(3):
        jacobian[:, 1 + i, 0] = -velocity[:, i] / n
        jacobian[:, 1 + i, 1 + i] = 1.0 / n

    # Pressure P_ij = Pi_ij - m F_i F_j / n and temperature T_ij = P_ij /
    # (k_B n)
    for k_, (i, j) in enumerate(_PAIRS):
        pressure = pi_raw[:, k_] - p_mass * flux[:, i] * flux[:, j] / n
        derived[:, 4 + k_] = pressure
        derived[:, 10 + k_] = pressure / (k_b * n)

        d_p = jacobian[:, 4 + k_]
        d_p[:, 0] = p_mass * velocity[:, i] * velocity[:, j]
        d_p[:, 1 + i] -= p_mass * velocity[:, j]
        d_p[:, 1 + j] -= p_mass * velocity[:, i]
        d_p[:, 4 + k_] = 1.0

        d_t = jacobian[:, 10 + k_]
        d_t[...] = d_p / (k_b * n[:, np.newaxis])
        d_t[:, 0] -= pressure / (k_b * n**2)

    # Heat flux Q_i = E_i - sum_j F_j Pi_ij / n + m F_i |F|^2 / n^2
    # - F_i tr(Pi) / (2 n)
    v_abs2 = np.sum(velocity**2, axis=1)
    pi_v = np.einsum("tij,tj->ti", pi_tensor, velocity)

    for i in range(3):
        derived[:, 16 + i] = (
            e_flux[:, i]
            - pi_v[:, i]
            + p_mass * n * velocity[:, i] * v_abs2
            - 0.5 * velocity[:, i] * trace
        )

        d_q = jacobian[:, 16 + i]
        d_q[:, 0] = (
            pi_v[:, i]
            - 2 * p_mass * n * velocity[:, i] * v_abs2
            + 0.5 * velocity[:, i] * trace
        ) / n

        for k in range(3):
            d_q[:, 1 + k] = (
                -pi_tensor[:, i, k] / n + 2 * p_mass * velocity[:, i] * velocity[:, k]
            )

        d_q[:, 1 + i] += p_mass * v_abs2 - 0.5 * trace / n

        for k_, (k, l) in enumerate(_PAIRS):
            if k == l:
                d_q[:, 4 + k_] = -0.5 * velocity[:, i]
                if k == i:
                    d_q[:, 4 + k_] -= velocity[:, k]
            else:
                if k == i:
                    d_q[:, 4 + k_] -= velocity[:, l]
                if l == i:
                    d_q[:, 4 + k_] -= velocity[:, k]

        d_q[:, 10 + i] = 1.0

    return derived, jacobian


def psd_moments_err(vdf, vdf_err, sc_pot):
    r"""Computes the counting-statistics uncertainties of the moments of
    `psd_moments`, from the error of the phase-space density.

    Parameters
    ----------
    vdf : xarray.Dataset
        3D skymap velocity distribution, as for `psd_moments` (e.g. the FPI
        ``dist`` skymap). Give it already corrected (e.g. with
        `remove_edist_background`) to get the errors of the corrected moments.
    vdf_err : xarray.Dataset
        Skymap of the 1-sigma error of the phase-space density, with the same
        bins and units as `vdf` (e.g. the FPI ``disterr`` skymap).
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential, as for `psd_moments`.

    Returns
    -------
    n_err : xarray.DataArray
        Time series of the error of the number density (cm^-3).
    v_err : xarray.DataArray
        Time series of the errors of the bulk velocity components (km/s).
    p_err : xarray.DataArray
        Time series of the errors of the pressure tensor components (nPa).
    t_err : xarray.DataArray
        Time series of the errors of the temperature tensor components (eV).
    q_err : xarray.DataArray
        Time series of the errors of the heat flux components (mW/m^2).
    cov : xarray.DataArray
        Covariance matrix (time, moment, moment) of the 19 moments n, Vx, Vy,
        Vz, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, Txx, ..., Tyz, Qx, Qy, Qz in the
        units above (the squared errors on its diagonal).

    Raises
    ------
    TypeError
        If `vdf` or `vdf_err` is not a xarray.Dataset.
    ValueError
        If `vdf_err` does not have the times, the shape or the units of `vdf`.

    Notes
    -----
    The errors are those of the moments computed by `psd_moments` with its
    default options (all the energy channels, spacecraft potential
    correction, the angular integration and the angular widths of the
    attributes of `vdf`) from the same `vdf` and `sc_pot`. They are not those
    of its options `energy_range`, `en_channels`, `partial_moments`,
    `inner_electron`, `pitch_angle_range` and `no_sc_pot`.

    A NaN of `vdf` is a bin without data (as `psd_moments`); a NaN of
    `vdf_err` where `vdf` is finite counts as a zero variance, with a
    warning.

    The bins are taken as independent. The 13 raw moments (density,
    particle flux, momentum flux, energy flux) are sums over the bins of the
    phase-space density times weights, and the velocity, pressure,
    temperature and heat flux are linearised around them, so that the
    moments are sums over the bins too, with weights given by the Jacobian;
    their covariance is the sum over the bins of the products of these
    weights times the variance of the phase-space density.

    For particle counting, the variance of the phase-space density of a bin
    is its value times its conversion factor from counts (Poisson
    statistics, [1]_), which is what the FPI ``disterr`` gives. The errors are
    then exact for the density and particle flux, and linearised for the
    bulk velocity, pressure, temperature and heat flux.

    The linearisation holds for well-sampled distributions. On FPI-DIS burst
    distributions of about 400 counts (2017-07-11), the errors of the bulk
    velocity and temperature are 15 to 35 % below the spread of Poisson
    resamplings through `psd_moments` (10 % with 4000 counts, within the
    resampling noise with 40000 and for FPI-DES distributions of 20000).

    The FPI fast survey distributions are lossy compressed (attribute
    ``compressionloss`` of `get_dist`): their counts are not Poisson where
    they are low, and do not go below a floor of about 2 per bin (Barrie et
    al., 2017; see `get_dist`). The variances of the bins with fewer true
    counts are then wrong; those of the burst distributions, mostly lossless,
    are not.

    See Also
    --------
    pyrfu.mms.psd_moments : the moments.

    References
    ----------
    .. [1]  Gershman, D. J., Dorelli, J. C., F.-Viñas, A., and Pollock,
            C. J. (2015), The calculation of moment uncertainties from
            velocity distribution functions with random errors, J. Geophys.
            Res. Space Physics, 120, 6633–6645,
            doi: https://doi.org/10.1002/2014JA020775

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Load the spacecraft potential, the electron distribution and its error

    >>> scpot = mms.get_data("V_edp_brst_l2", tint_brst, 1)
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint_brst, 1)
    >>> vdf_e_err = mms.get_data("pderre_fpi_brst_l2", tint_brst, 1)

    Compute the moments and their errors

    >>> moments_e = mms.psd_moments(vdf_e, scpot)
    >>> errors_e = mms.psd_moments_err(vdf_e, vdf_e_err, scpot)
    """

    _check_error_skymap(vdf, vdf_err)

    sc_pot = _sc_pot_energy(vdf, sc_pot)

    # Covariance of the derived moments in SI units
    _, cov = _moments_cov(vdf, vdf_err, sc_pot)

    # In the units of psd_moments: /cc, km/s, nPa, eV and mW/m^2
    scale = np.array(
        [1e-6]
        + [1e-3] * 3
        + [1e9] * 6
        + [constants.Boltzmann / constants.elementary_charge] * 6
        + [1e3] * 3
    )
    cov = cov * scale[:, np.newaxis] * scale[np.newaxis, :]
    errors = np.sqrt(np.einsum("tii->ti", cov))

    # Errors of the tensor components as (time, 3, 3)
    p_err = np.zeros((len(errors), 3, 3))
    t_err = np.zeros((len(errors), 3, 3))

    for k_, (i, j) in enumerate(_PAIRS):
        p_err[:, i, j] = p_err[:, j, i] = errors[:, 4 + k_]
        t_err[:, i, j] = t_err[:, j, i] = errors[:, 10 + k_]

    time = vdf.time.data
    cov = xr.DataArray(
        cov,
        coords=[time, _DERIVED, _DERIVED],
        dims=["time", "moment", "moment_"],
        attrs={"UNITS": "products of cm^-3, km/s, nPa, eV, mW/m^2"},
    )
    return (
        ts_scalar(time, errors[:, 0], {"UNITS": "cm^-3"}),
        ts_vec_xyz(time, errors[:, 1:4], {"UNITS": "km/s"}),
        ts_tensor_xyz(time, p_err, {"UNITS": "nPa"}),
        ts_tensor_xyz(time, t_err, {"UNITS": "eV"}),
        ts_vec_xyz(time, errors[:, 16:19], {"UNITS": "mW/m^2"}),
        cov,
    )
