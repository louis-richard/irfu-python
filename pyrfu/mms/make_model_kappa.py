#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants, special

# Local imports
from ..pyrf.resample import resample
from ._psd_units import _attrs

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def make_model_kappa(vdf, n_s, v_xyz_s, t_s, kappa: float = 7.0, sc_pot=None):
    r"""Makes an isotropic kappa distribution function from particle moments,
    on the bins of the skymap `vdf`, as the kappa distribution of irfu-matlab
    PDist.generate_dist.

    Parameters
    ----------
    vdf : xarray.Dataset
        Particle distribution (skymap), ions or electrons.
    n_s : xarray.DataArray
        Time series of the number density (cm^-3).
    v_xyz_s : xarray.DataArray
        Time series of the bulk velocity (km/s), in the coordinates of `vdf`.
    t_s : xarray.DataArray
        Time series of the scalar temperature (eV).
    kappa : float, Optional
        Kappa index, larger than 3/2. Default is 7.
    sc_pot : xarray.DataArray, Optional
        Time series of the spacecraft potential (V): the energies are
        corrected as in `make_model_vdf`. Default is no correction.

    Returns
    -------
    model_vdf : xarray.Dataset
        Kappa distribution in the format of `vdf`, in s^3/m^6, as
        irfu-matlab. Channels whose energy is below the spacecraft potential
        are NaN, as in `make_model_vdf`.

    Raises
    ------
    ValueError
        If the species is not ions or electrons, or `kappa` is not larger
        than 3/2.

    See Also
    --------
    pyrfu.mms.make_model_vdf

    Notes
    -----
    .. math::

        f(v) = \frac{n \Gamma(\kappa + 1)}{\Gamma(\kappa - 1/2)
        (\pi w^2)^{3/2}} \left(1 + \frac{|v - V|^2}{w^2}\right)^{-(\kappa + 1)}

    with :math:`w^2 = (\kappa - 3/2) v_{th}^2` and
    :math:`v_{th}^2 = 2 T / m`, so that the temperature of the model is T
    for any κ and it tends to the Maxwellian as κ → ∞. Not yet generalised
    to bi-kappa distributions.

    """
    if kappa <= 1.5:
        raise ValueError("kappa must be larger than 3/2")

    species = str(vdf.attrs["species"]).lower()

    if species[:1] not in ["i", "e"]:
        raise ValueError(f"Invalid species {vdf.attrs['species']!r}")

    q_e = constants.elementary_charge
    mass = constants.electron_mass if species[0] == "e" else constants.proton_mass

    # Moments at the times of the distribution, in SI units
    n_s = 1e6 * resample(n_s, vdf.time).data
    v_bulk = 1e3 * resample(v_xyz_s, vdf.time).data
    w_2 = (kappa - 1.5) * 2 * q_e * resample(t_s, vdf.time).data / mass

    # Speeds of the channels (time, energy, 1, 1), corrected for the
    # spacecraft potential as make_model_vdf (electrons lose and ions gain
    # q V); NaN below it
    energy = np.broadcast_to(vdf.energy.data, (len(vdf.time), vdf.data.shape[1]))
    energy = energy.astype(np.float64)

    if sc_pot is not None:
        sign = 1.0 if species[0] == "e" else -1.0
        energy = energy - sign * resample(sc_pot, vdf.time).data[:, None]

    below = energy < 0
    speed = np.sqrt(2 * q_e * np.clip(energy, 0.0, None) / mass)
    speed = speed[..., None, None]

    # Particle velocity directions (minus the look directions), each time
    # with its own azimuths: (time, 1, phi, theta)
    phi = np.deg2rad(np.atleast_2d(vdf.phi.data))[:, :, None]
    theta = np.deg2rad(vdf.theta.data)[None, None, :]
    dirs = [
        -np.cos(phi) * np.sin(theta),
        -np.sin(phi) * np.sin(theta),
        -np.cos(theta) * np.ones_like(phi),
    ]
    v_dot = sum(d * v_bulk[:, i, None, None] for i, d in enumerate(dirs))[:, None]

    # |v - V|^2 = v^2 - 2 v (d.V) + V^2, without 4-D tiles
    v_bulk2 = np.sum(v_bulk**2, axis=1)[:, None, None, None]
    dv_2 = speed**2 - 2 * speed * v_dot + v_bulk2

    w_2 = w_2[:, None, None, None]
    # Γ(κ + 1) / Γ(κ - 1/2) through the logarithms (Γ overflows above 170)
    gamma_ratio = np.exp(special.gammaln(kappa + 1) - special.gammaln(kappa - 0.5))
    coeff = n_s[:, None, None, None] * gamma_ratio / (np.pi * w_2) ** 1.5
    model = coeff * (1 + dv_2 / w_2) ** -(kappa + 1)
    model[np.broadcast_to(below[..., None, None], model.shape)] = np.nan

    model_vdf = vdf.copy()
    model_vdf["data"] = vdf.data.copy(data=model)
    # Without the label of the input units if they differ (e.g. a DEF)
    model_vdf.data.attrs = _attrs(vdf.data.attrs, "s^3/m^6")

    return model_vdf
