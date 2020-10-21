#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_model_vdf.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from astropy import constants

from ..pyrf import resample, dec_par_perp, norm
from . import rotate_tensor


def make_model_vdf(vdf=None, b_xyz=None, sc_pot=None, n=None, v_xyz=None, t_xyz=None):
    """
    Make a general bi-Maxwellian distribution function based on particle moment data in the same
    format as PDist.

    Parameters
    ----------
    vdf : xarray.Dataset
        Particle distribution (skymap).

    b_xyz : xarray.DataArray
        Time series of the background magnetic field.

    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.

    n : xarray.DataArray
        Time series of the number density.

    v_xyz : xarray.DataArray
        Time series of the bulk velocity.

    t_xyz : xarray.DataArray
        Time series of the temperature tensor.

    Returns
    -------
    model_vdf : xarray.Dataset
        Distribution function in the same format as vdf.

    See also
    --------
    pyrfu.mms.calculate_epsilon : Calculates epsilon parameter using model distribution.

    Examples
    --------
    >>> from pyrfu import mms
    >>> model_vdf = mms.make_model_vdf(vdf_e, b_xyz, sc_pot, n_e, v_xyz_e, t_xyz_e)

    """

    # Check that PDist and moments have the same times
    """
    if abs(median(diff(PDist.time-n.time))) > 0
        modelPDist = NaN;
        irf.log('critical','PDist and moments have different times.')
        return;
    end
    """

    assert vdf is not None and isinstance(vdf, xr.Dataset)
    assert b_xyz is not None and isinstance(b_xyz, xr.DataArray)
    assert sc_pot is not None and isinstance(sc_pot, xr.DataArray)
    assert n is not None and isinstance(n, xr.DataArray)
    assert v_xyz is not None and isinstance(v_xyz, xr.DataArray)
    assert t_xyz is not None and isinstance(t_xyz, xr.DataArray)

    # Convert to SI units
    vdf /= 1e18

    # Resample b_xyz and sc_pot to particle data resolution
    b_xyz, sc_pot = [resample(b_xyz, n), resample(sc_pot, n)]

    # Define directions based on b_xyz and v_xyz, calculate relevant temperatures
    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")  # N.B makes final distribution gyrotropic
    t_para, t_ratio = [t_xyzfac[:, 0, 0], t_xyzfac[:, 0, 0] / t_xyzfac[:, 1, 1]]

    v_para, v_perp, alpha = dec_par_perp(v_xyz, b_xyz)

    v_perp_mag, b_xyz_mag = [norm(v_perp), norm(b_xyz)]
    v_perp_dir, b_xyz_dir = [v_perp / v_perp_mag, b_xyz / b_xyz_mag]

    # Define constants
    qe = constants.e.value

    # Check whether particles are electrons or ions
    if vdf.attrs["species"].lower() == "e":
        p_mass = constants.m_e.value
        print("notice : Particles are electrons")
    elif vdf.attrs["species"].lower() == "i":
        p_mass = constants.m_p.value
        sc_pot.data = -sc_pot.data
        print("notice : Particles are ions")
    else:
        raise ValueError("Invalid specie")

    # Convert moments to SI units
    vth_para = np.sqrt(2 * t_para * qe / p_mass)

    v_perp_mag.data *= 1e3
    v_para.data *= 1e3
    n.data *= 1e6

    # Defines dimensions of array below
    n_ti = len(vdf.time)
    n_ph = len(vdf.phi[0, :])
    n_th = len(vdf.theta)
    n_en = len(vdf.attrs["energy0"])

    # Get energy array
    energy = vdf.energy

    # Define Cartesian coordinates
    x, y, z = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    r = np.zeros((n_ti, n_en))

    for ii in range(n_ti):
        x[ii, ...] = np.outer(
            -np.cos(vdf.phi.data[ii, :] * np.pi / 180) * np.sin(vdf.theta.data * np.pi / 180))
        y[ii, ...] = np.outer(
            -np.sin(vdf.phi.data[ii, :] * np.pi / 180) * np.sin(vdf.theta.data * np.pi / 180))
        z[ii, ...] = np.outer(-np.ones(n_ph) * np.cos(vdf.theta.data * np.pi / 180))
        r[ii, ...] = np.real(np.sqrt(2 * (energy[ii, :] - sc_pot.data[ii]) * qe / p_mass))

    r[r == 0] = 0.0

    # Define rotation vectors based on B and Ve directions
    r_x, r_y, r_z = v_perp_dir.data, np.cross(b_xyz_dir.data, v_perp_dir.data), b_xyz_dir.data

    # Rotated coordinate system for computing bi-Maxwellian distribution
    x_p, y_p, z_p = [np.zeros((n_ti, n_ph, n_th)) for _ in range(3)]

    for ii in range(n_ti):
        x_p[ii, ...] = x[ii, ...] * r_x[ii, 0] + y[ii, ...] * r_x[ii, 1] + z[ii, ...] * r_x[ii, 2]
        y_p[ii, ...] = x[ii, ...] * r_y[ii, 0] + y[ii, ...] * r_y[ii, 1] + z[ii, ...] * r_y[ii, 2]
        z_p[ii, ...] = x[ii, ...] * r_z[ii, 0] + y[ii, ...] * r_z[ii, 1] + z[ii, ...] * r_z[ii, 2]

    # Make 4D position matrix
    x_p = np.tile(x_p, [n_en, 1, 1, 1])
    x_p = np.transpose(x_p, [1, 0, 2, 3])

    y_p = np.tile(x_p, [n_en, 1, 1, 1])
    y_p = np.transpose(y_p, [1, 0, 2, 3])

    z_p = np.tile(x_p, [n_en, 1, 1, 1])
    z_p = np.transpose(z_p, [1, 0, 2, 3])

    r_mat = np.tile(r, [n_ph, n_th, 1, 1])
    r_mat = np.transpose(r_mat, [2, 3, 0, 1])

    # Can use 4D matrices. Too much memory is used for my computer; too slow for large files.
    # Make 4D matrices required for distribution calculation
    # nmat = repmat(n.data,1,lengthenergy,lengthphi,lengththeta);
    # Tratmat = repmat(Trat.data,1,lengthenergy,lengthphi,lengththeta);
    # Vpmagmat = repmat(Vpmag.data,1,lengthenergy,lengthphi,lengththeta);
    # Vparmat = repmat(Vpar.data,1,lengthenergy,lengthphi,lengththeta);
    # vthparmat = repmat(vthpar.data,1,lengthenergy,lengthphi,lengththeta);

    # Calculate bi-Maxwellian distribution function
    # bimaxdist = nmat * Tratmat./(sqrt(pi^3)*vthparmat.^3);
    # bimaxdist = bimaxdist.*exp(-(xp.*rmat-Vpmagmat).^2./(vthparmat.^2).*Tratmat);
    # bimaxdist = bimaxdist.*exp(-(yp.*rmat).^2./(vthparmat.^2).*Tratmat);
    # bimaxdist = bimaxdist.*exp(-(zp.*rmat-Vparmat).^2./(vthparmat.^2));

    # Construct bi-Maxwellian distribution function
    bi_max_dist = np.zeros(r_mat.shape)

    for ii in range(n_ti):
        coeff = n.data[ii] * t_ratio.data[ii] / (np.sqrt(np.pi ** 3) * vth_para.data[ii] ** 3)

        bi_max_temp = coeff * np.exp(
            -(x_p[ii, ...] * r_mat[ii, ...] - v_perp_mag.data[ii]) ** 2 / (vth_para.data[ii] ** 2) *
            t_ratio.data[ii])
        bi_max_temp = bi_max_temp * np.exp(
            -(y_p[ii, ...] * r_mat[ii, ...]) ** 2 / (vth_para.data[ii] ** 2) * t_ratio.data[ii])
        bi_max_temp = bi_max_temp * np.exp(
            -(z_p[ii, ...] * r_mat[ii, ...] - v_para.data[ii]) ** 2 / (vth_para.data[ii] ** 2))

        bi_max_dist[ii, ...] = bi_max_temp

    # Make modelPDist file for output
    model_vdf = vdf
    model_vdf.data = bi_max_dist
    model_vdf *= 1e18

    return model_vdf

