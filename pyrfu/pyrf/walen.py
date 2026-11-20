#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from pyrfu._core.units import _check_units

from .pres_anis import pres_anis
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def walen(v_xyz, b_xyz, n_i, v_ht, p_fac=None):
    r"""Walén test: linear fit of the plasma velocity in the de Hoffmann-Teller
    frame against the Alfvén velocity, as irfu-matlab ``irf_walen``.

    .. math::

        \mathbf{V} - \mathbf{V}_{HT} = s \, \mathbf{V}_A \sqrt{1 - \alpha}
        + c, \quad \mathbf{V}_A = \frac{\mathbf{B}}{\sqrt{\mu_0 n_i m_p}},
        \quad \alpha = \mu_0 \frac{P_\parallel - P_\perp}{B^2}

    Across a rotational discontinuity the plasma flows along the field at the
    Alfvén velocity in the de Hoffmann-Teller frame: the slope s is +1 or -1
    (flow along or against B) and the correlation coefficient close to +1 or
    -1 [1]_ [2]_.

    Parameters
    ----------
    v_xyz : xarray.DataArray
        Time series of the ion bulk velocity [km/s].
    b_xyz : xarray.DataArray
        Time series of the magnetic field [nT], in the coordinate system of
        `v_xyz`, resampled to its times.
    n_i : xarray.DataArray
        Time series of the ion number density [cm^-3], resampled to the times
        of `v_xyz`. The mass density is that of protons.
    v_ht : array_like
        De Hoffmann-Teller velocity [km/s] (3 values, e.g. from `vht`), in the
        coordinate system of `v_xyz`.
    p_fac : xarray.DataArray, Optional
        Time series of the pressure tensor in field-aligned coordinates
        (parallel first) [nPa], as `pres_anis`: that of the ions, or the sum of
        those of the ions and electrons. Default is None: no anisotropy
        correction (:math:`\alpha = 0`).

    Returns
    -------
    slope : float
        Slope s of the fit of :math:`\mathbf{V} - \mathbf{V}_{HT}` against
        :math:`\mathbf{V}_A \sqrt{1 - \alpha}`.
    offset : float
        Offset c of the fit [km/s].
    cc : float
        Correlation coefficient of :math:`\mathbf{V} - \mathbf{V}_{HT}` and
        :math:`\mathbf{V}_A \sqrt{1 - \alpha}`.
    v_tht : xarray.DataArray
        Time series of the velocity in the de Hoffmann-Teller frame,
        :math:`\mathbf{V} - \mathbf{V}_{HT}` [km/s], on the times of `v_xyz`.
    v_a : xarray.DataArray
        Time series of the Alfvén velocity, :math:`\mathbf{V}_A \sqrt{1 -
        \alpha}` with `p_fac` [km/s], on the times of `v_xyz`.

    Raises
    ------
    ValueError
        If `v_ht` is not 3 values, or if no sample has both velocities.

    Notes
    -----
    The fit and the correlation use the three components of the samples
    together, as irfu-matlab, where both velocities are finite: the samples
    with :math:`\alpha \geq 1` (firehose unstable) have no Alfvén velocity
    (NaN) and are left out. Unlike irfu-matlab, the samples with a density
    below 0.6 cm^-3 are kept and the anisotropy is not smoothed (filter the
    inputs, e.g. with `time_clip` or `movmean`), and the constants are exact:
    the factor 22 of irfu-matlab gives an Alfvén velocity 0.87 % higher.

    References
    ----------
    .. [1]  Paschmann, G., Papamastorakis, I., Baumjohann, W., Sckopke, N.,
            Carlson, C. W., Sonnerup, B. U. Ö., & Lühr, H. (1986). The
            magnetopause for large magnetic shear: AMPTE/IRM observations,
            Journal of Geophysical Research, 91(A10), 11099-11115.
            https://doi.org/10.1029/JA091iA10p11099

    .. [2]  Paschmann, G., and Sonnerup, B. U. Ö. (2008), Proper frame
            determination and Walén test, in Multi-Spacecraft Analysis Methods
            Revisited, edited by G. Paschmann and P. W. Daly, ISSI SR-008, pp.
            65-74, Int. Space Sci. Inst., Bern.

    See Also
    --------
    pyrfu.pyrf.vht : De Hoffmann-Teller velocity.
    pyrfu.pyrf.pres_anis : Pressure anisotropy factor.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-16T10:33:20.000", "2015-10-16T10:34:00.000"]

    Magnetic field, electric field, ion bulk velocity, density and pressure

    >>> b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint, 1)
    >>> e_xyz = mms.get_data("e_gse_edp_brst_l2", tint, 1)
    >>> v_xyz_i = mms.get_data("vi_gse_fpi_brst_l2", tint, 1)
    >>> n_i = mms.get_data("ni_fpi_brst_l2", tint, 1)
    >>> p_xyz_i = mms.get_data("pi_gse_fpi_brst_l2", tint, 1)

    De Hoffmann-Teller velocity and Walén test with the anisotropy correction

    >>> v_ht, _, _ = pyrf.vht(e_xyz, b_xyz)
    >>> p_fac_i = mms.rotate_tensor(p_xyz_i, "fac", b_xyz)
    >>> slope, offset, cc, v_tht, v_a = pyrf.walen(
    ...     v_xyz_i, b_xyz, n_i, v_ht, p_fac_i
    ... )

    """

    # The factors assume V in km/s, B in nT and n in cm^-3
    _check_units(v_xyz, "km/s", "v_xyz")
    _check_units(b_xyz, "nT", "b_xyz")
    _check_units(n_i, "cm^-3", "n_i")

    v_ht = np.asarray(v_ht, dtype=np.float64)

    if v_ht.shape != (3,):
        raise ValueError("v_ht must be a vector (3 values)")

    time = v_xyz.time.data
    b_data = resample(b_xyz, v_xyz).data
    n_data = resample(n_i, v_xyz).data

    # Alfven velocity, km/s
    rho = 1e6 * n_data * constants.proton_mass
    v_a = 1e-12 * b_data / np.sqrt(constants.mu_0 * rho)[:, None]

    if p_fac is not None:
        # The anisotropy on the times of the pressure, then of the velocity
        alpha = resample(pres_anis(p_fac, b_xyz), v_xyz).data

        with np.errstate(invalid="ignore"):
            v_a *= np.sqrt(1.0 - alpha)[:, None]

    v_tht = v_xyz.data - v_ht

    valid = np.all(np.isfinite(v_tht), axis=1) & np.all(np.isfinite(v_a), axis=1)

    if not np.any(valid):
        raise ValueError("no sample has both the velocity and Alfven velocity")

    # Fit and correlation of the three components together
    v_a_fit = v_a[valid].ravel()
    v_tht_fit = v_tht[valid].ravel()
    slope, offset = np.polyfit(v_a_fit, v_tht_fit, 1)
    cc = np.corrcoef(v_a_fit, v_tht_fit)[0, 1]

    v_tht = ts_vec_xyz(time, v_tht, {"UNITS": "km/s"})
    v_a = ts_vec_xyz(time, v_a, {"UNITS": "km/s"})

    return float(slope), float(offset), float(cc), v_tht, v_a
