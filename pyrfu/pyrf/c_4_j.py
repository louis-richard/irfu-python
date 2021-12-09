#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from scipy import constants

# Local imports
from .avg_4sc import avg_4sc
from .c_4_grad import c_4_grad
from .cross import cross

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.10"
__status__ = "Prototype"


def c_4_j(r_list, b_list):
    r"""Calculate current density :math:`J` from using 4
    spacecraft technique [4]_, the divergence of the magnetic field
    :math:`\nabla . B`, magnetic field at the center of
    mass of the tetrahedron, :math:`J \times B`
    force, part of the divergence of stress associated with
    curvature :math:`\nabla.T_{shear}` and gradient of
    the magnetic pressure :math:`\nabla P_b`. Where :

    .. math::

        J = \frac{\nabla \times B}{\mu_0}

        J \times B = \nabla.T_{shear} + \nabla P_b

        \nabla.T_{shear} = \frac{(B.\nabla) B}{\mu_0}

        \nabla P_b = \nabla \frac{B^2}{2\mu_0}

    The divergence of the magnetic field is current density units as
    it shows the error on the estimation of the current density [5]_ .

    Parameters
    ----------
    r_list : list of xarray.DataArray
        Time series of the spacecraft position [km].
    b_list : list of xarray.DataArray
        Time series of the magnetic field [nT].

    Returns
    -------
    j : xarray.DataArray
        Time series of the current density [A.m^{-2}].
    div_b : xarray.DataArray
        Time series of the divergence of the magnetic field [A.m^{-2}].
    b_avg : xarray.DataArray
        Time series of the magnetic field at the center of mass of the
        tetrahedron, sampled at 1st SC time steps [nT].
    jxb : xarray.DataArray
        Time series of the :math:`J\timesB` force
        [T.A].
    div_t_shear : xarray.DataArray
        Time series of the part of the divergence of stress associated
        with curvature units [T A/m^2].
    div_pb : xarray.DataArray
        Time series of the gradient of the magnetic pressure.

    See also
    --------
    pyrfu.pyrf.c_4_k : Calculates reciprocal vectors in barycentric
                            coordinates.
    pyrfu.pyrf.c_4_grad : Calculate gradient of physical field using 4
                            spacecraft technique.

    References
    ----------
    .. [4]	Dunlop, M. W., A. Balogh, K.-H. Glassmeier, and P. Robert
            (2002a), Four-point Cluster application of magnetic field
            analysis tools: The Curl- ometer, J. Geophys. Res.,
            107(A11), 1384, doi : https://doi.org/10.1029/2001JA005088.

    .. [5]	Robert, P., et al. (1998), Accuracy of current determination,
            in Analysis Methods for Multi-Spacecraft Data, edited by G.
            Paschmann and P. W. Daly, pp. 395–418, Int. Space Sci. Inst.,
            Bern. url : http://www.issibern.ch/forads/sr-001-16.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_list = np.arange(1,5)

    Load magnetic field and spacecraft position

    >>> b_mms = [mms.get_data("B_gse_fgm_srvy_l2", tint, i) for i in mms_list]
    >>> r_mms = [mms.get_data("R_gse", tint, i) for i in mms_list]

    Compute current density, divergence of b, etc. using curlometer technique

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms,
    b_mms)

    """

    mu0 = constants.mu_0

    b_avg = 1e-9 * avg_4sc(b_list)

    # Estimate divB/mu0. unit is A/m2
    div_b = c_4_grad(r_list, b_list, "div")

    # to get right units
    div_b *= 1.0e-3 * 1e-9 / mu0

    # estimate current j [A/m2]
    j = c_4_grad(r_list, b_list, "curl")
    j.data *= 1.0e-3 * 1e-9 / mu0

    # estimate jxB force [T A/m2]
    jxb = cross(j, b_avg)

    # estimate divTshear = (1/muo) (B*div)B [T A/m2]
    div_t_shear = c_4_grad(r_list, b_list, "bdivb")
    div_t_shear.data *= 1.0e-3 * 1e-9 * 1e-9 / mu0

    # estimate divPb = (1/muo) grad (B^2/2) = divTshear-jxB
    div_pb = div_t_shear.copy()
    div_pb.data -= jxb.data

    return j, div_b, b_avg, jxb, div_t_shear, div_pb
