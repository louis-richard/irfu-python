#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np

from scipy import constants

# Local imports
from ..pyrf import estimate
from .photo_current import photo_current
from .thermal_current import thermal_current

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


class Plasma(object):
    r"""Describes plasma model consisting of several plasma components where
    each component is characterized by charge size and sign, density, mass of
    particles, temperature and drift velocity."""

    def __init__(self, name: str = "", n: Union[float, list] = None,
                 mp: Union[float, list] = None, qe: Union[int, list] = None,
                 t: Union[float, list] = None):
        r"""Setup plasma properties. They can be a single number applicable to
        all plasma components or a vector of the length equal to the number of
        plasma components.

        Parameters
        ----------
        name : str
            Name of the plasma.
        n : float or list
            Species number densities.
        mp : float or list
            Species masses in terms of proton mass.
        qe : float or list
            Species charges in terms of elementary charge.
        t : float or float
            Species temperatures.

        """

        self.name = name
        self.n = np.atleast_1d(n)
        self.mp = np.atleast_1d(mp)
        self.qe = np.atleast_1d(qe)
        self.t = np.atleast_1d(t)

        # Computes mass and charge in SI units
        self._m()
        self._q()

    def _m(self):
        self.m = self.mp * constants.proton_mass
        self.m[self.m == 0] = constants.electron_mass

    def _q(self):
        self.qe * constants.elementary_charge


class LangmuirProbe(object):
    r"""Defines either spherical, cylindrical, conical or spherical +
    cylindrical/conical probes. Probe belonging to LangmuirProbe is defined
    with properties."""

    def __init__(self, name: str, surface: str = "cluster",
                 r_sphere: Union[list, float, int] = None,
                 r_wire: Union[list, float, int] = None,
                 l_wire: Union[list, float, int] = None,
                 s_photoemission: float = None):
        r"""Setup probe properties.

        Parameters
        ----------
        name : str
            Name of the probe.
        surface : str, Optional
            Surface materials or surface photoemission in [A/m^2].
            Default is "cluster".
        r_sphere : float, Optional
            Radius of the sphere in m.
        r_wire : float or list, Optional
            Radius of the wire in m. Can be a float for wire or a list of two
            floats for a stazer.
        l_wire : float, Optional
            Length of the wire/stazer in m.
        s_photoemission : float, Optional
            Surface photoemission in A/m^2. If not given obtain from surface
            type.

        Raises
        ------
        AssertionError : if r_wire as more than 2 elements.

        """

        message = "The wire radius should be a numeric vector of " \
                  "length 1 (wire) or 2 (stazer)."
        assert not r_wire or np.atleast_1d(r_wire) <= 2, message

        self.name = name
        self.surface = surface
        self.r_sphere = r_sphere
        self.r_wire = np.atleast_1d(r_wire)
        self.l_wire = l_wire
        self.s_photoemission = s_photoemission

        # Probe type according to the specified parameters: Sphere + Wire or
        # Wire or Sphere
        self._get_probe_type()

        # Get probe area
        self._get_probe_area()

        # Get probe capacitance
        self._get_probe_capa()

    def _get_probe_type(self):
        r"""Define probe type according to the specified parameters."""
        if self.r_sphere and self.r_wire and self.l_wire:
            self.probe_type = "sphere+wire"
        elif self.r_wire and self.l_wire:
            self.probe_type = "wire"
        elif self.r_sphere:
            self.probe_type = "sphere"
        else:
            self.probe_type = None

    def _get_probe_area(self):
        r"""Computes probe area."""
        a_sphere_sunlit, a_sphere_total = [0, 0]
        a_wire_sunlit, a_wire_total = [0, 0]

        if isinstance(self.r_sphere, (float, int)):
            a_sphere_sunlit = np.pi * self.r_sphere ** 2
            a_sphere_total = 4 * a_sphere_sunlit

        if self.r_wire and isinstance(self.r_wire, (float, int, list)) \
                and self.l_wire and isinstance(self.l_wire, (float, int)):
            a_wire_sunlit = 2 * np.mean(self.r_wire) * self.l_wire
            a_wire_total = np.pi * a_wire_sunlit

        self.area = {"sphere": a_sphere_total, "wire": a_wire_total,
                     "total": a_sphere_total + a_wire_total,
                     "sunlit": a_sphere_sunlit + a_wire_sunlit}
        self.area["total_sunlit"] = self.area["total"] / self.area["sunlit"]
        self.area["sunlit_total"] = 1 / self.area["total_sunlit"]

    def _get_probe_capa(self):
        r"""Computes probe capacitance.

        Raises
        ------
        ValueError : if length > radius.

        See Also
        --------
        pyrfu.pyrf.estimate.py

        """

        c_wire = 0
        c_sphere = estimate("capacitance_sphere", self.r_sphere)

        if self.r_wire and isinstance(self.r_wire, (float, int, list)) \
                and self.l_wire and isinstance(self.l_wire, (float, int)):
            if self.l_wire > 10 * list([self.r_wire]):
                c_wire = estimate("capacitance_wire", np.mean(self.r_wire),
                                  self.l_wire)
            elif self.l_wire > list(self.r_wire):
                c_wire = estimate("capacitance_cylinder", np.mean(self.r_wire),
                                  self.l_wire)
            else:
                raise ValueError("estimate of capacitance for cylinder "
                                 "requires length > radius")

        self.capacitance = np.sum([c_sphere, c_wire])

    def _get_probe_surface_photoemission(self):
        r"""Computes (or get) surface photo emission."""
        if self.surface:
            self.s_photoemission = self.s_photoemission
        else:
            self.s_photoemission = photo_current(1., 0., 1., self.surface)
