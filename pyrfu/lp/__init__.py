#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import estimate
from .photo_current import photo_current
from .thermal_current import thermal_current

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["LangmuirProbe", "photo_current", "thermal_current"]


class LangmuirProbe:
    r"""Defines either spherical, cylindrical, conical or spherical +
    cylindrical/conical probes. Probe belonging to LangmuirProbe is defined
    with properties."""

    def __init__(
        self,
        name: str,
        surface: str = "cluster",
        r_sphere: Union[list, float, int] = None,
        r_wire: Union[list, float, int] = None,
        l_wire: Union[list, float, int] = None,
        s_photoemission: float = None,
    ):
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

        message = (
            "The wire radius should be a numeric vector of "
            "length 1 (wire) or 2 (stazer)."
        )
        assert not r_wire or np.atleast_1d(r_wire) <= 2, message

        self.name = name
        self.wire = {"l": l_wire, "r": np.atleast_1d(r_wire)}
        self.sphere = {"r": r_sphere, "surface": surface}
        self.s_photoemission = s_photoemission

        # Probe type according to the specified parameters: Sphere + Wire or
        # Wire or Sphere
        self.get_probe_type()

        # Get probe area
        self.get_probe_area()

        # Get probe capacitance
        self.get_probe_capa()

    def get_probe_type(self):
        r"""Define probe type according to the specified parameters."""
        if self.sphere["r"] and self.wire["r"] and self.wire["l"]:
            self.probe_type = "sphere+wire"
        elif self.wire["r"] and self.wire["l"]:
            self.probe_type = "wire"
        elif self.sphere["r"]:
            self.probe_type = "sphere"
        else:
            self.probe_type = None

    def get_probe_area(self):
        r"""Computes probe area."""
        a_sphere_sunlit, a_sphere_total = [0, 0]
        a_wire_sunlit, a_wire_total = [0, 0]

        if isinstance(self.sphere["r"], (float, int)):
            a_sphere_sunlit = np.pi * self.sphere["r"] ** 2
            a_sphere_total = 4 * a_sphere_sunlit

        if (
            self.wire["r"]
            and isinstance(self.wire["r"], (float, int, list))
            and self.wire["l"]
            and isinstance(self.wire["l"], (float, int))
        ):
            a_wire_sunlit = 2 * np.mean(self.wire["r"]) * self.wire["l"]
            a_wire_total = np.pi * a_wire_sunlit

        self.area = {
            "sphere": a_sphere_total,
            "wire": a_wire_total,
            "total": a_sphere_total + a_wire_total,
            "sunlit": a_sphere_sunlit + a_wire_sunlit,
        }
        self.area["total_sunlit"] = self.area["total"] / self.area["sunlit"]
        self.area["sunlit_total"] = 1 / self.area["total_sunlit"]

    def get_probe_capa(self):
        r"""Computes probe capacitance.

        Raises
        ------
        ValueError : if length > radius.

        See Also
        --------
        pyrfu.pyrf.estimate.py

        """

        c_wire = 0
        c_sphere = estimate("capacitance_sphere", self.sphere["r"])

        if (
            self.wire["r"]
            and isinstance(self.wire["r"], (float, int, list))
            and self.wire["l"]
            and isinstance(self.wire["l"], (float, int))
        ):
            if self.wire["l"] > 10 * list([self.wire["r"]]):
                c_wire = estimate(
                    "capacitance_wire",
                    np.mean(self.wire["r"]),
                    self.wire["l"],
                )
            elif self.wire["l"] > list(self.wire["r"]):
                c_wire = estimate(
                    "capacitance_cylinder",
                    np.mean(self.wire["r"]),
                    self.wire["l"],
                )
            else:
                raise ValueError(
                    "estimate of capacitance for cylinder requires length > radius",
                )

        self.capacitance = np.sum([c_sphere, c_wire])

    def get_probe_surface_photoemission(self):
        r"""Computes (or get) surface photo emission."""
        if self.sphere["surface"]:
            self.s_photoemission = self.s_photoemission
        else:
            self.s_photoemission = photo_current(1.0, 0.0, 1.0, self.sphere["surface"])
