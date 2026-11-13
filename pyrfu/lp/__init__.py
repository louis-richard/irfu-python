#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import estimate
from .photo_current import photo_current
from .thermal_current import thermal_current

__all__ = ["LangmuirProbe", "photo_current", "thermal_current"]


class LangmuirProbe:
    r"""Defines either spherical, cylindrical, conical or spherical +
    cylindrical/conical probes, as irfu-matlab lp.lprobe. Probe belonging to
    LangmuirProbe is defined with properties."""

    def __init__(
        self,
        name: str,
        surface: str = "cluster",
        r_sphere: list | float | int | None = None,
        r_wire: list | float | int | np.ndarray | None = None,
        l_wire: list | float | int | None = None,
        s_photoemission: float | None = None,
    ):
        r"""Setup probe properties.

        Parameters
        ----------
        name : str
            Name of the probe.
        surface : str, Optional
            Surface material (see `photo_current`), or "user defined" with
            `s_photoemission`. Default is "cluster".
        r_sphere : float, Optional
            Radius of the sphere in m.
        r_wire : float or list, Optional
            Radius of the wire in m. Can be a float for wire or a list of two
            floats for a stazer (their mean is used).
        l_wire : float, Optional
            Length of the wire/stazer in m.
        s_photoemission : float, Optional
            Surface photoemission in A/m^2. If not given obtain from surface
            type.

        Raises
        ------
        ValueError
            If r_wire has more than 2 elements (an assertion compared the
            radii with 2 instead of their number).

        """

        if r_wire is not None and np.size(r_wire) > 2:
            raise ValueError(
                "The wire radius should be a numeric vector of length 1 (wire) "
                "or 2 (stazer)."
            )

        self.name = name
        r_wire = None if r_wire is None else np.atleast_1d(r_wire).astype(float)
        self.wire = {"l": l_wire, "r": r_wire}
        self.sphere = {"r": r_sphere, "surface": surface}
        self.s_photoemission = s_photoemission

        # Probe type according to the specified parameters: Sphere + Wire or
        # Wire or Sphere
        self.get_probe_type()

        # Get probe area
        self.get_probe_area()

        # Get probe capacitance
        self.get_probe_capa()

        # Get surface photoemission
        self.get_probe_surface_photoemission()

    def _has_sphere(self):
        return self.sphere["r"] is not None

    def _has_wire(self):
        return self.wire["r"] is not None and self.wire["l"] is not None

    def get_probe_type(self):
        r"""Define probe type according to the specified parameters."""
        if self._has_sphere() and self._has_wire():
            self.probe_type = "sphere+wire"
        elif self._has_wire():
            self.probe_type = "wire"
        elif self._has_sphere():
            self.probe_type = "sphere"
        else:
            self.probe_type = None

    def get_probe_area(self):
        r"""Computes probe area: of the sphere, of the wire, total, sunlit (in
        projection) and their ratios. The wire area was always zero (its
        radius, an array, was not recognised)."""
        a_sphere_sunlit, a_sphere_total = [0.0, 0.0]
        a_wire_sunlit, a_wire_total = [0.0, 0.0]

        if self._has_sphere():
            a_sphere_sunlit = np.pi * self.sphere["r"] ** 2
            a_sphere_total = 4 * a_sphere_sunlit

        if self._has_wire():
            a_wire_sunlit = 2 * np.mean(self.wire["r"]) * self.wire["l"]
            a_wire_total = np.pi * a_wire_sunlit

        self.area = {
            "sphere": a_sphere_total,
            "wire": a_wire_total,
            "total": a_sphere_total + a_wire_total,
            "sunlit": a_sphere_sunlit + a_wire_sunlit,
        }

        # NaN without probe (it raised ZeroDivisionError)
        with np.errstate(divide="ignore", invalid="ignore"):
            total, sunlit = np.float64(self.area["total"]), self.area["sunlit"]
            self.area["total_sunlit"] = total / sunlit
            self.area["sunlit_total"] = sunlit / total

    def get_probe_capa(self):
        r"""Computes probe capacitance: of the sphere, plus that of the wire,
        as a wire if longer than 10 radii, else as a cylinder (it raised
        TypeError for a wire, and for a probe without sphere).

        Raises
        ------
        ValueError : if the wire is not longer than its radius.

        See Also
        --------
        pyrfu.pyrf.estimate

        Notes
        -----
        The capacitance of a cylinder is that of `pyrfu.pyrf.estimate`, which
        corrects the formula of irfu-matlab (51 % too low here).

        """

        c_sphere, c_wire = 0.0, 0.0

        if self._has_sphere():
            c_sphere = estimate("capacitance_sphere", self.sphere["r"])

        if self._has_wire():
            r_wire, l_wire = np.mean(self.wire["r"]), self.wire["l"]

            if np.all(l_wire > 10 * self.wire["r"]):
                c_wire = estimate("capacitance_wire", r_wire, l_wire)
            elif np.all(l_wire > self.wire["r"]):
                # (pyrf.estimate takes the half length)
                c_wire = estimate("capacitance_cylinder", r_wire, l_wire / 2)
            else:
                raise ValueError(
                    "estimate of capacitance for cylinder requires length > radius",
                )

        self.capacitance = c_sphere + c_wire

    def get_probe_surface_photoemission(self):
        r"""Computes (or get) surface photo emission (A/m^2): the one given,
        else that of the surface material (the condition was inverted)."""
        if self.sphere["surface"] != "user defined" and self.s_photoemission is None:
            j_photo = photo_current(1.0, 0.0, 1.0, self.sphere["surface"])
            self.s_photoemission = float(np.squeeze(j_photo))

        return self.s_photoemission
