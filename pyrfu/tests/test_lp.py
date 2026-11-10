#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import unittest

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from .. import lp, pyrf
from ..lp.photo_current import surface_materials

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"


class ThermalCurrentTestCase(unittest.TestCase):
    def test_thermal_current_probe_type_case(self):
        # "Sphere" passed the check but gave the cylinder current (up to 25 %
        # different at 10 V)
        u = np.array([-10.0, 0.0, 10.0])
        args = (1e7, 1e5, constants.electron_mass, 0.0, -1, u, 1.0)
        sphere = lp.thermal_current(*args, "sphere")
        cylinder = lp.thermal_current(*args, "cylinder")
        self.assertFalse(np.allclose(sphere, cylinder))
        for p_type, expected in [("Sphere", sphere), ("CYLINDER", cylinder)]:
            np.testing.assert_array_equal(lp.thermal_current(*args, p_type), expected)

    # irfu-matlab lp.thermal_current(p_type, 1e6, 1e5, 9.109e-31, v, z, U, 0.1)
    # at U = -5, -1, 0, 1, 5 V (R2024a)
    MATLAB = {
        (-1, 0.0, "sphere"): [
            4.4049181257783492e-09,
            7.0069491856484543e-09,
            7.8691305405817698e-09,
            8.7823052201720137e-09,
            1.2435003938532988e-08,
        ],
        (-1, 4e5, "cylinder"): [
            4.6279783626038865e-09,
            7.1924311659661311e-09,
            8.0305781504977771e-09,
            8.7352564147953619e-09,
            1.0820765913406462e-08,
        ],
        (1, 0.0, "cylinder"): [
            1.0719091992905071e-08,
            8.5921602690457715e-09,
            7.8691305405817698e-09,
            7.0069491856484543e-09,
            4.4049181257783492e-09,
        ],
        (1, 4e5, "sphere"): [
            1.24565220034722e-08,
            8.915766921092661e-09,
            8.0305781504977771e-09,
            7.1924311659661311e-09,
            4.6279783626038865e-09,
        ],
    }

    def test_thermal_current_irfu_matlab(self):
        # Both charges and probe types, the body at rest and moving (V > 0.1
        # V_th), as irfu-matlab
        u = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        for (z, v, p_type), expected in self.MATLAB.items():
            result = lp.thermal_current(1e6, 1e5, 9.109e-31, v, z, u, 0.1, p_type)
            np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_thermal_current_no_plasma(self):
        # No density or temperature: None; no charge: no current
        args = (1e5, 9.109e-31, 0.0, -1, np.zeros(3), 0.1, "sphere")
        self.assertIsNone(lp.thermal_current(0.0, *args))
        self.assertIsNone(lp.thermal_current(1e6, 0.0, *args[1:]))
        for p_type in ["sphere", "cylinder"]:
            result = lp.thermal_current(
                1e6, 1e5, 9.109e-31, 0.0, 0, np.ones(3), 0.1, p_type
            )
            np.testing.assert_array_equal(result, 0.0)


class LangmuirProbeTestCase(unittest.TestCase):
    # irfu-matlab lp.lprobe (surface "cluster", R2024a): areas (sphere, wire,
    # total, sunlit, m^2) and capacitance (F)
    MATLAB = {
        "sphere": (
            {"r_sphere": 0.04},
            [0.020106192982974676, 0.0, 0.020106192982974676, 0.0050265482457436689],
            4.4506002217914817e-12,
        ),
        "wire": (
            {"r_wire": 1.1e-3, "l_wire": 50.0},
            [0.0, 0.34557519189487723, 0.34557519189487723, 0.11],
            2.6679310286914562e-10,
        ),
        "sphere+wire": (
            {"r_sphere": 0.04, "r_wire": 1.1e-3, "l_wire": 50.0},
            [
                0.020106192982974676,
                0.34557519189487723,
                0.3656813848778519,
                0.11502654824574367,
            ],
            2.7124370309093709e-10,
        ),
        "stazer": (
            {"r_wire": [1e-3, 2e-3], "l_wire": 10.0},
            [0.0, 0.094247779607693788, 0.094247779607693788, 0.03],
            6.5385723472730023e-11,
        ),
    }

    def test_langmuir_probe_irfu_matlab(self):
        # Type, areas, capacitance and photoemission as irfu-matlab (epsilon_0
        # of scipy): the wire area was 0, the wire capacitance raised
        # TypeError, and the photoemission was None
        for name, (kwargs, areas, capacitance) in self.MATLAB.items():
            probe = lp.LangmuirProbe(name, **kwargs)
            self.assertEqual(probe.probe_type, "wire" if name == "stazer" else name)
            result = [probe.area[k] for k in ["sphere", "wire", "total", "sunlit"]]
            np.testing.assert_allclose(result, areas, rtol=1e-15)
            self.assertAlmostEqual(probe.area["total_sunlit"], areas[2] / areas[3])
            np.testing.assert_allclose(probe.capacitance, capacitance, rtol=1e-8)
            self.assertEqual(probe.s_photoemission, 2.5000000000000005e-05)

    def test_langmuir_probe_cylinder(self):
        # Shorter than 10 radii: the capacitance of a cylinder of this length
        # (pyrf.estimate takes the half length; twice irfu-matlab, whose
        # formula is 51 % too low)
        probe = lp.LangmuirProbe("short", r_wire=0.01, l_wire=0.05)
        expected = pyrf.estimate("capacitance_cylinder", 0.01, 0.025)
        self.assertAlmostEqual(probe.capacitance, expected)
        np.testing.assert_allclose(probe.capacitance, 2 * 9.5124641410315939e-13)

        with self.assertRaisesRegex(ValueError, "length > radius"):
            lp.LangmuirProbe("disk", r_wire=0.01, l_wire=0.005)

    def test_langmuir_probe_inputs(self):
        # A stazer of two radii (the check compared the radii with 2), not
        # three; no probe (ZeroDivisionError): no type, NaN ratios; the
        # photoemission given
        with self.assertRaisesRegex(ValueError, "length 1"):
            lp.LangmuirProbe("p", r_wire=[1e-3, 2e-3, 3e-3], l_wire=10.0)

        probe = lp.LangmuirProbe("none")
        self.assertIsNone(probe.probe_type)
        self.assertTrue(np.isnan(probe.area["total_sunlit"]))
        self.assertEqual(probe.capacitance, 0.0)

        probe = lp.LangmuirProbe("p", "user defined", 0.04, s_photoemission=3e-5)
        self.assertEqual(probe.get_probe_surface_photoemission(), 3e-5)
        probe = lp.LangmuirProbe("p", "tin", 0.04)
        self.assertEqual(
            probe.s_photoemission,
            float(np.squeeze(lp.photo_current(1.0, 0.0, 1.0, "tin"))),
        )


class PhotoCurrentTestCase(unittest.TestCase):
    # Area 2 m^2 at 0.5 AU: 8 times the photocurrent of 1 m^2 at 1 AU
    area, distance = 2.0, 0.5
    u = np.array([-5.0, 0.0, 1.0])

    def test_photo_current_material(self):
        # THEMIS curve (50 uA/m^2 below 0.1 V, 27 uA/m^2 at 1 V) scaled by j0
        result = lp.photo_current(self.area, self.u, self.distance, "cluster")
        expected = 8.0 * 25e-6 * np.array([1.0, 1.0, 27.0 / 50.0])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_photo_current_material_case(self):
        expected = lp.photo_current(self.area, self.u, self.distance, "tin")

        for flag in ["TiN", "TIN", "Cluster"]:
            result = lp.photo_current(self.area, self.u, self.distance, flag)
            np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_photo_current_1ev(self):
        result = lp.photo_current(self.area, self.u, self.distance, "1eV")
        expected = 8.0 * 5e-5 * np.exp(-np.maximum(self.u, 0.0))
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_photo_current_photoemission(self):
        # Integer and float photoemissions in A/m^2
        factor = 5.0 / 5.6 + 1.2 / 5.6 * np.exp(-10.0 / 14.427)

        for flag in [2, 2.0]:
            result = lp.photo_current(self.area, self.u[:2], self.distance, flag)
            np.testing.assert_allclose(result, 8.0 * 2.0 * np.array([1.0, factor]))

    def test_photo_current_listing(self):
        with self.assertLogs("pyrfu.lp.photo_current", level="INFO") as logs:
            self.assertIsNone(lp.photo_current())

        self.assertEqual(len(logs.records), len(surface_materials))
        self.assertTrue(logs.records[0].getMessage().startswith("cluster: Io= 25.00"))

    def test_photo_current_invalid_flag(self):
        with self.assertRaises(TypeError):
            lp.photo_current(self.area, self.u, self.distance, ["cluster"])

        with self.assertRaises(ValueError):
            lp.photo_current(self.area, self.u, self.distance, "bazinga")


if __name__ == "__main__":
    unittest.main()
