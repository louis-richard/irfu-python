#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import unittest

# 3rd party imports
import numpy as np

# Local imports
from .. import lp
from ..lp.photo_current import surface_materials

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"


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
