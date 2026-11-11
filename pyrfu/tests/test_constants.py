#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import unittest

# Local imports
from .. import constants


class ConstantsTestCase(unittest.TestCase):
    """Tests of the values in pyrfu.constants."""

    def test_earth_radius(self):
        """R_E is the IGRF reference radius in km, as irf_units in irfu-matlab."""
        self.assertEqual(constants.R_E, 6371.2)


if __name__ == "__main__":
    unittest.main()
