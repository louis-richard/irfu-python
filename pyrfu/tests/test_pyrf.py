#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import unittest

# 3rd party imports
import numpy as np

from pyrfu import mms, pyrf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


class TestPyrf(unittest.TestCase):
    r"""Library test class"""

    def setUp(self):
        """integration test setup."""
        tint = ["2019-09-14T08:00:00.000", "2019-09-14T08:00:30.000"]
        self.b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, 2)
        self.e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, 2)

    def test_calc_fs(self):
        """integration test on sampling frequency computation"""
        fs = pyrf.calc_fs(self.b_xyz)

        self.assertEqual(np.round(fs), 128.0)

    def test_calc_dt(self):
        """integration test on sampling tme step computation"""
        dt = pyrf.calc_dt(self.b_xyz)

        self.assertEqual(int(dt * 1e9), 7813000)

    def test_cross(self):
        """integration test on vector cross product computation"""
        exb = pyrf.cross(self.e_xyz, self.b_xyz)
        res = pyrf.dot(exb, self.e_xyz)
        res = np.mean(res)
        self.assertTrue(res < 1e-5)

    def test_resample(self):
        """integration test on time resampling"""
        e_xyz = pyrf.resample(self.e_xyz, self.b_xyz)

        self.assertTrue(
            (pyrf.resample(e_xyz, self.b_xyz).time.data == self.b_xyz.time.data).all(),
        )


if __name__ == "__main__":
    unittest.main()
