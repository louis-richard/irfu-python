#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

# 3rd party imports
import numpy as np
from ddt import data, ddt, unpack

# Local imports
from .. import models
from . import generate_timeline

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.4"
__status__ = "Prototype"


@ddt
class IgrfTestCase(unittest.TestCase):
    @data(
        (
            generate_timeline(64.0, 100, ref_time="1789-07-14T00:00:00.000000000"),
            "dipole",
        ),
    )
    @unpack
    def test_igrf_input_time(self, timeline, flag):
        with self.assertWarns(UserWarning):
            models.igrf(timeline.astype(np.int64) / 1e9, flag)

    @data((generate_timeline(64.0, 100), "bazinga!"))
    @unpack
    def test_igrf_input_flag(self, timeline, flag):
        with self.assertRaises(NotImplementedError):
            models.igrf(timeline.astype(np.int64) / 1e9, flag)

    @data((generate_timeline(64.0, 100), "dipole"))
    @unpack
    def test_igrf_output(self, timeline, flag):
        result = models.igrf(timeline.astype(np.int64) / 1e9, flag)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertListEqual(list(result[0].shape), list(timeline.shape))
        self.assertIsInstance(result[1], np.ndarray)
        self.assertListEqual(list(result[1].shape), list(timeline.shape))


@ddt
class MagnetopauseNormalTestCase(unittest.TestCase):
    @data(
        (np.random.rand(3), random.randint(1, 10), random.randint(1, 10), "bazinga!!")
    )
    @unpack
    def test_magnetopause_normal_input(self, r_gsm, b_z_imf, p_sw, model):
        with self.assertRaises(NotImplementedError):
            models.magnetopause_normal(r_gsm, b_z_imf, p_sw, model)

    @data(
        (np.random.rand(3), random.randint(1, 10), random.randint(1, 10), "mp_shue97"),
        (np.random.rand(3), random.randint(1, 10), random.randint(1, 10), "bs97"),
        (np.random.rand(3), -random.randint(1, 10), random.randint(1, 10), "bs97"),
        (np.random.rand(3), random.randint(1, 10), random.randint(1, 10), "mp_shue98"),
        (np.random.rand(3), random.randint(1, 10), random.randint(1, 10), "bs98"),
    )
    @unpack
    def test_magnetopause_normal_output(self, r_gsm, b_z_imf, p_sw, model):
        models.magnetopause_normal(r_gsm, b_z_imf, p_sw, model)

    @data(
        ([8.0, 0.0, 5.0], "mp_shue1997", 1.056268, [0.931547, 0.0, 0.363620]),
        ([8.0, 0.0, 5.0], "MP_SHUE97", 1.056268, [0.931547, 0.0, 0.363620]),
        ([8.0, 0.0, 5.0], "mp_shue1998", 1.185722, [0.931236, 0.0, 0.364416]),
        ([8.0, 0.0, 5.0], "mp_shue98", 1.185722, [0.931236, 0.0, 0.364416]),
        ([-20.0, 15.0, 0.0], "mp_shue1997", 7.173864, [0.220119, 0.975473, 0.0]),
        ([-60.0, 25.0, 0.0], "mp_shue1997", 2.614401, [0.087786, 0.996139, 0.0]),
    )
    @unpack
    def test_magnetopause_normal_values(self, r_gsm, model, min_dist, n_vec):
        result = models.magnetopause_normal(np.array(r_gsm), -2.0, 2.0, model)
        self.assertAlmostEqual(result[0], min_dist, places=5)
        np.testing.assert_allclose(result[1], n_vec, atol=1e-5)

    def test_magnetopause_normal_default_model(self):
        result = models.magnetopause_normal(np.array([8.0, 0.0, 5.0]), -2.0, 2.0)
        self.assertAlmostEqual(result[0], 1.056268, places=5)

    def test_magnetopause_normal_axisymmetric(self):
        # The model is symmetric about the x axis: the distance depends only
        # on x and sqrt(y**2 + z**2), and the normal is a unit vector.
        for r_gsm in [[8.0, 5.0, 0.0], [8.0, 0.0, 5.0], [8.0, 3.0, -4.0]]:
            result = models.magnetopause_normal(np.array(r_gsm), -2.0, 2.0)
            self.assertAlmostEqual(result[0], 1.056268, places=5)
            self.assertAlmostEqual(np.linalg.norm(result[1]), 1.0, places=12)

    def test_magnetopause_normal_bow_shock_aliases(self):
        r_gsm = np.array([20.0, 3.0, 4.0])
        result_bs = models.magnetopause_normal(r_gsm, -2.0, 2.0, "bs")
        result_bs97 = models.magnetopause_normal(r_gsm, -2.0, 2.0, "bs97")
        self.assertEqual(result_bs[0], result_bs97[0])
        np.testing.assert_array_equal(result_bs[1], result_bs97[1])


if __name__ == "__main__":
    unittest.main()
