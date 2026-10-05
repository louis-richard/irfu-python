#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
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

    def test_magnetopause_normal_bow_shock_values(self):
        # The normal is rotated about the x axis with the position, including
        # in the z = 0 plane and on the x axis.
        n_vec_0 = [0.99479552, 0.10189149]
        for r_gsm, n_vec in [
            ([20.0, 0.0, 3.0], [n_vec_0[0], 0.0, n_vec_0[1]]),
            ([20.0, 3.0, 0.0], [n_vec_0[0], n_vec_0[1], 0.0]),
            ([20.0, 0.0, -3.0], [n_vec_0[0], 0.0, -n_vec_0[1]]),
            ([20.0, -1.8, 2.4], [n_vec_0[0], -0.6 * n_vec_0[1], 0.8 * n_vec_0[1]]),
        ]:
            result = models.magnetopause_normal(np.array(r_gsm), -2.0, 2.0, "bs")
            self.assertAlmostEqual(result[0], -6.654952, places=5)
            np.testing.assert_allclose(result[1], n_vec, atol=1e-7)

        result = models.magnetopause_normal(np.array([20.0, 0.0, 0.0]), -2.0, 2.0, "bs")
        self.assertAlmostEqual(result[0], -6.501315, places=5)
        np.testing.assert_allclose(result[1], [1.0, 0.0, 0.0], atol=1e-7)


@ddt
class IonAnisotropyThreshTestCase(unittest.TestCase):
    @data(
        ("proton-cyclotron", "10^-2", 1.0, 1.649),
        ("mirror", "10^-2", 1.0, 1.0 + 1.040 / 1.012**0.633),
        ("parallel-firehose", "10^-2", 1.0, 1.0 - 0.647 / 0.287**0.583),
        ("oblique-firehose", "10^-2", 1.0, 1.0 - 1.447 / 1.148),
        ("proton-cyclotron", "10^-3", 1.0, 1.0 + 0.437 / 1.003**0.428),
        ("mirror", "10^-4", 1.0, 1.0 + 0.702 / 1.009**0.674),
    )
    @unpack
    def test_ion_anisotropy_thresh_values(self, instability, growth, beta, ref):
        result = models.ion_anisotropy_thresh(beta, instability, growth)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, ref, places=12)

    def test_ion_anisotropy_thresh_input_unchanged(self):
        beta = np.array([0.1, 0.5, 1.0, 10.0])
        result = models.ion_anisotropy_thresh(beta, "parallel-firehose")
        np.testing.assert_array_equal(beta, [0.1, 0.5, 1.0, 10.0])
        self.assertTrue(np.all(np.isnan(result[:2])))
        self.assertTrue(np.all(np.isfinite(result[2:])))

    def test_ion_anisotropy_thresh_undefined(self):
        # NaN (not inf) where beta <= beta0
        result = models.ion_anisotropy_thresh(
            np.array([0.0, 0.713]), "proton-cyclotron"
        )
        self.assertTrue(np.isnan(result[0]))
        result = models.ion_anisotropy_thresh(0.713, "parallel-firehose")
        self.assertTrue(np.isnan(result))

    def test_ion_anisotropy_thresh_types(self):
        result = models.ion_anisotropy_thresh(np.array([1, 2]), "proton-cyclotron")
        np.testing.assert_allclose(result, [1.649, 1.0 + 0.649 / 2**0.4])

        time = generate_timeline(1.0, 4)
        beta = xr.DataArray([0.5, 1.0, 2.0, 4.0], coords=[time], dims=["time"])
        result = models.ion_anisotropy_thresh(beta, "proton-cyclotron")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.time.data, time)
        self.assertAlmostEqual(float(result[1]), 1.649, places=12)

    @data(("mirror", "10^-5"), ("bazinga!", "10^-2"))
    @unpack
    def test_ion_anisotropy_thresh_input(self, instability, growth):
        with self.assertRaises(ValueError):
            models.ion_anisotropy_thresh(1.0, instability, growth)


if __name__ == "__main__":
    unittest.main()
