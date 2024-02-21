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
__email__ = "louisr@irfu.se"
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


if __name__ == "__main__":
    unittest.main()
