#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt

# Local imports
from .. import dispersion

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.4"
__status__ = "Prototype"


class DispSurfCalcTestCase(unittest.TestCase):
    def test_disp_surf_calc_output(self):
        kx, kz, wf, extra_param = dispersion.disp_surf_calc(
            random.random(), random.random(), random.random(), random.random()
        )
        self.assertIsInstance(kx, np.ndarray)
        self.assertIsInstance(kz, np.ndarray)
        self.assertIsInstance(wf, np.ndarray)
        self.assertIsInstance(extra_param, dict)


@ddt
class OneFluidDispersionTestCase(unittest.TestCase):
    @data(
        (
            random.random(),
            random.random(),
            {"n": random.random(), "t": random.random(), "gamma": random.random()},
            {"n": random.random(), "t": random.random(), "gamma": random.random()},
            random.randint(10, 1000),
        )
    )
    def test_one_fluid_dispersion_output(self, value):
        b_0, theta, ions, electrons, n_k = value
        result = dispersion.one_fluid_dispersion(b_0, theta, ions, electrons, n_k)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)
        self.assertIsInstance(result[2], xr.DataArray)


if __name__ == "__main__":
    unittest.main()
