#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import random
import unittest

# 3rd party imports
import numpy as np
from ddt import data, ddt, unpack

# Local imports
from .. import solo

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.4"
__status__ = "Prototype"


class DbInitTestCase(unittest.TestCase):
    def test_db_init_inpput(self):
        with self.assertRaises(AssertionError):
            solo.db_init("/Volumes/solo/remote/data")

    def test_db_init_output(self):
        self.assertIsNone(solo.db_init(os.getcwd()))


@ddt
class ReadLFRDensityTestCase(unittest.TestCase):
    @data(
        ([], ".", False),
        ([np.datetime64("2023-01-01T00:00:00"), "2023-01-01T00:10:00"], ".", False),
        (["2023-01-01T00:00:00", np.datetime64("2023-01-01T00:10:00")], ".", False),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], "/bazinga", False),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], ".", "i am groot"),
    )
    @unpack
    def test_read_lfr_density_input(self, tint, data_path, tree):
        with self.assertRaises(AssertionError):
            solo.read_lfr_density(tint, data_path, tree)

    def test_read_lfr_density_output(self):
        tint = ["2023-01-01T00:00:00", "2023-01-01T00:10:00"]
        self.assertIsNone(solo.read_lfr_density(tint))


@ddt
class ReadTNRTestCase(unittest.TestCase):
    @data(
        ([], 1, "."),
        ([np.datetime64("2023-01-01T00:00:00"), "2023-01-01T00:10:00"], 1, "."),
        (["2023-01-01T00:00:00", np.datetime64("2023-01-01T00:10:00")], 1, "."),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], random.random(), "."),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 1, "/bazinga"),
    )
    @unpack
    def test_read_tnr_input(self, tint, sensor, data_path):
        with self.assertRaises(AssertionError):
            solo.read_tnr(tint, sensor, data_path)

    @data(
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 1, ""),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 2, ""),
    )
    @unpack
    def test_read_tnr_output(self, tint, sensor, data_path):
        self.assertIsNone(solo.read_tnr(tint, sensor, data_path))


if __name__ == "__main__":
    unittest.main()
