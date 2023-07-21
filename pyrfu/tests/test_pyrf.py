#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import unittest

# 3rd party imports
import numpy as np
import xarray

from pyrfu import pyrf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def generate_timeline(f_s, n_pts: int = 10000):
    ref_time = np.datetime64("2019-01-01T00:00:00.000")
    times = [ref_time + np.timedelta64(int(i * 1e9 / f_s), "ns") for i in range(n_pts)]
    return np.array(times)


def generate_data(n_pts, kind: str = "scalar"):
    if kind == "scalar":
        data = np.random.random((n_pts,))
    elif kind == "vector":
        data = np.random.random((n_pts, 3))
    elif kind == "tensor":
        data = np.random.random((n_pts, 3, 3))
    else:
        raise ValueError("Invalid kind of data!!")

    return data


def generate_ts(f_s, n_pts, kind: str = "scalar"):
    if kind == "scalar":
        out = pyrf.ts_scalar(generate_timeline(f_s, n_pts), generate_data(n_pts, kind))
    elif kind == "vector":
        out = pyrf.ts_vec_xyz(generate_timeline(f_s, n_pts), generate_data(n_pts, kind))
    elif kind == "tesnor":
        out = pyrf.ts_tensor_xyz(
            generate_timeline(f_s, n_pts), generate_data(n_pts, kind)
        )
    else:
        raise ValueError("Invalid kind of data!!")

    return out


def generate_ts_scalar(f_s, n_pts):
    return pyrf.ts_scalar(generate_timeline(f_s, n_pts), generate_data(n_pts, "vector"))


class TsScalarTestCase(unittest.TestCase):
    def test_ts_scalar_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.ts_scalar(0, 0)
            pyrf.ts_scalar(
                list(generate_timeline(64.0, 100)), list(generate_data(100, "scalar"))
            )

    def test_ts_scalar_input_shape(self):
        with self.assertRaises(AssertionError):
            # Raises error if data and timeline don't have the same size
            pyrf.ts_scalar(generate_timeline(64.0, 100), generate_data(99, "scalar"))
            # Raises error if vector as input
            pyrf.ts_scalar(generate_timeline(64.0, 100), generate_data(100, "vector"))
            # Raises error if tensor as input
            pyrf.ts_scalar(generate_timeline(64.0, 100), generate_data(100, "tensor"))

    def test_ts_scalar_output_type(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, "scalar")
        )
        self.assertIsInstance(result, xarray.DataArray)

    def test_ts_scalar_output_shape(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, "scalar")
        )
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_ts_scalar_dims(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, "scalar")
        )
        self.assertListEqual(list(result.dims), ["time"])

    def test_ts_scalar_meta(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, "scalar")
        )
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class TsVecXYZTestCase(unittest.TestCase):
    def test_ts_vec_xyz_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.ts_vec_xyz(0, 0)
            pyrf.ts_vec_xyz(
                list(generate_timeline(64.0, 100)), list(generate_data(100, "vector"))
            )

    def test_ts_vec_xyz_input_shape(self):
        with self.assertRaises(AssertionError):
            # Raises error if data and timeline don't have the same size
            pyrf.ts_vec_xyz(generate_timeline(64.0, 100), generate_data(99, "vector"))
            # Raises error if vector as input
            pyrf.ts_vec_xyz(generate_timeline(64.0, 100), generate_data(100, "scalar"))
            # Raises error if tensor as input
            pyrf.ts_vec_xyz(generate_timeline(64.0, 100), generate_data(100, "tensor"))

    def test_ts_vec_xyz_output_type(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, "vector")
        )
        self.assertIsInstance(result, xarray.DataArray)

    def test_ts_vec_xyz_output_shape(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, "vector")
        )
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)

    def test_ts_vec_xyz_dims(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, "vector")
        )
        self.assertListEqual(list(result.dims), ["time", "comp"])

    def test_ts_vec_xyz_meta(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, "vector")
        )
        self.assertEqual(result.attrs["TENSOR_ORDER"], 1)


class TsTensorXYZTestCase(unittest.TestCase):
    def test_ts_tensor_xyz_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.ts_tensor_xyz(0, 0)
            pyrf.ts_tensor_xyz(
                list(generate_timeline(64.0, 100)), list(generate_data(100, "tensor"))
            )

    def test_ts_tensor_xyz_input_shape(self):
        with self.assertRaises(AssertionError):
            # Raises error if data and timeline don't have the same size
            pyrf.ts_tensor_xyz(
                generate_timeline(64.0, 100), generate_data(99, "tensor")
            )
            # Raises error if vector as input
            pyrf.ts_tensor_xyz(
                generate_timeline(64.0, 100), generate_data(100, "scalar")
            )
            # Raises error if tensor as input
            pyrf.ts_tensor_xyz(
                generate_timeline(64.0, 100), generate_data(100, "vector")
            )

    def test_ts_tensor_xyz_output_type(self):
        result = pyrf.ts_tensor_xyz(
            generate_timeline(64.0, 100), generate_data(100, "tensor")
        )
        self.assertIsInstance(result, xarray.DataArray)

    def test_ts_tensor_xyz_output_shape(self):
        result = pyrf.ts_tensor_xyz(
            generate_timeline(64.0, 100), generate_data(100, "tensor")
        )
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], 3)

    def test_ts_tensor_xyz_dims(self):
        result = pyrf.ts_tensor_xyz(
            generate_timeline(64.0, 100), generate_data(100, "tensor")
        )
        self.assertListEqual(list(result.dims), ["time", "comp_h", "comp_v"])

    def test_ts_tensor_xyz_meta(self):
        result = pyrf.ts_tensor_xyz(
            generate_timeline(64.0, 100), generate_data(100, "tensor")
        )
        self.assertEqual(result.attrs["TENSOR_ORDER"], 2)


class CalcFsTestCase(unittest.TestCase):
    def test_calc_fs_input_type(self):
        self.assertTrue(pyrf.calc_fs(generate_ts(64.0, 100)))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_fs(0)
            pyrf.calc_fs(generate_data(100))

    def test_calc_fs_output_type(self):
        self.assertIsInstance(pyrf.calc_fs(generate_ts(64.0, 100)), float)

    def test_calc_fs_output_value(self):
        self.assertEqual(pyrf.calc_fs(generate_ts(64.0, 100)), 64.0)


class CalcDtTestCase(unittest.TestCase):
    def test_calc_dt_input_type(self):
        self.assertTrue(pyrf.calc_dt(generate_ts(64.0, 100)))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_dt(0)
            pyrf.calc_dt(generate_data(100))

    def test_calc_dt_output_type(self):
        self.assertIsInstance(pyrf.calc_dt(generate_ts(64.0, 100)), float)

    def test_calc_dt_output_value(self):
        self.assertEqual(pyrf.calc_dt(generate_ts(64.0, 100)), 1 / 64.0)


if __name__ == "__main__":
    unittest.main()
