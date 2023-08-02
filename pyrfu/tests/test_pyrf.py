#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import itertools
import random
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, idata, unpack

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
    elif kind == "tensor":
        out = pyrf.ts_tensor_xyz(
            generate_timeline(f_s, n_pts), generate_data(n_pts, kind)
        )
    else:
        raise ValueError("Invalid kind of data!!")

    return out


def generate_vdf(f_s, n_pts, shape, energy01: bool = False, specie: str = "ions"):
    times = generate_timeline(f_s, n_pts)

    phi = np.arange(shape[1])
    phi = np.tile(phi, (n_pts, 1))
    theta = np.arange(shape[2])
    data = np.random.random((n_pts, *shape))

    if energy01:
        energy0 = np.arange(shape[0])
        energy1 = np.arange(shape[0]) + 1
        esteptable = np.arange(100) % 2
        energy = np.tile(energy0, (n_pts, 1))
        energy[esteptable == 1, :] = np.tile(energy1, (np.sum(esteptable), 1))
    else:
        energy = np.arange(shape[0])
        energy = np.tile(energy, (n_pts, 1))
        energy0 = energy[0, :]
        energy1 = energy[1, :]
        esteptable = np.zeros(n_pts)

    attrs = {}
    glob_attrs = {
        "specie": specie,
        "delta_energy_plus": np.ones((n_pts, shape[0])),
        "delta_energy_minus": np.ones((n_pts, shape[0])),
    }

    out = pyrf.ts_skymap(
        times,
        data,
        energy,
        phi,
        theta,
        energy0=energy0,
        energy1=energy1,
        esteptable=esteptable,
        attrs=attrs,
        glob_attrs=glob_attrs,
    )
    return out


class AutoCorrTestCase(unittest.TestCase):
    def test_autocorr_input_type(self):
        self.assertIsNotNone(pyrf.autocorr(generate_ts(64.0, 100, "scalar")))
        self.assertIsNotNone(pyrf.autocorr(generate_ts(64.0, 100, "scalar"), 25))
        self.assertIsNotNone(pyrf.autocorr(generate_ts(64.0, 100, "scalar"), 25, True))

    def test_autocorr_input_shape(self):
        self.assertIsNotNone(pyrf.autocorr(generate_ts(64.0, 100, "scalar")))
        self.assertIsNotNone(pyrf.autocorr(generate_ts(64.0, 100, "vector")))

    def test_autocorr_input_values(self):
        with self.assertRaises(ValueError):
            pyrf.autocorr(generate_ts(64.0, 100, "scalar"), 100)

    def test_autocorr_output_type(self):
        self.assertIsInstance(
            pyrf.autocorr(generate_ts(64.0, 100, "scalar")), xr.DataArray
        )
        self.assertIsInstance(
            pyrf.autocorr(generate_ts(64.0, 100, "vector")), xr.DataArray
        )

    def test_autocorr_output_shape(self):
        result = pyrf.autocorr(generate_ts(64.0, 100, "scalar"))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 100)

        result = pyrf.autocorr(generate_ts(64.0, 100, "scalar"), 25)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 26)

        result = pyrf.autocorr(generate_ts(64.0, 100, "vector"))
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)


class AverageVDFTestCase(unittest.TestCase):
    def test_average_vdf_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.average_vdf(0, 3)
            pyrf.average_vdf(np.random.random((100, 32, 32, 16)), 3)
            pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), [3, 5])

    def test_average_vdf_values(self):
        with self.assertRaises(AssertionError):
            pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), 2)

    def test_average_vdf_output_type(self):
        self.assertIsInstance(
            pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), 3), xr.Dataset
        )

    def test_average_vdf_output_meta(self):
        avg_inds = np.arange(1, 99, 3, dtype=int)
        result = pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), 3)

        self.assertIsInstance(result.attrs["delta_energy_plus"], np.ndarray)
        self.assertEqual(result.attrs["delta_energy_plus"].ndim, 2)
        self.assertEqual(len(result.attrs["delta_energy_plus"]), len(avg_inds))

        self.assertIsInstance(result.attrs["delta_energy_minus"], np.ndarray)
        self.assertEqual(result.attrs["delta_energy_minus"].ndim, 2)
        self.assertEqual(len(result.attrs["delta_energy_minus"]), len(avg_inds))


class Avg4SCTestCase(unittest.TestCase):
    def test_avg_4sc_input(self):
        self.assertIsNotNone(
            pyrf.avg_4sc(
                [
                    generate_ts(64.0, 100, "scalar"),
                    generate_ts(64.0, 100, "scalar"),
                    generate_ts(64.0, 100, "scalar"),
                    generate_ts(64.0, 100, "scalar"),
                ]
            )
        )
        self.assertIsNotNone(
            pyrf.avg_4sc(
                [
                    generate_ts(64.0, 100, "vector"),
                    generate_ts(64.0, 100, "vector"),
                    generate_ts(64.0, 100, "vector"),
                    generate_ts(64.0, 100, "vector"),
                ]
            )
        )
        self.assertIsNotNone(
            pyrf.avg_4sc(
                [
                    generate_ts(64.0, 100, "tensor"),
                    generate_ts(64.0, 100, "tensor"),
                    generate_ts(64.0, 100, "tensor"),
                    generate_ts(64.0, 100, "tensor"),
                ]
            )
        )

        with self.assertRaises(TypeError):
            pyrf.avg_4sc(
                [
                    generate_data(100, "tensor"),
                    generate_data(100, "tensor"),
                    generate_data(100, "tensor"),
                    generate_data(100, "tensor"),
                ]
            )

    def test_avg_4sc_output(self):
        result = pyrf.avg_4sc(
            [
                generate_ts(64.0, 100, "tensor"),
                generate_ts(64.0, 100, "tensor"),
                generate_ts(64.0, 100, "tensor"),
                generate_ts(64.0, 100, "tensor"),
            ]
        )

        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3, 3])


class C4GradTestCase(unittest.TestCase):
    def test_c_4_grad_input(self):
        with self.assertRaises(AssertionError):
            pyrf.c_4_grad(
                generate_ts(64.0, 100, kind="vector"),
                generate_ts(64.0, 100, kind="vector"),
            )
            pyrf.c_4_grad([], [])

            pyrf.c_4_grad(
                [generate_ts(64.0, 100, kind="vector") for _ in range(4)],
                [generate_ts(64.0, 100, kind="vector") for _ in range(4)],
                0,
            )

            pyrf.c_4_grad(
                [generate_ts(64.0, 100, kind="vector") for _ in range(4)],
                [generate_ts(64.0, 100, kind="vector") for _ in range(4)],
                "bazinga",
            )

    def test_c_4_grad_output(self):
        r_mms = [generate_ts(64.0, 100, kind="vector") for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, kind="vector") for _ in range(4)]
        n_mms = [generate_ts(64.0, 100, kind="scalar") for _ in range(4)]

        result = pyrf.c_4_grad(r_mms, b_mms, "grad")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3, 3])

        result = pyrf.c_4_grad(r_mms, b_mms, "div")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(
            list(result.shape),
            [
                100,
            ],
        )

        result = pyrf.c_4_grad(r_mms, b_mms, "curl")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.c_4_grad(r_mms, b_mms, "bdivb")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.c_4_grad(r_mms, b_mms, "curv")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.c_4_grad(r_mms, n_mms, "grad")
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])


class C4JTestCase(unittest.TestCase):
    def test_c_4_j_input(self):
        with self.assertRaises(AssertionError):
            pyrf.c_4_j(
                generate_ts(64.0, 100, kind="vector"),
                generate_ts(64.0, 100, kind="vector"),
            )
            pyrf.c_4_j([], [])

    def test_c_4_j_output(self):
        r_mms = [generate_ts(64.0, 100, kind="vector") for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, kind="vector") for _ in range(4)]
        j, div_b, b_avg, jxb, div_t_shear, div_pb = pyrf.c_4_j(r_mms, b_mms)

        self.assertIsInstance(j, xr.DataArray)
        self.assertListEqual(list(j.shape), [100, 3])

        self.assertIsInstance(div_b, xr.DataArray)
        self.assertListEqual(
            list(div_b.shape),
            [
                100,
            ],
        )

        self.assertIsInstance(b_avg, xr.DataArray)
        self.assertListEqual(list(b_avg.shape), [100, 3])

        self.assertIsInstance(jxb, xr.DataArray)
        self.assertListEqual(list(jxb.shape), [100, 3])

        self.assertIsInstance(div_t_shear, xr.DataArray)
        self.assertListEqual(list(div_t_shear.shape), [100, 3])

        self.assertIsInstance(div_pb, xr.DataArray)
        self.assertListEqual(list(div_pb.shape), [100, 3])


class CalcAgTestCase(unittest.TestCase):
    def test_calc_ag_input_type(self):
        self.assertIsNotNone(pyrf.calc_ag(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_ag(0.0)
            pyrf.calc_ag(generate_data(100))

    def test_calc_ag_output_type(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, "tensor"))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_ag_output_shape(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_calc_ag_dims(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, "tensor"))
        self.assertListEqual(list(result.dims), ["time"])

    def test_calc_ag_meta(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class CalcAgyroTestCase(unittest.TestCase):
    def test_calc_agyro_input_type(self):
        self.assertIsNotNone(pyrf.calc_agyro(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_agyro(0.0)
            pyrf.calc_agyro(generate_data(100))

    def test_calc_agyro_output_type(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, "tensor"))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_agyro_output_shape(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_calc_agyro_dims(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, "tensor"))
        self.assertListEqual(list(result.dims), ["time"])

    def test_calc_agyro_meta(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class CalcDngTestCase(unittest.TestCase):
    def test_calc_dng_input_type(self):
        self.assertIsNotNone(pyrf.calc_dng(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_dng(0.0)
            pyrf.calc_dng(generate_data(100))

    def test_calc_dng_output_type(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, "tensor"))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_dng_output_shape(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_calc_dng_dims(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, "tensor"))
        self.assertListEqual(list(result.dims), ["time"])

    def test_calc_dng_meta(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class CalcDtTestCase(unittest.TestCase):
    def test_calc_dt_input_type(self):
        self.assertIsNotNone(pyrf.calc_dt(generate_ts(64.0, 100, "scalar")))
        self.assertIsNotNone(pyrf.calc_dt(generate_ts(64.0, 100, "vector")))
        self.assertIsNotNone(pyrf.calc_dt(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_dt(0)
            pyrf.calc_dt(generate_data(100))

    def test_calc_dt_output_type(self):
        self.assertIsInstance(pyrf.calc_dt(generate_ts(64.0, 100)), float)


class CalcFsTestCase(unittest.TestCase):
    def test_calc_fs_input_type(self):
        self.assertIsNotNone(pyrf.calc_fs(generate_ts(64.0, 100)))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_fs(0)
            pyrf.calc_fs(generate_data(100))

    def test_calc_fs_output_type(self):
        self.assertIsInstance(pyrf.calc_fs(generate_ts(64.0, 100)), float)


class CalcSqrtQTestCase(unittest.TestCase):
    def test_calc_sqrtq_input_type(self):
        self.assertIsNotNone(pyrf.calc_sqrtq(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            # Raises error if input is not a xarray
            pyrf.calc_sqrtq(0.0)
            pyrf.calc_sqrtq(generate_data(100))

    def test_calc_sqrtq_output_type(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, "tensor"))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_sqrtq_output_shape(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_calc_sqrtq_dims(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, "tensor"))
        self.assertListEqual(list(result.dims), ["time"])

    def test_calc_sqrtq_meta(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, "tensor"))
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class Cart2SphTestCase(unittest.TestCase):
    def test_cart2sph_output(self):
        result = pyrf.cart2sph(1.0, 1.0, 1.0)
        self.assertIsInstance(result[0], np.float64)
        self.assertIsInstance(result[1], np.float64)
        self.assertIsInstance(result[2], np.float64)

        result = pyrf.cart2sph(
            np.random.random(100), np.random.random(100), np.random.random(100)
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertListEqual(
            list(result[0].shape),
            [
                100,
            ],
        )
        self.assertIsInstance(result[1], np.ndarray)
        self.assertListEqual(
            list(result[1].shape),
            [
                100,
            ],
        )
        self.assertIsInstance(result[2], np.ndarray)
        self.assertListEqual(
            list(result[2].shape),
            [
                100,
            ],
        )


class Cart2SphTsTestCase(unittest.TestCase):
    def test_cart2sph_ts_input(self):
        with self.assertRaises(AssertionError):
            pyrf.cart2sph_ts(0.0)
            pyrf.cart2sph_ts(generate_data(100, "vector"))
            pyrf.cart2sph_ts(generate_ts(64.0, 100, "scalar"))
            pyrf.cart2sph_ts(generate_ts(64.0, 100, "vector"), 2)

    def test_cart2sph_ts_output(self):
        result = pyrf.cart2sph_ts(generate_ts(64.0, 100, "vector"), 1)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cart2sph_ts(generate_ts(64.0, 100, "vector"), -1)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])


class CdfEpoch2Datetime64TestCase(unittest.TestCase):
    def test_cdfepoch2datetime64_input_type(self):
        ref_time = 599572869184000000
        self.assertIsNotNone(pyrf.cdfepoch2datetime64(ref_time))
        time_line = np.arange(ref_time, int(ref_time + 100))
        self.assertIsNotNone(pyrf.cdfepoch2datetime64(time_line))
        self.assertIsNotNone(pyrf.cdfepoch2datetime64(list(time_line)))

    def test_cdfepoch2datetime64_output_type(self):
        ref_time = 599572869184000000
        self.assertIsInstance(pyrf.cdfepoch2datetime64(ref_time), np.ndarray)
        time_line = np.arange(ref_time, int(ref_time + 100))
        self.assertIsInstance(pyrf.cdfepoch2datetime64(time_line), np.ndarray)
        self.assertIsInstance(pyrf.cdfepoch2datetime64(list(time_line)), np.ndarray)

    def test_cdfepoch2datetime64_output_shape(self):
        ref_time = 599572869184000000
        self.assertEqual(len(pyrf.cdfepoch2datetime64(ref_time)), 1)
        time_line = np.arange(ref_time, int(ref_time + 100))
        self.assertEqual(len(pyrf.cdfepoch2datetime64(time_line)), 100)
        self.assertEqual(len(pyrf.cdfepoch2datetime64(list(time_line))), 100)


@ddt
class CompressCwtTestCase(unittest.TestCase):
    @data(([], 10), (np.random.random((100, 100)), 100))
    @unpack
    def test_compress_cwt_input(self, cwt, nc):
        with self.assertRaises(AssertionError):
            pyrf.compress_cwt(cwt, nc)

    def test_compress_cwt_output(self):
        times = generate_timeline(64.0, 1000)
        freqs = np.logspace(0, 3, 100)
        cwt_x = xr.DataArray(
            np.random.random((1000, 100)), coords=[times, freqs], dims=["time", "f"]
        )
        cwt_y = xr.DataArray(
            np.random.random((1000, 100)), coords=[times, freqs], dims=["time", "f"]
        )
        cwt_z = xr.DataArray(
            np.random.random((1000, 100)), coords=[times, freqs], dims=["time", "f"]
        )
        cwt = xr.Dataset({"x": cwt_x, "y": cwt_y, "z": cwt_z})
        result = pyrf.compress_cwt(cwt, 10)
        self.assertIsInstance(result[0], np.ndarray)

        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)


class ConvertFACTestCase(unittest.TestCase):
    def test_convert_fac_input(self):
        with self.assertRaises(AssertionError):
            pyrf.convert_fac(0, 0)
            pyrf.convert_fac(generate_data(100, "vector"), generate_data(100, "vector"))

        with self.assertRaises(TypeError):
            pyrf.convert_fac(
                generate_ts(64.0, 100, "tensor"), generate_ts(64.0, 100, "vector")
            )

        with self.assertRaises(TypeError):
            pyrf.convert_fac(
                generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "scalar")
            )

    def test_convert_fac_output(self):
        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector")
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "vector"), generate_ts(64.0, 98, "vector")
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            np.random.random(3),
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 97, "vector"),
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.convert_fac(
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 2])


@ddt
class CotransTestCase(unittest.TestCase):
    def test_cotrans_input(self):
        with self.assertRaises(TypeError):
            pyrf.cotrans(0.0, "gse>gsm")

        with self.assertRaises(IndexError):
            pyrf.cotrans(generate_data(100), "gse>gsm")

        with self.assertRaises(ValueError):
            pyrf.cotrans(generate_ts(64.0, 100, "vector"), "gsm")

    @idata(itertools.permutations(["gei", "geo", "gse", "gsm", "mag", "sm"], 2))
    def test_cotrans_output(self, value):
        transf = f"{value[0]}>{value[1]}"
        result = pyrf.cotrans(generate_ts(64.0, 100, "vector"), transf)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cotrans(generate_ts(64.0, 100, "vector"), transf, hapgood=False)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        inp = generate_ts(64.0, 100, "vector")
        inp.attrs["COORDINATE_SYSTEM"] = value[0]
        result = pyrf.cotrans(inp, value[1], hapgood=False)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        inp = generate_ts(64.0, 100, "vector")
        inp.attrs["COORDINATE_SYSTEM"] = value[0]
        result = pyrf.cotrans(inp, value[1], hapgood=True)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])


class CrossTestCase(unittest.TestCase):
    def test_cross_input(self):
        with self.assertRaises(AssertionError):
            pyrf.cross(
                generate_ts(64.0, 100, "scalar"), generate_ts(64.0, 100, "scalar")
            )
            pyrf.cross(
                generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "scalar")
            )
            pyrf.cross(
                generate_ts(64.0, 100, "scalar"), generate_ts(64.0, 100, "vector")
            )

    def test_cross_output(self):
        result = pyrf.cross(
            generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector")
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cross(
            generate_ts(64.0, 100, "vector"), generate_ts(64.0, 97, "vector")
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])


@ddt
class DateStrTestCase(unittest.TestCase):
    def test_date_str_input(self):
        with self.assertRaises(AssertionError):
            pyrf.date_str("2019-01-01T00:00:00")
            pyrf.date_str([np.datetime64("2019-01-01T00:00:00"), "2019-01-01T00:10:00"])
            pyrf.date_str(["2019-01-01T00:00:00", "2019-01-01T00:10:00"], 1)

            tint = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]
            pyrf.date_str(tint, 0)
            pyrf.date_str(tint, 5)

    @idata(range(1, 5))
    def test_date_str_output(self, value):
        tint = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]
        result = pyrf.date_str(tint, value)
        self.assertIsInstance(result, str)


class Datetime2Iso8601TestCase(unittest.TestCase):
    def test_datetime2iso8601_input_type(self):
        ref_time = datetime.datetime(2019, 1, 1, 0, 0, 0, 0)
        time_line = [ref_time + datetime.timedelta(seconds=i) for i in range(10)]
        self.assertIsNotNone(pyrf.datetime2iso8601(ref_time))
        self.assertIsNotNone(pyrf.datetime2iso8601(time_line))

    def test_datetime2iso8601_output_type(self):
        ref_time = datetime.datetime(2019, 1, 1, 0, 0, 0, 0)
        time_line = [ref_time + datetime.timedelta(seconds=i) for i in range(10)]
        self.assertIsInstance(pyrf.datetime2iso8601(ref_time), str)
        self.assertIsInstance(pyrf.datetime2iso8601(time_line), list)

    def test_datetime2iso8601_output_shape(self):
        ref_time = datetime.datetime(2019, 1, 1, 0, 0, 0, 0)
        time_line = [ref_time + datetime.timedelta(seconds=i) for i in range(10)]

        # ISO8601 contains 29 characters (nanosecond precision)
        self.assertEqual(len(pyrf.datetime2iso8601(ref_time)), 29)
        self.assertEqual(len(pyrf.datetime2iso8601(time_line)), 10)


@ddt
class Datetime642Iso8601TestCase(unittest.TestCase):
    @data(
        datetime.datetime(2019, 1, 1, 0, 0, 0),
        "2019-01-01T00:00:00.000000000",
    )
    def test_datetime642iso8601_input(self, value):
        with self.assertRaises(TypeError):
            pyrf.datetime642iso8601(value)

    @data(np.datetime64("2019-01-01T00:00:00.000000000"), generate_timeline(64.0, 100))
    def test_datetime642iso8601_output(self, value):
        self.assertIsInstance(pyrf.datetime642iso8601(value), np.ndarray)


@ddt
class Datetime642TtnsTestCase(unittest.TestCase):
    @data(
        datetime.datetime(2019, 1, 1, 0, 0, 0),
        "2019-01-01T00:00:00.000000000",
    )
    def test_datetime642ttns_input(self, value):
        with self.assertRaises(TypeError):
            pyrf.datetime642ttns(value)

    @data(np.datetime64("2019-01-01T00:00:00.000000000"), generate_timeline(64.0, 100))
    def test_datetime642ttns_output(self, value):
        self.assertIsInstance(pyrf.datetime642ttns(value), np.ndarray)


@ddt
class Datetime642UnixTestCase(unittest.TestCase):
    @data(
        datetime.datetime(2019, 1, 1, 0, 0, 0),
        "2019-01-01T00:00:00.000000000",
    )
    def test_datetime642unix_input(self, value):
        with self.assertRaises(TypeError):
            pyrf.datetime642unix(value)

    @data(np.datetime64("2019-01-01T00:00:00.000000000"), generate_timeline(64.0, 100))
    def test_datetime642unix_output(self, value):
        self.assertIsInstance(pyrf.datetime642unix(value), np.ndarray)


@ddt
class DecParPerpTestCase(unittest.TestCase):
    @data(
        (generate_data(100, "vector"), generate_ts(64.0, 100, "vector"), False),
        (generate_ts(64.0, 100, "vector"), generate_data(100, "vector"), False),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), 0),
        (generate_ts(64.0, 100, "scalar"), generate_ts(64.0, 100, "vector"), False),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "scalar"), False),
    )
    @unpack
    def test_dec_par_perp_input(self, inp, b_bgd, flag_spin_plane):
        with self.assertRaises(AssertionError):
            pyrf.dec_par_perp(inp, b_bgd, flag_spin_plane)

    @data(
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), False),
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector") * 1e-4,
            False,
        ),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), True),
    )
    @unpack
    def test_dec_par_perp_output(self, inp, b_bgd, flag_spin_plane):
        a_para, a_perp, alpha = pyrf.dec_par_perp(inp, b_bgd, flag_spin_plane)
        self.assertIsInstance(a_para, xr.DataArray)
        self.assertIsInstance(a_perp, xr.DataArray)


@ddt
class DistAppendTestCase(unittest.TestCase):
    @data(
        (None, generate_vdf(64.0, 100, [32, 32, 16])),
        (generate_vdf(64.0, 100, [32, 32, 16]), generate_vdf(64.0, 100, [32, 32, 16])),
    )
    @unpack
    def test_dist_append_output(self, inp0, inp1):
        result = pyrf.dist_append(inp0, inp1)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class DynamicPressTestCase(unittest.TestCase):
    @data(
        (generate_data(100, "scalar"), generate_ts(64.0, 100, "vector"), "ions"),
        (generate_ts(64.0, 100, "scalar"), generate_data(100, "vector"), "ions"),
    )
    @unpack
    def test_dynamic_press_input(self, n_s, v_xyz, specie):
        with self.assertRaises(AssertionError):
            pyrf.dynamic_press(n_s, v_xyz, specie)

    @data("ions", "electrons")
    def test_dynamic_press_output(self, value):
        result = pyrf.dynamic_press(
            generate_ts(64.0, 100, "scalar"), generate_ts(64.0, 100, "vector"), value
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, 1)


@ddt
class EVxBTestCase(unittest.TestCase):
    @data(
        (generate_data(100, "vector"), generate_ts(64.0, 100, "vector"), "vxb"),
        (generate_ts(64.0, 100, "vector"), generate_data(100, "vector"), "vxb"),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), "bazinga"),
    )
    @unpack
    def test_e_vxb_input(self, v_xyz, b_xyz, flag):
        with self.assertRaises((TypeError, AssertionError)):
            pyrf.e_vxb(v_xyz, b_xyz, flag)

    @data(
        (generate_ts(64.0, 100, "vector"), "vxb"),
        (generate_ts(64.0, 100, "vector"), "exb"),
        (np.random.random(3), "vxb"),
        (np.random.random(3), "exb"),
    )
    @unpack
    def test_e_vxb_output(self, v_xyz, flag):
        result = pyrf.e_vxb(v_xyz, generate_ts(64.0, 100, "vector"), flag)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])


class EbNRFTestCase(unittest.TestCase):
    def test_eb_nrf_output(self):
        pyrf.eb_nrf(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            "a",
        )
        pyrf.eb_nrf(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            "b",
        )
        pyrf.eb_nrf(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            np.random.random(3),
        )


@ddt
class EbspTestCase(unittest.TestCase):
    @data(
        (
            None,
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 98, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 120, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            None,
            [1e0, 1e1],
        ),
    )
    @unpack
    def test_ebsp_input_pass(self, e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int):
        self.assertIsNotNone(pyrf.ebsp(e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int))

    @data(
        (
            generate_ts(64.0, 100, "vector"),
            None,
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            None,
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            None,
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
    )
    @unpack
    def test_ebsp_input_fail(self, e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int):
        with self.assertRaises((AssertionError, TypeError, IndexError)):
            pyrf.ebsp(e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int)

    @data(
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": True,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": True,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": False,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": True,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": True,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": random.randint(2, 50),
            "fac_matrix": None,
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": True,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": generate_ts(64.0, 100, "tensor"),
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": generate_ts(64.0, 100, "tensor"),
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": None,
            "m_width_coeff": random.random(),
        },
    )
    def test_ebsp_options(self, value):
        result = pyrf.ebsp(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
            **value,
        )

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["bb_xxyyzzss"], xr.DataArray)

    @data("pc12", "pc35", [1e0, 1e1])
    def test_ebsp_freq_int_pass(self, value):
        self.assertIsNotNone(
            pyrf.ebsp(
                generate_ts(64.0, 100000, "vector"),
                generate_ts(64.0, 100000, "vector"),
                generate_ts(64.0, 100000, "vector"),
                generate_ts(64.0, 100000, "vector"),
                generate_ts(64.0, 100000, "vector"),
                value,
            )
        )

    @data(random.random(), np.random.random(3), "bazinga", [1, 100])
    def test_ebsp_freq_int_fail(self, value):
        with self.assertRaises((AssertionError, ValueError)):
            pyrf.ebsp(
                generate_ts(64.0, 10000, "vector"),
                generate_ts(64.0, 10000, "vector"),
                generate_ts(64.0, 10000, "vector"),
                generate_ts(64.0, 10000, "vector"),
                generate_ts(64.0, 10000, "vector"),
                value,
            )


class TraceTestCase(unittest.TestCase):
    def test_trace_input(self):
        self.assertIsNotNone(pyrf.trace(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            pyrf.trace(generate_data(100, "tensor"))
            pyrf.trace(generate_ts(64.0, 100, "scalar"))
            pyrf.trace(generate_ts(64.0, 100, "vector"))

    def test_trace_output(self):
        result = pyrf.trace(generate_ts(64.0, 100, "tensor"))
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(
            list(result.shape),
            [
                100,
            ],
        )


@ddt
class ShockNormalTestCase(unittest.TestCase):
    def test_shock_normal_input(self):
        with self.assertRaises(AssertionError):
            pyrf.shock_normal([])

        with self.assertRaises(TypeError):
            pyrf.shock_normal(
                {
                    "b_u": np.random.random((3)),
                    "b_d": np.random.random((3)),
                    "v_u": np.random.random((3)),
                    "v_d": np.random.random((3)),
                    "n_u": random.random(),
                    "n_d": random.random(),
                    "r_xyz": random.random(),
                }
            )

    @data(
        {
            "b_u": np.random.random((3)),
            "b_d": np.random.random((3)),
            "v_u": np.random.random((3)),
            "v_d": np.random.random((3)),
            "n_u": random.random(),
            "n_d": random.random(),
        },
        {
            "b_u": np.random.random((2, 3)),
            "b_d": np.random.random((2, 3)),
            "v_u": np.random.random((2, 3)),
            "v_d": np.random.random((2, 3)),
            "n_u": np.random.random((2, 1)),
            "n_d": np.random.random((2, 1)),
        },
        {
            "b_u": np.random.random((3)),
            "b_d": np.random.random((3)),
            "v_u": np.random.random((3)),
            "v_d": np.random.random((3)),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": np.random.random((3)),
        },
        {
            "b_u": np.random.random((3)),
            "b_d": np.random.random((3)),
            "v_u": np.random.random((3)),
            "v_d": np.random.random((3)),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": generate_ts(64.0, 100, "vector"),
        },
        {
            "b_u": np.random.random((3)),
            "b_d": np.random.random((3)),
            "v_u": np.random.random((3)),
            "v_d": np.random.random((3)),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": generate_ts(64.0, 100, "vector"),
            "d2u": random.choice([-1, 1]),
            "dt_f": random.random(),
            "f_cp": random.random(),
        },
    )
    def test_shock_normal_ouput(self, value):
        result = pyrf.shock_normal(value)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["v_sh"], dict)


class StartTestCase(unittest.TestCase):
    def test_start_input_type(self):
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, "scalar")))
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, "vector")))
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, "tensor")))

        with self.assertRaises(AssertionError):
            pyrf.start(0)
            pyrf.start(generate_timeline(64.0, 100))

    def test_start_output(self):
        result = pyrf.start(generate_ts(64.0, 100, "scalar"))
        self.assertIsInstance(result, np.float64)
        self.assertEqual(
            np.datetime64(int(result * 1e9), "ns"),
            np.datetime64("2019-01-01T00:00:00.000"),
        )


class TsSkymapTestCase(unittest.TestCase):
    def test_ts_skymap_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.ts_skymap(0, 0, 0, 0, 0)

    def test_ts_skymap_output_type(self):
        result = pyrf.ts_skymap(
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            np.random.random(16),
        )
        self.assertIsInstance(result, xr.Dataset)

    def test_ts_skymap_output_shape(self):
        result = pyrf.ts_skymap(
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            np.random.random(16),
        )
        self.assertEqual(result.data.ndim, 4)
        self.assertListEqual(list(result.data.shape), [100, 32, 32, 16])
        self.assertEqual(result.energy.ndim, 2)
        self.assertListEqual(list(result.energy.shape), [100, 32])
        self.assertEqual(result.phi.ndim, 2)
        self.assertListEqual(list(result.phi.shape), [100, 32])
        self.assertEqual(result.theta.ndim, 1)
        self.assertListEqual(list(result.theta.shape), [16])

    def test_ts_skymap_output_meta(self):
        result = pyrf.ts_skymap(
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            np.random.random(16),
        )
        self.assertListEqual(
            list(result.attrs.keys()), ["energy0", "energy1", "esteptable"]
        )
        self.assertListEqual(
            list(result.attrs["energy0"].shape),
            [
                32,
            ],
        )
        self.assertListEqual(
            list(result.attrs["energy1"].shape),
            [
                32,
            ],
        )
        self.assertListEqual(
            list(result.attrs["esteptable"].shape),
            [
                100,
            ],
        )

        for k in result:
            self.assertEqual(result[k].attrs, {})


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
        self.assertIsInstance(result, xr.DataArray)

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
        self.assertIsInstance(result, xr.DataArray)

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
        self.assertIsInstance(result, xr.DataArray)

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


@ddt
class WaveletTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, "tensor"), {}),
        (generate_ts(64, 100, "vector"), {"linear": [random.randint(10, 100)]}),
    )
    @unpack
    def test_wavelet_input(self, inp, options):
        with self.assertRaises(TypeError):
            pyrf.wavelet(inp, **options)

    @data(
        (generate_ts(64, 100, "scalar"), {}),
        (generate_ts(64, 99, "scalar"), {}),
        (generate_ts(64, 100, "vector"), {}),
        (generate_ts(64, 100, "vector"), {"linear": True}),
        (generate_ts(64, 100, "vector"), {"linear": random.randint(10, 100)}),
        (
            generate_ts(64, 100, "vector"),
            {"linear": random.randint(10, 100), "return_power": False},
        ),
    )
    @unpack
    def test_wavelet_output(self, inp, options):
        self.assertIsNotNone(pyrf.wavelet(inp, **options))


if __name__ == "__main__":
    unittest.main()
