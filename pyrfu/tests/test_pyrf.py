#!/usr/bin/env python
# -*- coding: utf-8 -*-

import builtins

# Built-in imports
import datetime
import itertools
import random
import unittest
from unittest import mock

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, idata, unpack

from pyrfu import pyrf
from pyrfu.pyrf.compress_cwt import _compress_cwt_1d
from pyrfu.pyrf.ebsp import _average_data, _censure_plot, _freq_int
from pyrfu.pyrf.int_sph_dist import _mc_cart_2d, _mc_cart_3d, _mc_pol_1d
from pyrfu.pyrf.wavelet import _power_c, _power_r, _ww

from . import generate_data, generate_timeline, generate_ts, generate_vdf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


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

    def test_compress_cwt_1d(self):
        result = _compress_cwt_1d.__wrapped__(
            np.random.random((1000, 100)), random.randint(2, 100)
        )
        self.assertIsInstance(result, np.ndarray)


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
    @data(
        (0.0, "gse>gsm", True),
        (generate_data(100), "gse>gsm", True),
        (generate_ts(64.0, 100, "vector"), "gsm", True),
        (generate_ts(64.0, 100, "tensor"), "gse>gsm", True),
        (
            generate_ts(64.0, 100, "vector", {"COORDINATE_SYSTEM": "gse"}),
            "gsm>sm",
            True,
        ),
    )
    @unpack
    def test_cotrans_input(self, inp, flag, hapgood):
        with self.assertRaises((TypeError, IndexError, ValueError, AssertionError)):
            pyrf.cotrans(inp, flag, hapgood)

    @idata(itertools.product(["gei", "geo", "gse", "gsm", "mag", "sm"], repeat=2))
    def test_cotrans_output_trans(self, value):
        transf = f"{value[0]}>{value[1]}"

        inp = generate_ts(64.0, 100, "vector")
        result = pyrf.cotrans(inp, transf, hapgood=True)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cotrans(inp, transf, hapgood=False)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        inp.attrs["COORDINATE_SYSTEM"] = value[0]
        result = pyrf.cotrans(inp, transf, hapgood=False)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cotrans(inp, value[1], hapgood=False)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cotrans(inp, value[1], hapgood=True)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

    def test_cotrans_output_exot(self):
        inp = generate_ts(64.0, 100, "scalar")
        result = pyrf.cotrans(inp, "gse>gsm", hapgood=True)
        self.assertIsInstance(result, xr.DataArray)
        result = pyrf.cotrans(inp, "dipoledirectiongse", hapgood=True)
        self.assertIsInstance(result, xr.DataArray)


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


@ddt
class EbNRFTestCase(unittest.TestCase):
    @data("a", "b", np.random.random(3))
    def test_eb_nrf_output(self, value):
        result = pyrf.eb_nrf(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            value,
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EdbTestCase(unittest.TestCase):
    @data("e.b=0", "e_perp+nan", "e_par")
    def test_edb_output(self, value):
        pyrf.edb(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            random.random() * 90,
            value,
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
            {},
        ),
        (
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 98, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 120, "vector"),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(99.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(40.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(40.0, 100, "vector"),
            generate_ts(40.0, 100, "vector"),
            generate_ts(40.0, 100, "vector"),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            generate_ts(64.0, 97, "vector"),
            [1e0, 1e1],
            {"fac_matrix": generate_ts(64.0, 100, "tensor")},
        ),
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            None,
            [1e0, 1e1],
            {},
        ),
    )
    @unpack
    def test_ebsp_input_pass(self, e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int, options):
        result = pyrf.ebsp(e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int, **options)
        self.assertIsInstance(result, dict)

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
        (
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e1, 1e0],
        ),
        (
            generate_ts(64.0, 100, "vector")[:, :2],
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            [1e0, 1e1],
        ),
    )
    @unpack
    def test_ebsp_input_fail(self, e_xyz, db_xyz, b_xyz, b_bgd, xyz, freq_int):
        with self.assertRaises((AssertionError, TypeError, IndexError, ValueError)):
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
            "fac": False,
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

    @data(([0.32, 3.2], generate_ts(64.0, 100000, "vector")))
    @unpack
    def test_average_data(self, freq_int, data):
        _, _, _, out_time = _freq_int(freq_int, data)
        in_time = data.time.data.astype(np.float64) / 1e9

        result = _average_data.__wrapped__(data.data, in_time, out_time, None)
        self.assertIsInstance(result, np.ndarray)
        self.assertListEqual(list(result.shape), [len(out_time), 3])

    @data(([0.32, 3.2], generate_ts(64.0, 100000, "vector")))
    @unpack
    def test_censure_plot(self, freq_int, data):
        _, _, out_sampling, out_time = _freq_int(freq_int, data)
        a_ = np.logspace(1, 2, 12)
        idx_nan = np.full(len(data), False)
        idx_nan[np.random.randint(100000, size=(100))] = True
        censure = np.floor(2 * a_ * out_sampling / 64.0 * 8)
        result = _censure_plot.__wrapped__(
            np.random.random((len(out_time), len(a_))),
            idx_nan,
            censure,
            len(data),
            a_,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertListEqual(list(result.shape), [len(out_time), len(a_)])


class EndTestCase(unittest.TestCase):
    def test_end_input(self):
        with self.assertRaises(AssertionError):
            pyrf.end(generate_timeline(64.0, 100))

    def test_end_output(self):
        pyrf.end(generate_ts(64.0, 100))


@ddt
class EstimateTestCase(unittest.TestCase):
    @data(
        ("bazinga", random.random(), None),
        ("capacitance_wire", 0, random.random()),
        ("capacitance_wire", random.randint(1, 9), random.randint(1, 9)),
        ("capacitance_cylinder", random.randint(20, 100), random.randint(1, 9)),
    )
    @unpack
    def test_estimate_input(self, what_to_estimate, radius, length):
        with self.assertRaises((NotImplementedError, ValueError)):
            pyrf.estimate(what_to_estimate, radius, length)

    @data(
        ("capacitance_disk", random.random(), None),
        ("capacitance_sphere", random.random(), None),
        ("capacitance_wire", random.random(), random.randint(10, 100)),
        ("capacitance_cylinder", random.randint(1, 9), random.randint(40, 100)),
        ("capacitance_cylinder", random.randint(1, 9), random.randint(5, 26)),
    )
    @unpack
    def test_estimate_output(self, what_to_estimate, radius, length):
        result = pyrf.estimate(what_to_estimate, radius, length)
        self.assertIsInstance(result, float)


@ddt
class ExtendTintTestCase(unittest.TestCase):
    def test_extend_tint_input(self):
        with self.assertRaises(TypeError):
            pyrf.extend_tint([0, 0], None)

    @data(
        (
            ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"],
            [-random.random(), random.random()],
        ),
        (["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"], None),
        (
            [
                np.datetime64("2019-01-01T00:00:00.000000000"),
                np.datetime64("2019-01-01T00:10:00.000000000"),
            ],
            None,
        ),
    )
    @unpack
    def test_extend_tint_ouput(self, tint, ext):
        pyrf.extend_tint(tint, ext)


@ddt
class FiltTestCase(unittest.TestCase):
    @data(
        (generate_data(100), 0, random.randint(1, 22), random.choice(range(1, 10, 2))),
        (
            generate_ts(64.0, 100),
            "bazinga",
            random.randint(1, 22),
            random.choice(range(1, 10, 2)),
        ),
        (
            generate_ts(64.0, 100),
            random.randint(1, 22),
            "bazinga",
            random.choice(range(1, 10, 2)),
        ),
        (generate_ts(64.0, 100), 0, random.randint(1, 22), "ORDEEERRRR"),
    )
    @unpack
    def test_filt_input(self, inp, f_min, f_max, order):
        with self.assertRaises(AssertionError):
            pyrf.filt(inp, f_min, f_max, order)

    @data(
        (
            generate_ts(64.0, 100, "scalar"),
            0,
            1,
            -1,
        ),
        (
            generate_ts(64.0, 100, "vector"),
            0,
            1,
            -1,
        ),
        (
            generate_ts(64.0, 100, "vector"),
            0,
            random.randint(2, 22),
            -1,
        ),
        (
            generate_ts(64.0, 100, "vector"),
            0,
            random.randint(2, 22),
            random.choice(range(1, 10, 2)),
        ),
        (
            generate_ts(64.0, 100, "vector"),
            random.randint(2, 22),
            0,
            -1,
        ),
        (
            generate_ts(64.0, 100, "vector"),
            random.randint(2, 22),
            0,
            random.choice(range(1, 10, 2)),
        ),
        (
            generate_ts(64.0, 100, "vector"),
            random.randint(2, 11),
            random.randint(12, 22),
            -1,
        ),
        (
            generate_ts(64.0, 100, "vector"),
            random.randint(2, 11),
            random.randint(12, 22),
            random.choice(range(1, 10, 2)),
        ),
    )
    @unpack
    def test_filt_output(self, inp, f_min, f_max, order):
        result = pyrf.filt(inp, f_min, f_max, order)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class GradientTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, "scalar"),
        generate_ts(64.0, 100, "vector"),
        generate_ts(64.0, 100, "tensor"),
    )
    def test_gradient_output(self, value):
        value.attrs = {"UNITS": "bazinga"}
        result = pyrf.gradient(value)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), list(value.shape))


@ddt
class Gse2GsmTestCase(unittest.TestCase):
    @data(
        (generate_data(100, "vector"), "gse>gsm"),
        (generate_ts(64.0, 100, "scalar"), "gse>gsm"),
        (generate_ts(64.0, 100, "vector"), "bazinga"),
    )
    @unpack
    def test_gse2gsm_input(self, inp, flag):
        with self.assertRaises(AssertionError):
            pyrf.gse2gsm(inp, flag)

    @data(
        (generate_ts(64.0, 100, "vector"), "gse>gsm"),
        (generate_ts(64.0, 100, "vector"), "gsm>gse"),
    )
    @unpack
    def test_gse2gsm_output(self, inp, flag):
        pyrf.gse2gsm(inp, flag)


@ddt
class HistogramTestCase(unittest.TestCase):
    @data(
        (random.randint(2, 100), None, None, None),
        (np.sort(np.random.random(10)), None, None, None),
        ("fd", None, None, None),
        ("auto", np.sort(np.random.random(2)), None, None),
        (100, None, np.random.random(1000), None),
        ("auto", None, None, True),
    )
    @unpack
    def test_histogram_output(self, bins, y_range, weights, density):
        result = pyrf.histogram(
            generate_ts(64.0, 1000), bins, y_range, weights, density
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Histogram2DTestCase(unittest.TestCase):
    @data(
        (random.randint(2, 100), None, None, None),
        (np.sort(np.random.random(100)), None, None, None),
        (np.random.randint(2, 100, size=(2,)), None, None, None),
        ([np.sort(np.random.random(100)) for _ in range(2)], None, None, None),
        (100, np.sort(np.random.random((2, 2)), axis=1), None, None),
        (100, None, np.random.random(1000), None),
        (100, None, None, True),
    )
    @unpack
    def test_histogram2d_output(self, bins, y_range, weights, density):
        result = pyrf.histogram2d(
            generate_ts(64.0, 1000),
            generate_ts(64.0, 900),
            bins,
            y_range,
            weights,
            density,
        )
        self.assertIsInstance(result, xr.DataArray)
        result = pyrf.histogram2d(
            generate_ts(64.0, 1000),
            generate_ts(64.0, 1000),
            bins,
            y_range,
            weights,
            density,
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class IncrementsTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, "scalar"),
        generate_ts(64.0, 100, "vector"),
        generate_ts(64.0, 100, "tensor"),
    )
    def test_increments_output(self, value):
        result = pyrf.increments(value, random.randint(1, 50))
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], xr.DataArray)


@ddt
class IntSphDistTestCase(unittest.TestCase):
    @data(
        {"projection_base": "pol", "projection_dim": "2d"},
    )
    def test_int_sph_dist_input(self, value):
        vdf = np.random.random((51, 32, 16))
        speed = np.linspace(0, 1, 51)
        phi = np.arange(32)
        theta = np.arange(16)
        speed_grid = np.linspace(-1, 1, 101)

        with self.assertRaises((RuntimeError, NotImplementedError)):
            pyrf.int_sph_dist(vdf, speed, phi, theta, speed_grid, **value)

    @data(
        {},
        {"weight": "lin"},
        {"weight": "log"},
        {"speed_edges": np.linspace(-0.01, 1.01, 52)},
        {"speed_grid_edges": np.linspace(-1.01, 1.01, 102)},
        {
            "phi_grid": np.arange(0, 32),
            "projection_base": "cart",
            "projection_dim": "2d",
        },
        {"projection_base": "cart", "projection_dim": "3d"},
    )
    def test_int_sph_dist_output(self, value):
        vdf = np.random.random((51, 32, 16))
        speed = np.linspace(0, 1, 51)
        phi = np.arange(32)
        theta = np.arange(16)
        speed_grid = np.linspace(-1, 1, 101)
        result = pyrf.int_sph_dist(vdf, speed, phi, theta, speed_grid, **value)
        self.assertIsInstance(result, dict)

    @data(
        (
            np.random.random((51, 32, 16)),
            np.linspace(0, 1, 51),
            np.arange(32),
            np.arange(16),
            np.ones(51) * 0.02,
            np.ones(51) * 0.01,
            np.ones(32),
            np.ones(16),
            np.linspace(-1.01, 1.01, 102),
            np.ones(101) * 0.02 * np.pi / 16,
            np.array([-np.inf, np.inf]),
            np.array([-np.pi, np.pi]),
            np.ones((51, 32, 16), dtype=np.int64) * 10,
            np.eye(3),
        )
    )
    def test_mc_pol_1d(self, value):
        vdf, *args = value
        vdf[vdf < 1e-2] = 0
        self.assertIsInstance(_mc_pol_1d.__wrapped__(vdf, *args), np.ndarray)

    @data(
        (
            np.random.random((51, 32, 16)),
            np.linspace(0, 1, 51),
            np.arange(32),
            np.arange(16),
            np.ones(51) * 0.02,
            np.ones(51) * 0.01,
            np.ones(32),
            np.ones(16),
            np.linspace(-1.01, 1.01, 102),
            0.02**2,
            np.array([-np.inf, np.inf]),
            np.array([-np.pi, np.pi]),
            (np.ones((51, 32, 16), dtype=np.int64) * 10).astype(int),
            np.eye(3),
        )
    )
    def test_mc_cart_2d(self, value):
        vdf, *args = value
        vdf[vdf < 1e-2] = 0
        self.assertIsInstance(_mc_cart_2d.__wrapped__(vdf, *args), np.ndarray)

    @data(
        (
            np.random.random((51, 32, 16)),
            np.linspace(0, 1, 51),
            np.arange(32),
            np.arange(16),
            np.ones(51) * 0.02,
            np.ones(51) * 0.01,
            np.ones(32),
            np.ones(16),
            np.linspace(-1.01, 1.01, 102),
            0.02**2,
            np.array([-np.inf, np.inf]),
            np.array([-np.pi, np.pi]),
            (np.ones((51, 32, 16), dtype=np.int64) * 10).astype(int),
            np.eye(3),
        )
    )
    def test_mc_cart_3d(self, value):
        vdf, *args = value
        vdf[vdf < 1e-2] = 0
        self.assertIsInstance(_mc_cart_3d.__wrapped__(vdf, *args), np.ndarray)


@ddt
class IntegrateTestCase(unittest.TestCase):
    @data(generate_ts(64.0, 100, "scalar"), generate_ts(64.0, 100, "vector"))
    def test_integrate_output(self, value):
        result = pyrf.integrate(value)
        self.assertIsInstance(result, xr.DataArray)


class IPlasmaCalcTestCase(unittest.TestCase):
    def test_iplasma_calc_output(self):
        with mock.patch.object(builtins, "input", lambda _: random.randint(10, 100)):
            result = pyrf.iplasma_calc(True, True)
            self.assertIsInstance(result, dict)

            result = pyrf.iplasma_calc(False, False)
            self.assertIsNone(result)


@ddt
class Iso86012DatetimeTestCase(unittest.TestCase):
    @data(
        "2019-01-01T00:00:00.000000000",
        ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"],
        np.array(["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]),
    )
    def test_iso86012datetime(self, value):
        result = pyrf.iso86012datetime(value)
        self.assertIsInstance(result, list)


@ddt
class Iso86012Unix(unittest.TestCase):
    @data(
        "2019-01-01T00:00:00.000000000",
        ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"],
        np.array(["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]),
    )
    def test_iso86012unix_output(self, value):
        result = pyrf.iso86012unix(value)
        self.assertIsInstance(result, np.ndarray)


@ddt
class Iso86012TimeVec(unittest.TestCase):
    @data(
        "2019-01-01T00:00:00.000000000",
        ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"],
        np.array(["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]),
    )
    def test_iso86012timevec_output(self, value):
        result = pyrf.iso86012timevec(value)
        self.assertIsInstance(result, np.ndarray)


@ddt
class LowPassTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 10000, "scalar"),
        generate_ts(64.0, 10000, "vector"),
        generate_ts(64.0, 10000, "tensor"),
    )
    def test_lowpass_output(self, value):
        pyrf.lowpass(value, random.random(), 64.0)


@ddt
class LShellTestCase(unittest.TestCase):
    @data("gei", "geo", "gse", "gsm", "mag", "sm")
    def test_l_shell_output(self, value):
        result = pyrf.l_shell(
            generate_ts(64.0, 100, "vector", {"COORDINATE_SYSTEM": value})
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class MeanTestCase(unittest.TestCase):
    @data(None, generate_ts(64.0, 100, "vector"))
    def test_mean_output(self, value):
        result = pyrf.mean(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "vector"),
            value,
        )
        self.assertIsInstance(result, xr.DataArray)


class MeanBinsTestCase(unittest.TestCase):
    def test_mean_bins_output(self):
        result = pyrf.mean_bins(
            generate_ts(64.0, 100), generate_ts(64.0, 100), random.randint(2, 20)
        )
        self.assertIsInstance(result, xr.Dataset)


class MedianBinsTestCase(unittest.TestCase):
    def test_median_bins_output(self):
        result = pyrf.median_bins(
            generate_ts(64.0, 100), generate_ts(64.0, 100), random.randint(2, 20)
        )
        self.assertIsInstance(result, xr.Dataset)


@ddt
class MvaTestCase(unittest.TestCase):
    @data("mvar", "<bn>=0", "td")
    def test_mva_output(self, method):
        result = pyrf.mva(generate_ts(64.0, 100, "vector"), method)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)


@ddt
class NewXyzTestCase(unittest.TestCase):
    @data(generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "tensor"))
    def test_new_xyz_output(self, inp):
        result = pyrf.new_xyz(inp, np.random.random((3, 3)))
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, inp.ndim)


class NormTestCase(unittest.TestCase):
    def test_norm_output(self):
        result = pyrf.norm(generate_ts(64.0, 100, "vector"))
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)


@ddt
class PlasmaBetaTestCase(unittest.TestCase):
    @data((generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "tensor")))
    @unpack
    def test_plasma_beta_output(self, b_xyz, p_xyz):
        result = pyrf.plasma_beta(b_xyz, p_xyz)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class StructFuncTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, "scalar"), None, random.randint(1, 100)),
        (generate_ts(64.0, 100, "vector"), None, random.randint(1, 100)),
        (generate_ts(64.0, 100, "tensor"), None, random.randint(1, 100)),
        (generate_ts(64.0, 100, "vector"), np.random.randint([1] * 50, [50] * 50), 1),
    )
    @unpack
    def test_struct_func_output(self, inp, scales, order):
        result = pyrf.struct_func(inp, scales, order)
        self.assertIsInstance(result, xr.DataArray)


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


class OptimizeNbins1DTestCase(unittest.TestCase):
    def test_optimize_nbins_1d(self):
        result = pyrf.optimize_nbins_1d(
            generate_ts(64.0, 1000),
            n_min=random.randint(2, 10),
            n_max=random.randint(20, 100),
        )

        self.assertIsInstance(result, int)


class OptimizeNbins2DTestCase(unittest.TestCase):
    def test_optimize_nbins_2d(self):
        result = pyrf.optimize_nbins_2d(
            generate_ts(64.0, 1000),
            generate_ts(64.0, 1000),
            n_min=[random.randint(2, 10), random.randint(2, 10)],
            n_max=[random.randint(20, 100), random.randint(20, 100)],
        )
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)


class Pid4SCTestCase(unittest.TestCase):
    def test_pid_4sc_output(self):
        result = pyrf.pid_4sc(
            [generate_ts(64.0, 100, "vector") for _ in range(4)],
            [generate_ts(64.0, 100, "vector") for _ in range(4)],
            [generate_ts(64.0, 100, "tensor") for _ in range(4)],
            [generate_ts(64.0, 100, "vector") for _ in range(4)],
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)


class PlasmaCalcTestCase(unittest.TestCase):
    def test_plasma_calc_output(self):
        result = pyrf.plasma_calc(
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
        )
        self.assertIsInstance(result, xr.Dataset)


@ddt
class ResampleTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100), generate_ts(640.0, 1000)),
        (generate_ts(640.0, 1000), generate_ts(64.0, 100)),
        (generate_vdf(64.0, 100, [32, 32, 16]), generate_ts(640.0, 1000)),
        (generate_ts(64.0, 100), generate_ts(640.0, 2)),
        (generate_ts(64.0, 100), generate_ts(640.0, 1)),
    )
    @unpack
    def test_resample_output(self, inp, ref):
        result = pyrf.resample(inp, ref)
        self.assertIsInstance(result, type(inp))


@ddt
class PoyntingFluxTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector")),
        (generate_ts(128.0, 200, "vector"), generate_ts(64.0, 100, "vector")),
        (generate_ts(64.0, 100, "vector"), generate_ts(128.0, 200, "vector")),
    )
    @unpack
    def test_poynting_flux_output(self, e_xyz, b_xyz):
        result = pyrf.poynting_flux(e_xyz, b_xyz, None)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)

        result = pyrf.poynting_flux(e_xyz, b_xyz, generate_ts(64.0, 100, "vector"))
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)
        self.assertIsInstance(result[2], xr.DataArray)


class PresAnisTestCase(unittest.TestCase):
    def test_pres_anis_output(self):
        result = pyrf.pres_anis(
            generate_ts(64.0, 100, "tensor"), generate_ts(64.0, 100, "vector")
        )
        self.assertIsInstance(result, xr.DataArray)


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


@ddt
class ShockParametersTestCase(unittest.TestCase):
    def test_shock_parameters_input(self):
        with self.assertRaises(AssertionError):
            pyrf.shock_parameters(
                {
                    "b": np.random.random(3),
                    "n": random.random(),
                    "v": np.random.random(3),
                    "t_i": random.random(),
                    "t_e": random.random(),
                    "v_sh": random.random(),
                    "nvec": np.random.random(3),
                    "ref_sys": "bazinga",
                }
            )

    @data(
        {
            "b": np.random.random(3),
            "n": random.random(),
            "v": np.random.random(3),
            "t_i": random.random(),
            "t_e": random.random(),
            "ref_sys": "nif",
        },
        {
            "b": np.random.random(3),
            "n": random.random(),
            "v": np.random.random(3),
            "t_i": random.random(),
            "t_e": random.random(),
            "v_sh": random.random(),
            "nvec": np.random.random(3),
            "ref_sys": "nif",
        },
        {
            "b": np.random.random(3),
            "n": random.random(),
            "v": np.random.random(3),
            "t_i": random.random(),
            "t_e": random.random(),
            "v_sh": random.random(),
            "nvec": np.random.random(3),
            "ref_sys": "sc",
        },
    )
    def test_shock_parameters_output(self, value):
        pyrf.shock_parameters(value)


@ddt
class SolidAngleTestCase(unittest.TestCase):
    @data(
        tuple(np.random.random(3) for _ in range(3)),
        tuple(generate_data(100, "vector") for _ in range(3)),
        tuple(generate_ts(64.0, 100, "vector") for _ in range(3)),
    )
    @unpack
    def test_solid_angle_ouput(self, inp0, inp1, inp2):
        result = pyrf.solid_angle(inp0, inp1, inp2)
        self.assertIsInstance(result, np.ndarray)


@ddt
class Sph2CartTestCase(unittest.TestCase):
    @data(
        tuple(generate_data(100, "scalar") for _ in range(3)),
        tuple(generate_ts(64.0, 100, "scalar") for _ in range(3)),
    )
    @unpack
    def test_sph2cart_output(self, azimuth, elevation, r):
        self.assertIsNotNone(pyrf.sph2cart(azimuth, elevation, r))


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


@ddt
class TsAppendTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, "scalar"),
        generate_ts(64.0, 100, "vector"),
        generate_ts(64.0, 100, "tensor"),
    )
    def test_ts_append_output(self, value):
        value.attrs = {
            "bazinga": "This is my spot!!",
            "I AM GROOT": "I AM STEVE ROGERS",
            "random": np.random.random(100),
        }
        value.time.attrs = {
            "bazinga": "This is my spot!!",
            "I AM GROOT": "I AM STEVE ROGERS",
            "random": np.random.random(100),
        }

        result = pyrf.ts_append(None, value)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, value.ndim)

        result = pyrf.ts_append(value, value)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, value.ndim)


@ddt
class TimeClipTestCase(unittest.TestCase):
    @data(
        [
            datetime.datetime(2019, 1, 1, 0, 0, 0, 312),
            datetime.datetime(2019, 1, 1, 0, 0, 0, 468),
        ],
        "2019-01-01T00:00:00.312",
    )
    def test_time_clip_input(self, value):
        with self.assertRaises(TypeError):
            pyrf.time_clip(generate_ts(64.0, 100), value)

    @data(generate_ts(64.0, 100), generate_vdf(64.0, 100, (32, 32, 16)))
    def test_time_clip_output(self, value):
        result = pyrf.time_clip(
            value, ["2019-01-01T00:00:00.312", "2019-01-01T00:00:00.468"]
        )
        self.assertIsInstance(result, type(value))

        result = pyrf.time_clip(
            value,
            [
                np.datetime64("2019-01-01T00:00:00.312"),
                np.datetime64("2019-01-01T00:00:00.468"),
            ],
        )
        self.assertIsInstance(result, type(value))

        result = pyrf.time_clip(value, generate_ts(64.0, 20))
        self.assertIsInstance(result, type(value))


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


@ddt
class TsSpectrTestCase(unittest.TestCase):
    @data(
        (0, np.random.random(10), np.random.random((100, 10)), "energy", None),
        (generate_timeline(64.0, 100), 0, np.random.random((100, 10)), "energy", None),
        (generate_timeline(64.0, 100), 0, np.random.random((100, 10)), "energy", None),
        (
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random((98, 10)),
            "energy",
            None,
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random((100, 9)),
            "energy",
            None,
        ),
    )
    @unpack
    def test_ts_spectr_input(self, time, ener, data, comp_name, attrs):
        with self.assertRaises(AssertionError):
            pyrf.ts_spectr(time, ener, data, comp_name, attrs)

    def test_ts_spectr_output(self):
        result = pyrf.ts_spectr(
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random((100, 10)),
        )
        self.assertIsInstance(result, xr.DataArray)


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
class Ttns2Datetime64TestCase(unittest.TestCase):
    @data(
        int(random.random() * 1e12),
        [int(random.random() * 1e12), int(random.random() * 1e12)],
        np.array([int(random.random() * 1e12), int(random.random() * 1e12)]),
    )
    def test_ttns2datetime64_output(self, value):
        result = pyrf.ttns2datetime64(value)
        self.assertIsInstance(result, np.ndarray)


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

    @data(
        (
            np.random.random((100, 1)),
            np.random.random((1, 200)),
            random.random(),
            np.random.random((100, 1)),
            random.randint(16, 96),
        )
    )
    @unpack
    def test_ww(self, s_ww, scales_mat, sigma, frequencies_mat, f_nyq):
        self.assertIsNotNone(
            _ww.__wrapped__(s_ww, scales_mat, sigma, frequencies_mat, f_nyq)
        )

    @data(
        (
            np.random.random((100, 3)) + np.random.random((100, 3)) * 1j,
            np.random.random((100, 3)),
        )
    )
    @unpack
    def test_power_r(self, power, new_freq_mat):
        self.assertIsNotNone(_power_r.__wrapped__(power, new_freq_mat))

    @data(
        (
            np.random.random((100, 3)) + np.random.random((100, 3)) * 1j,
            np.random.random((100, 3)),
        )
    )
    @unpack
    def test_power_c(self, power, new_freq_mat):
        self.assertIsNotNone(_power_c.__wrapped__(power, new_freq_mat))


@ddt
class VhtTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), True),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 103, "vector"), True),
        (generate_ts(64.0, 100, "vector"), generate_ts(64.0, 100, "vector"), False),
    )
    @unpack
    def test_vht_output(self, e, b, no_ez):
        result = pyrf.vht(e, b, no_ez)

        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], xr.DataArray)
        self.assertIsInstance(result[2], np.ndarray)


if __name__ == "__main__":
    unittest.main()
