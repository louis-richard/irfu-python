#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import builtins
import datetime
import itertools
import random
import unittest
from unittest import mock

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from ddt import data, ddt, idata, unpack

# Local imports
from .. import pyrf
from ..pyrf.compress_cwt import _compress_cwt_1d
from ..pyrf.ebsp import _average_data, _censure_plot, _freq_int
from ..pyrf.int_sph_dist import (
    _mc_cart_2d,
    _mc_cart_3d,
    _mc_pol_1d,
    _speed_bin_edges,
    _uniform_step,
)
from ..pyrf.shock_normal import _shock_angle
from ..pyrf.wavelet import _power_c, _power_r, _ww
from . import generate_data, generate_timeline, generate_ts, generate_vdf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


@ddt
class AutoCorrTestCase(unittest.TestCase):

    def test_autocorr_input_type(self):
        with self.assertRaises(TypeError):
            pyrf.autocorr(generate_data(100, 3))

    @data(
        (generate_ts(64.0, 100, tensor_order=2), None, True),
        (generate_ts(64.0, 100, tensor_order=0), 100, True),
    )
    @unpack
    def test_autocorr_input_value(self, inp, maxlags, normed):
        with self.assertRaises(ValueError):
            pyrf.autocorr(inp, maxlags, normed)

    def test_autocorr_output_type(self):
        self.assertIsInstance(
            pyrf.autocorr(generate_ts(64.0, 100, tensor_order=0)), xr.DataArray
        )
        self.assertIsInstance(
            pyrf.autocorr(generate_ts(64.0, 100, tensor_order=1)), xr.DataArray
        )

    def test_autocorr_output_value(self):
        result = pyrf.autocorr(generate_ts(64.0, 100, tensor_order=0))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 100)

        result = pyrf.autocorr(generate_ts(64.0, 100, tensor_order=0), 25)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 26)

        result = pyrf.autocorr(generate_ts(64.0, 100, tensor_order=1))
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)


@ddt
class AverageVDFTestCase(unittest.TestCase):
    @data(
        (0, 3),
        (np.random.random((100, 32, 32, 16)), 3),
        (generate_vdf(64.0, 100, [32, 32, 16]), [3, 5]),
    )
    @unpack
    def test_average_vdf_input_type(self, vdf, n_pts):
        with self.assertRaises(TypeError):
            pyrf.average_vdf(vdf, n_pts)

    def test_average_vdf_values(self):
        with self.assertRaises(ValueError):
            pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), 2)

    def test_average_vdf_method_value(self):
        with self.assertRaises(NotImplementedError):
            pyrf.average_vdf(
                generate_vdf(64.0, 100, [32, 32, 16]), 3, method="bazinga!"
            )

    @data("mean", "sum")
    def test_average_vdf_output_type(self, method):
        result = pyrf.average_vdf(
            generate_vdf(64.0, 100, [32, 32, 16]), 3, method=method
        )
        self.assertIsInstance(result, xr.Dataset)

    def test_average_vdf_output_meta(self):
        avg_inds = np.arange(1, 99, 3, dtype=int)
        result = pyrf.average_vdf(generate_vdf(64.0, 100, [32, 32, 16]), 3)

        self.assertIsInstance(result.attrs["delta_energy_plus"], np.ndarray)
        self.assertEqual(result.attrs["delta_energy_plus"].ndim, 2)
        self.assertEqual(len(result.attrs["delta_energy_plus"]), len(avg_inds))

        self.assertIsInstance(result.attrs["delta_energy_minus"], np.ndarray)
        self.assertEqual(result.attrs["delta_energy_minus"].ndim, 2)
        self.assertEqual(len(result.attrs["delta_energy_minus"]), len(avg_inds))


@ddt
class Avg4SCTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100) for _ in range(4)),
        [generate_data(100) for _ in range(4)],
    )
    def test_avg_4sc_input(self, value):

        with self.assertRaises(TypeError):
            pyrf.avg_4sc(value)

    @idata(range(3))
    def test_avg_4sc_output(self, tensor_order):
        result = pyrf.avg_4sc(
            [
                generate_ts(64.0, 100, tensor_order=tensor_order),
                generate_ts(64.0, 100, tensor_order=tensor_order),
                generate_ts(64.0, 100, tensor_order=tensor_order),
                generate_ts(64.0, 100, tensor_order=tensor_order),
            ]
        )

        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, *[3] * tensor_order])


class NanAvg4SCTestCase(unittest.TestCase):
    def test_nanavg_4sc_values(self):
        # MMS1-3 at 2, MMS4 missing: the average is 2 (NaNs used to count as 0,
        # giving 1.5), and NaN where no spacecraft has data.
        time = generate_timeline(1.0, 5)
        b_list = [
            pyrf.ts_vec_xyz(time, np.full((5, 3), value), attrs={"mmsId": i + 1})
            for i, value in enumerate([2.0, 2.0, 2.0, np.nan])
        ]
        for b_xyz in b_list[:3]:
            b_xyz.data[2] = np.nan

        result = pyrf.nanavg_4sc(b_list)

        expected = np.full((5, 3), 2.0)
        expected[2] = np.nan
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.attrs["mmsId"], "4sc_avg")

        # The caller's attributes are unchanged
        self.assertEqual(b_list[0].attrs["mmsId"], 1)

    def test_nanavg_4sc_input(self):
        with self.assertRaises(TypeError):
            pyrf.nanavg_4sc(tuple(generate_ts(64.0, 100) for _ in range(4)))


class C4VTestCase(unittest.TestCase):
    @staticmethod
    def _plane_crossing():
        # Plane discontinuity with normal (1, 2, 2) / 3 moving at 50 km/s past a
        # tetrahedron that is not aligned with the axes (positions in km)
        time = generate_timeline(1.0, 20)
        r_0 = np.array([60000.0, 20000.0, 5000.0])
        offsets = np.array(
            [[0, 0, 0], [30, 5, 10], [-10, 40, 15], [5, -12, 35]], dtype=float
        )
        r_xyz = [pyrf.ts_vec_xyz(time, np.tile(r_0 + o, (20, 1))) for o in offsets]
        velocity = 50.0 * np.array([1.0, 2.0, 2.0]) / 3.0
        delta_t = offsets @ velocity / np.linalg.norm(velocity) ** 2
        t_cross = time[5] + (delta_t * 1e9).astype("timedelta64[ns]")
        return r_xyz, t_cross, velocity, delta_t

    def test_c_4_v_velocity_from_times(self):
        # Used to raise for every input, and the separation matrix was transposed
        r_xyz, t_cross, velocity, _ = self._plane_crossing()
        np.testing.assert_allclose(pyrf.c_4_v(r_xyz, t_cross), velocity, rtol=1e-6)

        t_sec = t_cross.astype(np.int64) * 1e-9
        np.testing.assert_allclose(pyrf.c_4_v(r_xyz, list(t_sec)), velocity, rtol=1e-6)

        # The caller's list of positions is unchanged
        self.assertTrue(all(isinstance(r, xr.DataArray) for r in r_xyz))

    def test_c_4_v_times_from_velocity(self):
        r_xyz, t_cross, velocity, delta_t = self._plane_crossing()
        result = pyrf.c_4_v(r_xyz, [t_cross[0], *velocity])
        np.testing.assert_allclose(result, delta_t, atol=1e-9)


class C4GradTestCase(unittest.TestCase):
    def test_c_4_grad_input(self):
        with self.assertRaises(TypeError):
            pyrf.c_4_grad(
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
            )
            pyrf.c_4_grad(
                [generate_data(100, tensor_order=1) for _ in range(4)],
                [generate_data(100, tensor_order=1) for _ in range(4)],
            )

        with self.assertRaises(ValueError):
            pyrf.c_4_grad([], [])

            pyrf.c_4_grad(
                [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)],
                [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)],
                "bazinga",
            )

    def test_c_4_grad_output(self):
        r_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
        n_mms = [generate_ts(64.0, 100, tensor_order=0) for _ in range(4)]

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
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
            )
            pyrf.c_4_j([], [])

    def test_c_4_j_output(self):
        r_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
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


@ddt
class CalcAgTestCase(unittest.TestCase):
    @data(0.0, generate_data(100))
    def test_calc_ag_input_type(self, inp):
        with self.assertRaises(TypeError):
            pyrf.calc_ag(inp)

    @data(
        generate_ts(64.0, 100, tensor_order=0), generate_ts(64.0, 100, tensor_order=1)
    )
    def test_calc_ag_input_value(self, inp):
        with self.assertRaises(ValueError):
            pyrf.calc_ag(inp)

    def test_calc_ag_output_type(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, tensor_order=2))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_ag_output_value(self):
        result = pyrf.calc_ag(generate_ts(64.0, 100, tensor_order=2))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)
        self.assertListEqual(list(result.dims), ["time"])
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class CalcAgyroTestCase(unittest.TestCase):
    def test_calc_agyro_input_type(self):
        self.assertIsNotNone(pyrf.calc_agyro(generate_ts(64.0, 100, tensor_order=2)))

        with self.assertRaises(TypeError):
            # Raises error if input is not a xarray
            pyrf.calc_agyro(0.0)
            pyrf.calc_agyro(generate_data(100))

    def test_calc_agyro_output_type(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, tensor_order=2))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_agyro_output_value(self):
        result = pyrf.calc_agyro(generate_ts(64.0, 100, tensor_order=2))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)
        self.assertListEqual(list(result.dims), ["time"])
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


class CalcDngTestCase(unittest.TestCase):
    def test_calc_dng_input_type(self):
        self.assertIsNotNone(pyrf.calc_dng(generate_ts(64.0, 100, tensor_order=2)))

        with self.assertRaises(TypeError):
            # Raises error if input is not a xarray
            pyrf.calc_dng(0.0)
            pyrf.calc_dng(generate_data(100))

    def test_calc_dng_output_type(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, tensor_order=2))

        # Output must be a xarray
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_dng_output_value(self):
        result = pyrf.calc_dng(generate_ts(64.0, 100, tensor_order=2))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)
        self.assertListEqual(list(result.dims), ["time"])
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


@ddt
class CalcDtTestCase(unittest.TestCase):
    @data(0, generate_data(100))
    def test_calc_dt_input_type(self, value):
        with self.assertRaises(TypeError):
            # Raises error if input is not a xarray
            pyrf.calc_dt(value)

    @data(
        generate_ts(64.0, 100, tensor_order=0),
        generate_ts(64.0, 100, tensor_order=1),
        generate_ts(64.0, 100, tensor_order=2),
    )
    def test_calc_dt_output_type(self, value):
        result = pyrf.calc_dt(value)
        self.assertIsInstance(result, float)


@ddt
class CalcFsTestCase(unittest.TestCase):
    @data(0, generate_data(100))
    def test_calc_fs_input_type(self, value):
        with self.assertRaises(TypeError):
            # Raises error if input is not a xarray
            pyrf.calc_fs(value)

    def test_calc_fs_output_type(self):
        self.assertIsInstance(pyrf.calc_fs(generate_ts(64.0, 100)), float)


@ddt
class CalcSqrtQTestCase(unittest.TestCase):
    @data(0, generate_data(100))
    def test_calc_sqrtq_input_type(self, inp):
        with self.assertRaises(TypeError):
            # Raises error if input is not a xarray
            pyrf.calc_sqrtq(inp)

    @data(
        generate_ts(64.0, 100, tensor_order=0), generate_ts(64.0, 100, tensor_order=1)
    )
    def test_calc_sqrtq_input_value(self, inp):
        with self.assertRaises(ValueError):
            # Raises error if input is not a xarray
            pyrf.calc_sqrtq(inp)

    def test_calc_sqrtq_output_type(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, tensor_order=2))
        self.assertIsInstance(result, xr.DataArray)

    def test_calc_sqrtq_output_value(self):
        result = pyrf.calc_sqrtq(generate_ts(64.0, 100, tensor_order=2))
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)
        self.assertListEqual(list(result.dims), ["time"])
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
            pyrf.cart2sph_ts(generate_data(100, tensor_order=1))
            pyrf.cart2sph_ts(generate_ts(64.0, 100, tensor_order=0))
            pyrf.cart2sph_ts(generate_ts(64.0, 100, tensor_order=1), 2)

    def test_cart2sph_ts_output(self):
        result = pyrf.cart2sph_ts(generate_ts(64.0, 100, tensor_order=1), 1)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cart2sph_ts(generate_ts(64.0, 100, tensor_order=1), -1)
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
            np.random.random((1000, 100)),
            np.arange(0, 1000, 10),
            random.randint(2, 100),
        )
        self.assertIsInstance(result, np.ndarray)


@ddt
class ConvertFACTestCase(unittest.TestCase):
    @data(
        (
            generate_data(100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1, 0, 0],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_data(100, tensor_order=1),
            [1, 0, 0],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            0,
        ),
    )
    @unpack
    def test_convert_fac_input_type(self, inp, b_bgd, r_xyz):
        with self.assertRaises(TypeError):
            pyrf.convert_fac(inp, b_bgd, r_xyz)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        ),
    )
    @unpack
    def test_convert_fac_input_shape(self, inp, b_bgd, r_xyz):
        with self.assertRaises(ValueError):
            pyrf.convert_fac(inp, b_bgd, r_xyz)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            None,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 98, tensor_order=1),
            None,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            np.random.random(3),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        ),
    )
    @unpack
    def test_convert_fac_output(self, inp, b_bgd, r_xyz):
        result = pyrf.convert_fac(inp, b_bgd, r_xyz)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class CotransTestCase(unittest.TestCase):
    @data(
        (0.0, "gse>gsm", True),
        (generate_data(100), "gse>gsm", True),
        (generate_ts(64.0, 100, tensor_order=1), "gsm", True),
        (generate_ts(64.0, 100, tensor_order=2), "gse>gsm", True),
        (
            generate_ts(64.0, 100, tensor_order=1, attrs={"COORDINATE_SYSTEM": "gse"}),
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

        inp = generate_ts(64.0, 100, tensor_order=1)
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
        inp = generate_ts(64.0, 100, tensor_order=0)
        result = pyrf.cotrans(inp, "gse>gsm", hapgood=True)
        self.assertIsInstance(result, xr.DataArray)
        result = pyrf.cotrans(inp, "dipoledirectiongse", hapgood=True)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class CrossTestCase(unittest.TestCase):
    @data(
        (generate_data(100, tensor_order=1), generate_ts(64.0, 100, tensor_order=1)),
        (generate_ts(64.0, 100, tensor_order=1), generate_data(100, tensor_order=1)),
    )
    @unpack
    def test_cross_input_type(self, inp0, inp1):
        with self.assertRaises(TypeError):
            pyrf.cross(inp0, inp1)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
        ),
    )
    @unpack
    def test_cross_input_shape(self, inp0, inp1):
        with self.assertRaises(ValueError):
            pyrf.cross(inp0, inp1)

    def test_cross_output(self):
        result = pyrf.cross(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

        result = pyrf.cross(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
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
        np.datetime64("2019-01-01T00:00:00.000000000"),
    )
    def test_datetime642unix_input(self, value):
        with self.assertRaises(TypeError):
            pyrf.datetime642unix(value)

    @data(
        [np.datetime64("2019-01-01T00:00:00.000000000")], generate_timeline(64.0, 100)
    )
    def test_datetime642unix_output(self, value):
        self.assertIsInstance(pyrf.datetime642unix(value), np.ndarray)


@ddt
class DecParPerpTestCase(unittest.TestCase):
    @data(
        (
            generate_data(100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            False,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_data(100, tensor_order=1),
            False,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            0,
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            False,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            False,
        ),
    )
    @unpack
    def test_dec_par_perp_input(self, inp, b_bgd, flag_spin_plane):
        with self.assertRaises(AssertionError):
            pyrf.dec_par_perp(inp, b_bgd, flag_spin_plane)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            False,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1) * 1e-4,
            False,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            True,
        ),
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
        (
            generate_data(100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            random.choice(["ions", "electrons"]),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_data(100, tensor_order=1),
            random.choice(["ions", "electrons"]),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            42,
        ),
    )
    @unpack
    def test_dynamic_press_input_type(self, n_s, v_xyz, specie):
        with self.assertRaises(TypeError):
            pyrf.dynamic_press(n_s, v_xyz, specie)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            random.choice(["ions", "electrons"]),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            random.choice(["ions", "electrons"]),
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            "I AM GROOT!!",
        ),
    )
    @unpack
    def test_dynamic_press_input_value(self, n_s, v_xyz, specie):
        with self.assertRaises(ValueError):
            pyrf.dynamic_press(n_s, v_xyz, specie)

    @data("ions", "electrons")
    def test_dynamic_press_output(self, value):
        result = pyrf.dynamic_press(
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            value,
        )
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, 1)


@ddt
class EVxBTestCase(unittest.TestCase):
    @data(
        (
            generate_data(100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            "vxb",
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_data(100, tensor_order=1),
            "vxb",
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            "bazinga",
        ),
    )
    @unpack
    def test_e_vxb_input(self, v_xyz, b_xyz, flag):
        with self.assertRaises((TypeError, AssertionError)):
            pyrf.e_vxb(v_xyz, b_xyz, flag)

    @data(
        (generate_ts(64.0, 100, tensor_order=1), "vxb"),
        (generate_ts(64.0, 100, tensor_order=1), "exb"),
    )
    @unpack
    def test_e_vxb_output(self, v_xyz, flag):
        result = pyrf.e_vxb(v_xyz, generate_ts(64.0, 100, tensor_order=1), flag)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100, 3])

    def test_e_vxb_values(self):
        # V = -400 km/s x, B = 5 nT z: E = -V x B = -2 mV/m y, and the E x B
        # drift of that field gives back V (the units label used to be mV/s)
        time = generate_timeline(1.0, 10)
        v_xyz = pyrf.ts_vec_xyz(time, np.tile([-400.0, 0.0, 0.0], (10, 1)))
        b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 5.0], (10, 1)))

        e_xyz = pyrf.e_vxb(v_xyz, b_xyz)
        np.testing.assert_allclose(e_xyz.data, np.tile([0.0, -2.0, 0.0], (10, 1)))
        self.assertEqual(e_xyz.attrs["UNITS"], "mV/m")

        v_exb = pyrf.e_vxb(e_xyz, b_xyz, "exb")
        np.testing.assert_allclose(v_exb.data, v_xyz.data)
        self.assertEqual(v_exb.attrs["UNITS"], "km/s")


@ddt
class EbNRFTestCase(unittest.TestCase):
    @data("a", "b", np.random.random(3))
    def test_eb_nrf_output(self, value):
        result = pyrf.eb_nrf(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            value,
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EdbTestCase(unittest.TestCase):
    @data("e.b=0", "e_perp+nan", "e_par")
    def test_edb_output(self, value):
        pyrf.edb(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            random.random() * 90,
            value,
        )


@ddt
class EbspTestCase(unittest.TestCase):
    @data(
        (
            None,
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 98, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 120, tensor_order=1),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(99.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(40.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(40.0, 100, tensor_order=1),
            generate_ts(40.0, 100, tensor_order=1),
            generate_ts(40.0, 100, tensor_order=1),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            [1e0, 1e1],
            {},
        ),
        (
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            generate_ts(64.0, 97, tensor_order=1),
            [1e0, 1e1],
            {"fac_matrix": generate_ts(64.0, 100, tensor_order=2)},
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
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
            generate_ts(64.0, 100, tensor_order=1),
            None,
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            None,
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            None,
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e1, 1e0],
        ),
        (
            generate_ts(64.0, 100, tensor_order=1)[:, :2],
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
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
            "fac_matrix": generate_ts(64.0, 100, tensor_order=2),
            "m_width_coeff": 1,
        },
        {
            "polarization": False,
            "no_resample": False,
            "fac": True,
            "de_dot_b0": False,
            "full_b_db": False,
            "nav": 8,
            "fac_matrix": generate_ts(64.0, 100, tensor_order=2),
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
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            [1e0, 1e1],
            **value,
        )

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["bb_xxyyzzss"], xr.DataArray)

    @data("pc12", "pc35", [1e0, 1e1])
    def test_ebsp_freq_int_pass(self, value):
        self.assertIsNotNone(
            pyrf.ebsp(
                generate_ts(64.0, 100000, tensor_order=1),
                generate_ts(64.0, 100000, tensor_order=1),
                generate_ts(64.0, 100000, tensor_order=1),
                generate_ts(64.0, 100000, tensor_order=1),
                generate_ts(64.0, 100000, tensor_order=1),
                value,
            )
        )

    @data(random.random(), np.random.random(3), "bazinga", [1, 100])
    def test_ebsp_freq_int_fail(self, value):
        with self.assertRaises((AssertionError, ValueError)):
            pyrf.ebsp(
                generate_ts(64.0, 10000, tensor_order=1),
                generate_ts(64.0, 10000, tensor_order=1),
                generate_ts(64.0, 10000, tensor_order=1),
                generate_ts(64.0, 10000, tensor_order=1),
                generate_ts(64.0, 10000, tensor_order=1),
                value,
            )

    @data(([0.32, 3.2], generate_ts(64.0, 100000, tensor_order=1)))
    @unpack
    def test_average_data(self, freq_int, data):
        _, _, _, out_time = _freq_int(freq_int, data)
        in_time = data.time.data.astype(np.float64) / 1e9

        result = _average_data.__wrapped__(data.data, in_time, out_time, None)
        self.assertIsInstance(result, np.ndarray)
        self.assertListEqual(list(result.shape), [len(out_time), 3])

    @data(([0.32, 3.2], generate_ts(64.0, 100000, tensor_order=1)))
    @unpack
    def test_censure_plot(self, freq_int, data):
        _, _, out_sampling, out_time = _freq_int(freq_int, data)
        a_ = np.logspace(1, 2, 12)
        idx_nan = np.full(len(data), False)
        idx_nan[np.random.randint(100000, size=100)] = True
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
            generate_ts(64.0, 100, tensor_order=0),
            0,
            1,
            -1,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            0,
            1,
            -1,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            0,
            random.randint(2, 22),
            -1,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            0,
            random.randint(2, 22),
            random.choice(range(1, 10, 2)),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            random.randint(2, 22),
            0,
            -1,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            random.randint(2, 22),
            0,
            random.choice(range(1, 10, 2)),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            random.randint(2, 11),
            random.randint(12, 22),
            -1,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
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
        generate_ts(64.0, 100, tensor_order=0),
        generate_ts(64.0, 100, tensor_order=1),
        generate_ts(64.0, 100, tensor_order=2),
    )
    def test_gradient_output(self, value):
        value.attrs = {"UNITS": "bazinga"}
        result = pyrf.gradient(value)
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), list(value.shape))


@ddt
class Gse2GsmTestCase(unittest.TestCase):
    @data(
        (generate_data(100, tensor_order=1), "gse>gsm"),
        (generate_ts(64.0, 100, tensor_order=0), "gse>gsm"),
        (generate_ts(64.0, 100, tensor_order=1), "bazinga"),
    )
    @unpack
    def test_gse2gsm_input(self, inp, flag):
        with self.assertRaises(AssertionError):
            pyrf.gse2gsm(inp, flag)

    @data(
        (generate_ts(64.0, 100, tensor_order=1), "gse>gsm"),
        (generate_ts(64.0, 100, tensor_order=1), "gsm>gse"),
    )
    @unpack
    def test_gse2gsm_output(self, inp, flag):
        pyrf.gse2gsm(inp, flag)

    @data("GSE>GSM", "Gse>Gsm")
    def test_gse2gsm_flag_case(self, flag):
        # Upper-case flags passed the validation but were rejected by cotrans
        inp = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 10),
            np.tile([1.0, 2.0, 3.0], (10, 1)),
            attrs={"COORDINATE_SYSTEM": "gse"},
        )
        expected = pyrf.gse2gsm(inp, "gse>gsm")
        np.testing.assert_allclose(pyrf.gse2gsm(inp, flag).data, expected.data)

        # Round trip back to GSE
        np.testing.assert_allclose(
            pyrf.gse2gsm(expected, "GSM>GSE").data, inp.data, atol=1e-12
        )


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

    @data(
        (30, (30, 30)),
        (np.int64(30), (30, 30)),
        (np.array(30), (30, 30)),
        ([20, 50], (20, 50)),
        ((20, 50), (20, 50)),
        (np.array([20, 50]), (20, 50)),
        (np.array([50, 20]), (50, 20)),
        (np.linspace(0, 1, 11), (10, 10)),
        ([np.linspace(0, 1, 11), np.linspace(0, 1, 6)], (10, 5)),
        (np.array([np.linspace(0, 1, 11)] * 2), (10, 10)),
    )
    @unpack
    def test_histogram2d_bins_shape(self, bins, shape):
        # A numpy array [nx, ny] used to be read as the edges of one bin (1x1
        # output), or to raise when nx > ny, so the output test was flaky.
        result = pyrf.histogram2d(
            generate_ts(64.0, 1000), generate_ts(64.0, 1000), bins
        )
        self.assertTupleEqual(result.shape, shape)


@ddt
class IncrementsTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, tensor_order=0),
        generate_ts(64.0, 100, tensor_order=1),
        generate_ts(64.0, 100, tensor_order=2),
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
        phi_grid = np.arange(-180.0, 180.0, 42)

        with self.assertRaises((RuntimeError, NotImplementedError)):
            pyrf.int_sph_dist(vdf, speed, phi, theta, speed_grid, phi_grid, **value)

    @data(
        {},
        {"weight": "lin"},
        {"weight": "log"},
        {"velocity_edges": np.linspace(-0.01, 1.01, 52)},
        {"velocity_grid_edges": np.linspace(-1.01, 1.01, 102)},
        {"projection_base": "cart", "projection_dim": "2d"},
        {"projection_base": "cart", "projection_dim": "3d"},
    )
    def test_int_sph_dist_output(self, value):
        vdf = np.random.random((51, 32, 16))
        speed = np.linspace(1, 2, 51)
        phi = np.arange(32)
        theta = np.arange(16)
        speed_grid = np.linspace(-1, 1, 101)
        d_phi_g = 2 * np.pi / 32
        phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, 32) + d_phi_g / 2

        result = pyrf.int_sph_dist(
            vdf, speed, phi, theta, speed_grid, phi_grid, **value
        )
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
        n_threads = numba.get_num_threads()
        result = _mc_pol_1d.__wrapped__(vdf, *args, n_threads)
        self.assertIsInstance(result, np.ndarray)

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
        v_step, n_threads = _uniform_step(args[7]), numba.get_num_threads()
        result = _mc_cart_2d.__wrapped__(vdf, *args, v_step, n_threads)
        self.assertIsInstance(result, np.ndarray)

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
        v_step, n_threads = _uniform_step(args[7]), numba.get_num_threads()
        result = _mc_cart_3d.__wrapped__(vdf, *args, v_step, n_threads)
        self.assertIsInstance(result, np.ndarray)

    @data(_mc_cart_2d, _mc_cart_3d)
    def test_mc_cart_fast_index_matches_searchsorted(self, kernel):
        # v_step > 0 indexes the grid arithmetically, v_step = 0 falls back to
        # np.searchsorted; with the same random draws both must give the same bins.
        rng = np.random.default_rng(0)
        vdf = rng.random((51, 32, 16))
        vdf[vdf < 1e-2] = 0
        edges = np.linspace(-1.01, 1.01, 102)
        args = (
            np.linspace(0, 1, 51),
            np.arange(32),
            np.arange(16),
            np.ones(51) * 0.02,
            np.ones(51) * 0.01,
            np.ones(32),
            np.ones(16),
            edges,
            0.02**2,
            np.array([-np.inf, np.inf]),
            np.array([-np.pi, np.pi]),
            np.ones((51, 32, 16), dtype=int) * 10,
            np.eye(3),
        )
        results = []
        for v_step in [_uniform_step(edges), 0.0]:
            random.seed(0)
            results.append(kernel.__wrapped__(vdf, *args, v_step, 1))

        self.assertGreater(_uniform_step(edges), 0.0)
        self.assertGreater(np.count_nonzero(results[0]), 0)
        np.testing.assert_array_equal(results[0], results[1])

    @staticmethod
    def _maxwellian(v_d=(300e3, -200e3, 100e3), t_ev=1000.0):
        # Drifting proton Maxwellian (n = 1 cm^-3) on FPI-like bins: 32
        # log-spaced energies (10 eV - 30 keV), 32 azimuths, 16 elevations.
        q_e, m_p = 1.602176634e-19, 1.67262192369e-27
        energy = 10.0 * 3000.0 ** (np.arange(32) / 31)
        speed = np.sqrt(2 * q_e * energy / m_p)
        phi = np.deg2rad(5.625 + 11.25 * np.arange(32))
        theta = np.deg2rad(-84.375 + 11.25 * np.arange(16))
        v_x = speed[:, None, None] * np.cos(theta) * np.cos(phi)[:, None]
        v_y = speed[:, None, None] * np.cos(theta) * np.sin(phi)[:, None]
        v_z = speed[:, None, None] * np.sin(theta) * np.ones_like(phi)[:, None]
        v_th2 = 2 * q_e * t_ev / m_p
        dv2 = (v_x - v_d[0]) ** 2 + (v_y - v_d[1]) ** 2 + (v_z - v_d[2]) ** 2
        vdf = 1e6 / (np.pi * v_th2) ** 1.5 * np.exp(-dv2 / v_th2)  # s^3/m^6
        return vdf, energy, speed, phi, theta

    @staticmethod
    def _moments(out, n_dim):
        # Density and bulk velocity of the projected distribution
        edges = out["vx_edges"]
        d_v = np.diff(edges)
        v_c = edges[:-1] + d_v / 2
        vol = np.prod(np.meshgrid(*[d_v] * n_dim, indexing="ij"), axis=0)
        n = np.sum(out["f"] * vol)
        v = []
        for axis in range(n_dim):
            shape = [1] * n_dim
            shape[axis] = -1
            v.append(np.sum(out["f"] * v_c.reshape(shape) * vol) / n)

        return n, np.array(v)

    @data(("pol", "1d", 20), ("cart", "2d", 20), ("cart", "3d", 5))
    @unpack
    def test_int_sph_dist_drifting_maxwellian(self, base, dim, n_mc):
        # The Monte-Carlo speeds were drawn in [v - 1.5 dv, v - 0.5 dv] and the
        # speed bin width was the spacing to the lower channel: n was 6 % and
        # the bulk velocity 7 % too low (as in irfu-matlab's irf_int_sph_dist).
        v_d = np.array([300e3, -200e3, 100e3])
        vdf, _, speed, phi, theta = self._maxwellian(v_d)
        d_phi_g = 2 * np.pi / 32
        phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, 32) + d_phi_g / 2
        n_grid = {"1d": 301, "2d": 101, "3d": 31}[dim]
        edges = np.linspace(-1500e3, 1500e3, n_grid + 1)

        random.seed(0)
        out = pyrf.int_sph_dist(
            vdf,
            speed,
            phi,
            theta,
            None,
            phi_grid,
            projection_base=base,
            projection_dim=dim,
            velocity_grid_edges=edges,
            n_mc=n_mc,
        )
        n_dim = int(dim[0])
        n, v = self._moments(out, n_dim)
        self.assertAlmostEqual(n / 1e6, 1.0, delta=0.01)
        np.testing.assert_allclose(v / 1e3, v_d[:n_dim] / 1e3, atol=3.0)

    def test_int_sph_dist_speed_widths(self):
        # Explicit speed widths from the true (geometric) channel edges give the
        # same result as the default edges, and as the equivalent velocity_edges.
        # The jitted kernels' random numbers can't be seeded from Python, so
        # compare the moments: n is set by the bin volumes, V agrees within noise.
        v_d = np.array([300e3, -200e3, 100e3])
        vdf, _, speed, phi, theta = self._maxwellian(v_d)
        e_edges = 10.0 * 3000.0 ** ((np.arange(33) - 0.5) / 31)
        v_edges = np.sqrt(2 * 1.602176634e-19 * e_edges / 1.67262192369e-27)
        np.testing.assert_allclose(_speed_bin_edges(speed), v_edges, rtol=1e-12)

        grid = np.linspace(-1500e3, 1500e3, 301)
        for options in [
            {},
            {"velocity_edges": v_edges},
            {"d_v_m": speed - v_edges[:-1], "d_v_p": v_edges[1:] - speed},
        ]:
            out = pyrf.int_sph_dist(
                vdf, speed, phi, theta, grid, None, n_mc=20, **options
            )
            n, v = self._moments(out, 1)
            self.assertAlmostEqual(n / 1e6, 1.0, delta=0.01)
            self.assertAlmostEqual(v[0] / 1e3, v_d[0] / 1e3, delta=3.0)

    def test_speed_bin_edges(self):
        # Geometric midpoints for log-spaced speeds
        edges = _speed_bin_edges(np.array([1.0, 2.0, 4.0]))
        np.testing.assert_allclose(edges, np.sqrt(2) * np.array([0.5, 1, 2, 4]))

        # Arithmetic midpoints (lower edge clipped at 0) with a zero speed
        edges = _speed_bin_edges(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(edges, [0.0, 0.5, 1.5, 2.5])

        with self.assertRaises(ValueError):
            _speed_bin_edges(np.array([1.0]))


@ddt
class IntegrateTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, tensor_order=0), generate_ts(64.0, 100, tensor_order=1)
    )
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
        generate_ts(64.0, 10000, tensor_order=0),
        generate_ts(64.0, 10000, tensor_order=1),
        generate_ts(64.0, 10000, tensor_order=2),
    )
    def test_lowpass_output(self, value):
        pyrf.lowpass(value, random.random(), 64.0)


@ddt
class LShellTestCase(unittest.TestCase):
    @data("gei", "geo", "gse", "gsm", "mag", "sm")
    def test_l_shell_output(self, value):
        result = pyrf.l_shell(
            generate_ts(64.0, 100, tensor_order=1, attrs={"COORDINATE_SYSTEM": value})
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class MeanTestCase(unittest.TestCase):
    @data(None, generate_ts(64.0, 100, tensor_order=1))
    def test_mean_output(self, value):
        result = pyrf.mean(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
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
        result = pyrf.mva(generate_ts(64.0, 100, tensor_order=1), method)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)


@ddt
class NewXyzTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, tensor_order=1), generate_ts(64.0, 100, tensor_order=2)
    )
    def test_new_xyz_output(self, inp):
        result = pyrf.new_xyz(inp, np.random.random((3, 3)))
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, inp.ndim)


class NormTestCase(unittest.TestCase):
    def test_norm_output(self):
        result = pyrf.norm(generate_ts(64.0, 100, tensor_order=1))
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)


@ddt
class PlasmaBetaTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, tensor_order=1), generate_ts(64.0, 100, tensor_order=2))
    )
    @unpack
    def test_plasma_beta_output(self, b_xyz, p_xyz):
        result = pyrf.plasma_beta(b_xyz, p_xyz)
        self.assertIsInstance(result, xr.DataArray)

    @data([6.0, 0.0, 8.0], [10.0, 0.0, 0.0])
    def test_plasma_beta_values(self, b_vec):
        # |B| = 10 nT -> P_b = 0.0397887 nPa. The trace of P is 3 x 0.0398 nPa, so
        # beta = 1.000283 (was 1e9 too large with P in nPa and P_b in Pa).
        time = generate_timeline(1.0, 10)
        b_xyz = pyrf.ts_vec_xyz(time, np.tile(b_vec, (10, 1)))
        p_mat = np.array([[0.06, 0.01, 0.0], [0.01, 0.03, 0.0], [0.0, 0.0, 0.0294]])
        p_xyz = pyrf.ts_tensor_xyz(time, np.tile(p_mat, (10, 1, 1)))
        result = pyrf.plasma_beta(b_xyz, p_xyz)
        np.testing.assert_allclose(result.data, 1.000283, rtol=1e-6)


@ddt
class StructFuncTestCase(unittest.TestCase):
    @data(
        (generate_ts(64.0, 100, tensor_order=0), None, random.randint(1, 100)),
        (generate_ts(64.0, 100, tensor_order=1), None, random.randint(1, 100)),
        (generate_ts(64.0, 100, tensor_order=2), None, random.randint(1, 100)),
        (
            generate_ts(64.0, 100, tensor_order=1),
            np.random.randint([1] * 50, [50] * 50),
            1,
        ),
    )
    @unpack
    def test_struct_func_output(self, inp, scales, order):
        result = pyrf.struct_func(inp, scales, order)
        self.assertIsInstance(result, xr.DataArray)


class TraceTestCase(unittest.TestCase):
    def test_trace_input_type(self):
        with self.assertRaises(TypeError):
            pyrf.trace(generate_data(100, tensor_order=2))

    def test_trace_input_value(self):
        with self.assertRaises(ValueError):
            pyrf.trace(generate_ts(64.0, 100, tensor_order=random.randint(0, 1)))

    def test_trace_output(self):
        result = pyrf.trace(generate_ts(64.0, 100, tensor_order=2))
        self.assertIsInstance(result, xr.DataArray)
        self.assertListEqual(list(result.shape), [100])


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
            [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)],
            [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)],
            [generate_ts(64.0, 100, tensor_order=2) for _ in range(4)],
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)


class PlasmaCalcTestCase(unittest.TestCase):
    def test_plasma_calc_output(self):
        result = pyrf.plasma_calc(
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
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
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        ),
        (
            generate_ts(128.0, 200, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(128.0, 200, tensor_order=1),
        ),
    )
    @unpack
    def test_poynting_flux_output(self, e_xyz, b_xyz):
        result = pyrf.poynting_flux(e_xyz, b_xyz, None)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)

        result = pyrf.poynting_flux(
            e_xyz, b_xyz, generate_ts(64.0, 100, tensor_order=1)
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)
        self.assertIsInstance(result[2], xr.DataArray)


class PresAnisTestCase(unittest.TestCase):
    def test_pres_anis_output(self):
        result = pyrf.pres_anis(
            generate_ts(64.0, 100, tensor_order=2),
            generate_ts(64.0, 100, tensor_order=1),
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
                    "b_u": np.random.random(3),
                    "b_d": np.random.random(3),
                    "v_u": np.random.random(3),
                    "v_d": np.random.random(3),
                    "n_u": random.random(),
                    "n_d": random.random(),
                    "r_xyz": random.random(),
                }
            )

    @data(
        {
            "b_u": np.random.random(3),
            "b_d": np.random.random(3),
            "v_u": np.random.random(3),
            "v_d": np.random.random(3),
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
            "b_u": np.random.random(3),
            "b_d": np.random.random(3),
            "v_u": np.random.random(3),
            "v_d": np.random.random(3),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": np.random.random(3),
        },
        {
            "b_u": np.random.random(3),
            "b_d": np.random.random(3),
            "v_u": np.random.random(3),
            "v_d": np.random.random(3),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": generate_ts(64.0, 100, tensor_order=1),
        },
        {
            "b_u": np.random.random(3),
            "b_d": np.random.random(3),
            "v_u": np.random.random(3),
            "v_d": np.random.random(3),
            "n_u": random.random(),
            "n_d": random.random(),
            "r_xyz": generate_ts(64.0, 100, tensor_order=1),
            "d2u": random.choice([-1, 1]),
            "dt_f": random.random(),
            "f_cp": random.random(),
        },
    )
    def test_shock_normal_ouput(self, value):
        result = pyrf.shock_normal(value)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["v_sh"], dict)

    def test_shock_normal_model_on_bow_shock(self):
        # Spacecraft at the nose of the Farris et al. (1991) bow shock model
        # (eps = 0.81, L = 24.8 R_E, aberration 3.8 deg): sigma = 1 and the model
        # normal is the aberrated x axis. The model parameters used to be
        # unpacked in the wrong order.
        eps, l_bs, alpha = 0.81, 24.8, np.deg2rad(3.8)
        rot = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0.0],
                [np.sin(alpha), np.cos(alpha), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        r_xyz = rot.T @ np.array([l_bs / (1 + eps), 0.0, 0.0]) * 6371.0
        spec = {
            "b_u": np.array([2.0, 3.0, 1.0]),
            "b_d": np.array([4.0, 12.0, 3.0]),
            "v_u": np.array([-400.0, 10.0, 0.0]),
            "v_d": np.array([-100.0, 5.0, 0.0]),
            "n_u": 5.0,
            "n_d": 15.0,
            "r_xyz": list(r_xyz),
        }
        result = pyrf.shock_normal(spec)

        self.assertAlmostEqual(result["info"]["sig"]["farris"], 1.0, delta=1e-3)
        n_expected = rot.T @ np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            abs(float(np.dot(result["n"]["farris"], n_expected))), 1.0, delta=1e-4
        )

    def test_shock_normal_leq90(self):
        # 120 deg folds to 60 deg (used to give 90 - 120 = -30 deg)
        n_vec = {"x": np.array([1.0, 0.0, 0.0])}
        b_u = np.array([np.cos(np.deg2rad(120.0)), np.sin(np.deg2rad(120.0)), 0.0])
        self.assertAlmostEqual(_shock_angle({"b_u": b_u}, n_vec, "b", True)["x"], 60.0)
        self.assertAlmostEqual(
            _shock_angle({"b_u": b_u}, n_vec, "b", False)["x"], 120.0
        )


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

    def test_shock_parameters_values(self):
        # Default (spacecraft) frame used to raise KeyError: 'v_sh', and r_cp was
        # 1000x too small (v in km/s used as m/s)
        spec = {
            "b_u": np.array([0.0, 0.0, 5.0]),
            "b_d": np.array([0.0, 0.0, 15.0]),
            "n_u": 5.0,
            "n_d": 15.0,
            "v_u": np.array([-400.0, 0.0, 0.0]),
            "v_d": np.array([-133.0, 0.0, 0.0]),
        }
        spec_ref = dict(spec)
        result = pyrf.shock_parameters(spec)

        # r = m_p v / (e B) = 835 km for 400 km/s in 5 nT (in m)
        self.assertAlmostEqual(result["r_cp_u"], 835.2e3, delta=1e3)
        # V_A = 48.8 km/s for 5 nT and 5 cm^-3
        self.assertAlmostEqual(result["m_a_u"], 400e3 / result["v_a_u"], places=6)
        self.assertAlmostEqual(result["v_a_u"], 48.77e3, delta=0.1e3)

        # The caller's dict is unchanged
        self.assertListEqual(sorted(spec), sorted(spec_ref))

    def test_shock_parameters_single_region(self):
        # A single region used to be discarded (KeyError: 'b')
        spec = {
            "b_u": np.array([0.0, 0.0, 5.0]),
            "n_u": 5.0,
            "v_u": np.array([-400.0, 0.0, 0.0]),
        }
        result = pyrf.shock_parameters(spec)
        self.assertListEqual(
            sorted(result), ["f_cp_u", "l_i_u", "m_a_u", "r_cp_u", "v_a_u"]
        )


@ddt
class SolidAngleTestCase(unittest.TestCase):
    @data(
        tuple(np.random.random(3) for _ in range(3)),
        tuple(generate_data(100, tensor_order=1) for _ in range(3)),
        tuple(generate_ts(64.0, 100, tensor_order=1) for _ in range(3)),
    )
    @unpack
    def test_solid_angle_ouput(self, inp0, inp1, inp2):
        result = pyrf.solid_angle(inp0, inp1, inp2)
        self.assertIsInstance(result, np.ndarray)


@ddt
class Sph2CartTestCase(unittest.TestCase):
    @data(
        tuple(generate_data(100, tensor_order=0) for _ in range(3)),
        tuple(generate_ts(64.0, 100, tensor_order=0) for _ in range(3)),
    )
    @unpack
    def test_sph2cart_output(self, azimuth, elevation, r):
        self.assertIsNotNone(pyrf.sph2cart(azimuth, elevation, r))


class StartTestCase(unittest.TestCase):
    def test_start_input_type(self):
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, tensor_order=0)))
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, tensor_order=1)))
        self.assertIsNotNone(pyrf.start(generate_ts(64.0, 100, tensor_order=2)))

        with self.assertRaises(AssertionError):
            pyrf.start(0)
            pyrf.start(generate_timeline(64.0, 100))

    def test_start_output(self):
        result = pyrf.start(generate_ts(64.0, 100, tensor_order=0))
        self.assertIsInstance(result, np.float64)
        self.assertEqual(
            np.datetime64(int(result * 1e9), "ns"),
            np.datetime64("2019-01-01T00:00:00.000"),
        )


@ddt
class TsAppendTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, tensor_order=0),
        generate_ts(64.0, 100, tensor_order=1),
        generate_ts(64.0, 100, tensor_order=2),
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


@ddt
class TsTimeTestCase(unittest.TestCase):
    def test_ts_skymap_input_type(self):
        with self.assertRaises(AssertionError):
            pyrf.ts_time(np.datetime64("1789-07-14T00:00:00.000000000"), {})

    @data(generate_timeline(64.0, 100, dtype=np.int64))
    def test_ts_time_inpu_datatype(self, timeline):
        with self.assertRaises(TypeError):
            pyrf.ts_time(timeline, {})

    @data(
        generate_timeline(64.0, 100, dtype=np.float64) / 1e9,
        generate_timeline(64.0, 100, dtype=np.datetime64),
    )
    def test_ts_time_output(self, timeline):
        result = pyrf.ts_time(timeline, {})
        self.assertIsInstance(result, xr.DataArray)


@ddt
class TsSkymapTestCase(unittest.TestCase):
    @data(
        (
            0,
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            np.random.random(16),
        ),
        (
            generate_timeline(64.0, 100),
            0,
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            np.random.random(16),
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            0,
            np.random.random((100, 32)),
            np.random.random(16),
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            0,
            np.random.random(16),
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random((100, 32, 32, 16)),
            np.random.random((100, 32)),
            np.random.random((100, 32)),
            0,
        ),
    )
    @unpack
    def test_ts_skymap_input_type(self, time, data, energy, phi, theta):
        with self.assertRaises(TypeError):
            pyrf.ts_skymap(time, data, energy, phi, theta)

    @data(
        (0, np.random.random(32), np.zeros(100)),
        (np.random.random(32), 0, np.zeros(100)),
        (np.random.random(32), np.random.random(32), 0),
    )
    @unpack
    def test_ts_skymap_input_optionals(self, energy0, energy1, esteptable):
        with self.assertRaises(TypeError):
            pyrf.ts_skymap(
                generate_timeline(64.0, 100),
                np.random.random((100, 32, 32, 16)),
                np.random.random((100, 32)),
                np.random.random((100, 32)),
                np.random.random(16),
                energy0=energy0,
                energy1=energy1,
                esteptable=esteptable,
            )

    @data((0, None, None), (None, 0, None), (None, None, 0))
    @unpack
    def test_ts_skymap_input_attrs(self, attrs, glob_attrs, coords_attrs):
        with self.assertRaises(TypeError):
            pyrf.ts_skymap(
                generate_timeline(64.0, 100),
                np.random.random((100, 32, 32, 16)),
                np.random.random((100, 32)),
                np.random.random((100, 32)),
                np.random.random(16),
                attrs=attrs,
                glob_attrs=glob_attrs,
                coords_attrs=coords_attrs,
            )

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


@ddt
class TsScalarTestCase(unittest.TestCase):
    @data(
        (0.0, generate_data(100, tensor_order=0), {}),
        (generate_data(100, tensor_order=0), 0.0, {}),
        (
            generate_data(100, tensor_order=0),
            generate_data(100, tensor_order=0),
            "bazinga!!",
        ),
    )
    @unpack
    def test_ts_scalar_input_type(self, time, data, attrs):
        with self.assertRaises(TypeError):
            pyrf.ts_scalar(time, data, attrs=attrs)

    @data(
        generate_data(99, tensor_order=0),
        generate_data(100, tensor_order=random.randint(1, 3)),
    )
    def test_ts_scalar_input_shape(self, data):
        with self.assertRaises(ValueError):
            pyrf.ts_scalar(generate_timeline(64.0, 100), data)

    def test_ts_scalar_output_type(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=0)
        )
        self.assertIsInstance(result, xr.DataArray)

    def test_ts_scalar_output_shape(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=0)
        )
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 100)

    def test_ts_scalar_dims(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=0)
        )
        self.assertListEqual(list(result.dims), ["time"])

    def test_ts_scalar_meta(self):
        result = pyrf.ts_scalar(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=0)
        )
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


@ddt
class TsSpectrTestCase(unittest.TestCase):
    @data(
        (0, np.random.random(10), np.random.random((100, 10)), "energy", None),
        (generate_timeline(64.0, 100), 0, np.random.random((100, 10)), "energy", None),
        (generate_timeline(64.0, 100), np.random.random(10), 0, "energy", None),
    )
    @unpack
    def test_ts_spectr_input_type(self, time, ener, data, comp_name, attrs):
        with self.assertRaises(TypeError):
            pyrf.ts_spectr(time, ener, data, comp_name, attrs)

    @data(
        (
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random(100),
            "energy",
            None,
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random((98, 10)),
            "energy",
            None,
        ),
        (
            generate_timeline(64.0, 100),
            np.random.random((98, 10)),
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
    def test_ts_spectr_input_value(self, time, ener, data, comp_name, attrs):
        with self.assertRaises(ValueError):
            pyrf.ts_spectr(time, ener, data, comp_name, attrs)

    def test_ts_spectr_output(self):
        result = pyrf.ts_spectr(
            generate_timeline(64.0, 100),
            np.random.random(10),
            np.random.random((100, 10)),
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class TsVecXYZTestCase(unittest.TestCase):
    @data(
        (list(generate_timeline(64.0, 100)), generate_data(100, 1), {}),
        (generate_timeline(64.0, 100), list(generate_data(100, 1)), {}),
        (generate_timeline(64.0, 100), generate_data(100, 1), "bazinga!"),
    )
    @unpack
    def test_ts_vec_xyz_input_type(self, time, data, attrs):
        with self.assertRaises(TypeError):
            pyrf.ts_vec_xyz(time, data, attrs)

    @data(
        generate_data(99, tensor_order=1),
        generate_data(100, tensor_order=random.randint(2, 3)),
    )
    def test_ts_vec_xyz_input_shape(self, data):
        with self.assertRaises(ValueError):
            pyrf.ts_vec_xyz(generate_timeline(64.0, 100), data)

    def test_ts_vec_xyz_output_type(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=1)
        )
        self.assertIsInstance(result, xr.DataArray)

    def test_ts_vec_xyz_output_shape(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=1)
        )
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)

    def test_ts_vec_xyz_dims(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=1)
        )
        self.assertListEqual(list(result.dims), ["time", "comp"])

    def test_ts_vec_xyz_meta(self):
        result = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=1)
        )
        self.assertEqual(result.attrs["TENSOR_ORDER"], 1)


@ddt
class TsTensorXYZTestCase(unittest.TestCase):
    @data(
        (0.0, generate_data(100, tensor_order=0)),
        (generate_timeline(64.0, 100), 0.0),
    )
    @unpack
    def test_ts_tensor_xyz_input_type(self, time, data):
        with self.assertRaises(TypeError):
            pyrf.ts_tensor_xyz(time, data)

    @data(
        (generate_timeline(64.0, 101), generate_data(100, tensor_order=2)),
        (generate_timeline(64.0, 100), generate_data(100, tensor_order=0)),
        (generate_timeline(64.0, 100), generate_data(100, tensor_order=1)),
    )
    @unpack
    def test_ts_tensor_xyz_input_value(self, time, data):
        with self.assertRaises(ValueError):
            # Raises error if data and timeline don't have the same size
            pyrf.ts_tensor_xyz(time, data)

    def test_ts_tensor_xyz_output(self):
        result = pyrf.ts_tensor_xyz(
            generate_timeline(64.0, 100), generate_data(100, tensor_order=2)
        )
        # Check if the output is a DataArray
        self.assertIsInstance(result, xr.DataArray)

        # Check that the output has the correct shape
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], 3)

        # Check that the output has the correct dimensions
        self.assertListEqual(list(result.dims), ["time", "rcomp", "ccomp"])

        # Check that the output has the correct metadata
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
        (generate_data(100, tensor_order=1), {}),
        (generate_ts(64.0, 100, tensor_order=1), {"linear": [random.randint(10, 100)]}),
    )
    @unpack
    def test_wavelet_input_type(self, inp, options):
        with self.assertRaises(TypeError):
            pyrf.wavelet(inp, **options)

    @data(
        (generate_ts(64.0, 100, tensor_order=2), {}),
    )
    @unpack
    def test_wavelet_input_value(self, inp, options):
        with self.assertRaises(ValueError):
            pyrf.wavelet(inp, **options)

    @data(
        (generate_ts(64.0, 100, tensor_order=0), None, True, None),
        (generate_ts(64.0, 101, tensor_order=0), None, True, None),
        (generate_ts(64.0, 100, tensor_order=1), None, True, None),
        (
            generate_ts(64.0, 100, tensor_order=0),
            [random.random(), random.random()],
            True,
            None,
        ),
        (generate_ts(64.0, 100, tensor_order=0), None, False, None),
        (generate_ts(1024.0, 100, tensor_order=0), None, True, True),
        (generate_ts(64.0, 100, tensor_order=0), None, True, False),
        (generate_ts(64.0, 100, tensor_order=0), None, True, random.uniform(1, 30)),
    )
    @unpack
    def test_wavelet_output(self, inp, f, return_power, linear):
        self.assertIsNotNone(
            pyrf.wavelet(inp, f=f, return_power=return_power, linear=linear)
        )

    @staticmethod
    def _sine(f_s=128.0, n_pts=4096, f_0=20.0):
        time = generate_timeline(f_s, n_pts)
        return pyrf.ts_scalar(time, np.sin(2 * np.pi * f_0 * np.arange(n_pts) / f_s))

    def test_wavelet_default_frequencies(self):
        # Default range used to reach 100x Nyquist (half of the bins above it)
        result = pyrf.wavelet(self._sine())
        self.assertAlmostEqual(result.frequency.data.max(), 64.0)
        self.assertAlmostEqual(result.frequency.data.min(), 0.64)
        power = np.nanmean(result.data, axis=0)
        self.assertAlmostEqual(result.frequency.data[np.argmax(power)], 20.0, delta=1.0)

    def test_wavelet_f_max_clipped_to_nyquist(self):
        result = pyrf.wavelet(self._sine(), f=[1.0, 200.0])
        self.assertAlmostEqual(result.frequency.data.max(), 64.0)

    def test_wavelet_linear_values(self):
        # bool is a subclass of int: True used to give 1 Hz spacing, False crashed
        inp = self._sine(f_s=1024.0)
        freqs = pyrf.wavelet(inp, linear=True).frequency.data
        np.testing.assert_allclose(np.abs(np.diff(freqs)), 100.0)
        self.assertEqual(len(pyrf.wavelet(inp, linear=False).frequency), 200)
        freqs = pyrf.wavelet(inp, linear=16).frequency.data
        np.testing.assert_allclose(np.abs(np.diff(freqs)), 16.0)

    @data(True, 100.0, 0, -1.0)
    def test_wavelet_linear_invalid(self, linear):
        # 100 Hz spacing is larger than the 64 Hz Nyquist frequency
        with self.assertRaises(ValueError):
            pyrf.wavelet(self._sine(), linear=linear)

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
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            True,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 103, tensor_order=1),
            True,
        ),
        (
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=1),
            False,
        ),
    )
    @unpack
    def test_vht_output(self, e, b, no_ez):
        result = pyrf.vht(e, b, no_ez)

        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], xr.DataArray)
        self.assertIsInstance(result[2], np.ndarray)


class NormalizeTestCase(unittest.TestCase):
    def test_normalize_input_type(self):
        with self.assertRaises(TypeError):
            pyrf.normalize(np.random.random((100, 3)))

    def test_normalize_input_shape(self):
        with self.assertRaises(ValueError):
            pyrf.normalize(generate_ts(64.0, 100, tensor_order=0))

    def test_normalize_output(self):
        result = pyrf.normalize(generate_ts(64.0, 100, tensor_order=1))
        self.assertIsInstance(result, xr.DataArray)


@ddt
class MeanFieldTestCase(unittest.TestCase):
    @data((generate_data(100, tensor_order=1), random.randint(0, 5)))
    @unpack
    def test_mean_field_input_type(self, inp, deg):
        with self.assertRaises(TypeError):
            pyrf.mean_field(inp, deg)

    @data((generate_ts(64.0, 100, tensor_order=1), random.randint(0, 5)))
    @unpack
    def test_mean_field_output(self, inp, deg):
        result = pyrf.mean_field(inp, deg)
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], xr.DataArray)


@ddt
class MedfiltTestCase(unittest.TestCase):
    @data(
        (generate_data(100, tensor_order=1), None),
    )
    @unpack
    def test_medfilt_input_type(self, inp, kernel_size):
        with self.assertRaises(TypeError):
            pyrf.medfilt(inp, kernel_size)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=random.randint(3, 10)),
            random.randint(0, 99),
        )
    )
    @unpack
    def test_medfilt_input_value(self, inp, kernel_size):
        with self.assertRaises(ValueError):
            pyrf.medfilt(inp, kernel_size)

    @data(
        (generate_ts(64.0, 100, tensor_order=0), None),
        (generate_ts(64.0, 100, tensor_order=0), random.randint(0, 99)),
        (generate_ts(64.0, 100, tensor_order=1), random.randint(0, 99)),
        (generate_ts(64.0, 100, tensor_order=2), random.randint(0, 99)),
    )
    @unpack
    def test_medfilt_output(self, inp, kernel_size):
        result = pyrf.medfilt(inp, kernel_size)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class MovmeanTestCase(unittest.TestCase):
    @data((generate_data(100, tensor_order=1), random.randint(2, 99)))
    @unpack
    def test_movmean_input_type(self, inp, window_size):
        with self.assertRaises(TypeError):
            pyrf.movmean(inp, window_size)

    @data(
        (generate_ts(64.0, 100, tensor_order=1), random.randint(0, 1)),
        (generate_ts(64.0, 100, tensor_order=1), random.randint(101, 666)),
    )
    @unpack
    def test_movmean_input_value(self, inp, window_size):
        with self.assertRaises(ValueError):
            pyrf.movmean(inp, window_size)

    @data(
        (generate_ts(64.0, 100, tensor_order=0), None),
        (generate_ts(64.0, 100, tensor_order=0), random.randint(2, 100)),
        (generate_ts(64.0, 100, tensor_order=1), random.randint(2, 100)),
        (generate_ts(64.0, 100, tensor_order=2), random.randint(2, 100)),
        (generate_ts(64.0, 100, tensor_order=3), random.randint(2, 100)),
    )
    @unpack
    def test_movmean_output(self, inp, window_size):
        result = pyrf.movmean(inp, window_size)
        self.assertIsInstance(result, xr.DataArray)


class EbspPhysicsTestCase(unittest.TestCase):
    """Value-level checks of ebsp against a synthetic circularly polarised
    wave with known propagation and Poynting directions (theta=40, phi=30)."""

    @staticmethod
    def _wave(f_s=64.0, duration=128.0, f_0=2.0, noise_n=0.0):
        rng = np.random.default_rng(0)
        t = np.arange(int(f_s * duration)) / f_s
        th, ph = np.deg2rad(40.0), np.deg2rad(30.0)
        n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        w = 2 * np.pi * f_0
        b = np.outer(np.cos(w * t), e1) + np.outer(np.sin(w * t), e2)
        b += 0.01 * rng.standard_normal(b.shape)
        b += noise_n * np.outer(rng.standard_normal(len(t)), n)
        e = 10 * np.cross(b, n)
        time = np.datetime64("2020-01-01T00:00:00", "ns") + (t * 1e9).astype(
            "timedelta64[ns]"
        )
        b0 = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 50.0], (len(t), 1)))
        return pyrf.ts_vec_xyz(time, e), pyrf.ts_vec_xyz(time, b), b0

    def test_ebsp_circular_wave(self):
        e, b, b0 = self._wave()
        res = pyrf.ebsp(e, b, b0, b0, None, [0.5, 8], polarization=True, fac=False)
        sel = dict(frequency=2.0, method="nearest")

        def med(x):
            return float(np.nanmedian(x.sel(**sel).data, axis=0).ravel()[0])

        self.assertAlmostEqual(med(res["k_tp"][..., 0]), 40.0, delta=0.5)
        self.assertAlmostEqual(med(res["k_tp"][..., 1]), 30.0, delta=0.5)
        self.assertAlmostEqual(med(res["pf_rtp"][..., 1]), 40.0, delta=0.5)
        self.assertAlmostEqual(med(res["pf_rtp"][..., 2]), 30.0, delta=0.5)
        self.assertAlmostEqual(med(res["ellipticity"]), 1.0, delta=0.01)
        self.assertAlmostEqual(med(res["planarity"]), 1.0, delta=0.01)
        dt = np.diff(res["t"]).astype(np.int64) / 1e9
        np.testing.assert_allclose(dt, 1 / (8 / 5), rtol=1e-6)
        # inputs must not be modified
        self.assertFalse(np.isnan(b.data).any())

    def test_ebsp_dop2d_in_polarisation_plane(self):
        e, b, b0 = self._wave(noise_n=4.0)  # compressional noise along k only
        res = pyrf.ebsp(e, b, b0, b0, None, [0.5, 8], polarization=True, fac=False)
        sel = dict(frequency=2.0, method="nearest")
        dop = float(np.nanmedian(res["dop"].sel(**sel)))
        dop2d = float(np.nanmedian(res["dop2d"].sel(**sel)))
        self.assertLess(dop, 0.9)
        self.assertGreater(dop2d, 0.95)

    def test_ebsp_peak_frequency_short_record(self):
        e, b, b0 = self._wave(duration=16.0, f_0=0.6)
        res = pyrf.ebsp(e, b, b0, b0, None, [0.3, 8], fac=False)
        spec = np.nanmean(res["bb_xxyyzzss"][..., 3].data, axis=0)
        f = res["f"]
        # 0.6 Hz lies between the 0.536 and 0.650 Hz bins; with a correct FFT
        # frequency vector their power ratio is ~1.07 (MATLAB), not ~0.38
        i_lo, i_hi = np.argsort(np.abs(f - 0.6))[:2][
            np.argsort(f[np.argsort(np.abs(f - 0.6))[:2]])
        ]
        self.assertTrue(0.9 < spec[i_hi] / spec[i_lo] < 1.25)

    def test_ebsp_e_gap_no_fac(self):
        e, b, b0 = self._wave()
        e.data[5000:5100] = np.nan
        res = pyrf.ebsp(e, b, b0, b0, None, [0.5, 8], polarization=True, fac=False)
        self.assertLess(float(np.isnan(res["pf_xyz"].data).mean()), 0.2)


if __name__ == "__main__":
    unittest.main()
