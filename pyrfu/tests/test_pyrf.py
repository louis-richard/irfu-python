#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import builtins
import datetime
import itertools
import math
import os
import random
import unittest
import warnings
from unittest import mock

# 3rd party imports
import numba
import numpy as np
import xarray as xr
from ddt import data, ddt, idata, unpack
from scipy import constants

# Local imports
from .. import pyrf
from ..constants import R_E
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
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


@ddt
class AnisotropyThresholdsTestCase(unittest.TestCase):
    # Ion coefficients for gamma = 0.01 (a, b, beta0)
    COEFFS_I = {
        "proton cyclotron": (0.649, 0.400, 0.0),
        "mirror mode": (1.040, 0.633, -0.012),
        "parallel firehose": (-0.647, 0.583, 0.713),
        "oblique firehose": (-1.447, 1.000, -0.148),
    }

    def test_anisotropy_thresholds_ions(self):
        # Each threshold depends only on beta_para: the oblique firehose was NaN
        # below the parallel firehose beta0 (0.713), set to NaN in the shared
        # input by the parallel firehose computed before it.
        beta = np.array([0.1, 0.5, 1.0, 2.0])
        result = pyrf.anisotropy_thresholds(beta)

        self.assertListEqual(list(result), list(self.COEFFS_I))

        for name, (a, b, beta0) in self.COEFFS_I.items():
            expected = [1 + a / (x - beta0) ** b if x > beta0 else np.nan for x in beta]
            np.testing.assert_allclose(result[name], expected, rtol=1e-12)

        np.testing.assert_allclose(
            result["oblique firehose"], [-4.835, -1.233, -0.260, 0.326], atol=1e-3
        )

    def test_anisotropy_thresholds_electrons(self):
        # NaN where beta_para ** -alpha is undefined (beta_para <= 0)
        beta = np.array([-1.0, 0.0, 0.5, 2.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pyrf.anisotropy_thresholds(beta, specie="e", gamma=0.1)

        expected = {"firehose": (-1.32, 0.61), "whistler": (1.0, 0.49)}

        for name, (s_coeff, alpha) in expected.items():
            values = 1 + s_coeff * beta[2:] ** -alpha
            np.testing.assert_allclose(result[name], [np.nan, np.nan, *values])

    @data(np.array, xr.DataArray)
    def test_anisotropy_thresholds_input_unchanged(self, wrap):
        beta = np.array([-0.5, 0.1, 0.5, 1.0, 2.0])
        inp = wrap(beta.copy())
        pyrf.anisotropy_thresholds(inp)
        pyrf.anisotropy_thresholds(inp, specie="e")

        np.testing.assert_array_equal(np.asarray(inp), beta)

    @data(
        (1.0, float),
        (np.array([1, 2, 3]), np.ndarray),  # integers
        ([0.5, 1.0], np.ndarray),
    )
    @unpack
    def test_anisotropy_thresholds_input_types(self, inp, out_type):
        result = pyrf.anisotropy_thresholds(inp)
        expected = pyrf.anisotropy_thresholds(np.atleast_1d(inp).astype(float))

        for name, value in result.items():
            self.assertIsInstance(value, out_type)
            np.testing.assert_allclose(np.atleast_1d(value), expected[name])

    def test_anisotropy_thresholds_dataarray(self):
        # Time series in, time series out (with the same coordinates)
        beta = generate_ts(64.0, 100, tensor_order=0)
        beta.data = np.abs(beta.data) + 0.1
        result = pyrf.anisotropy_thresholds(beta)

        for name, value in result.items():
            self.assertIsInstance(value, xr.DataArray)
            self.assertEqual(value.name, name)
            np.testing.assert_array_equal(value.time.data, beta.time.data)
            np.testing.assert_allclose(
                value.data, pyrf.anisotropy_thresholds(beta.data)[name]
            )

    @data(("p", 0.01), ("i", 0.5), ("e", 0.001))
    @unpack
    def test_anisotropy_thresholds_input_values(self, specie, gamma):
        with self.assertRaises(ValueError):
            pyrf.anisotropy_thresholds(1.0, specie=specie, gamma=gamma)


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

    def test_c_4_j_linear_field(self):
        # B = (c z, 0, b_0 + a x) nT with x, z in km is reproduced exactly
        a, b_0, c = 1.0, 20.0, 0.5
        r_sc = 10.0 * np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
            ]
        )
        time = generate_timeline(1.0, 3)
        r_mms = [pyrf.ts_vec_xyz(time, np.tile(r, (3, 1))) for r in r_sc]
        b_mms = [
            pyrf.ts_vec_xyz(time, np.tile([c * r[2], 0.0, b_0 + a * r[0]], (3, 1)))
            for r in r_sc
        ]
        j, div_b, b_avg, jxb, div_t_shear, div_pb = pyrf.c_4_j(r_mms, b_mms)

        # B at the center of the tetrahedron [nT]
        b_x, b_z = c * np.mean(r_sc[:, 2]), b_0 + a * np.mean(r_sc[:, 0])
        # nT/km -> T/m and nT^2/km -> T^2/m
        j_y = (c - a) * 1e-12 / constants.mu_0

        np.testing.assert_allclose(b_avg.data, [[b_x, 0.0, b_z]] * 3, rtol=1e-12)
        np.testing.assert_allclose(j.data, [[0.0, j_y, 0.0]] * 3, rtol=1e-9)
        np.testing.assert_allclose(div_b.data, 0.0, atol=1e-18)
        np.testing.assert_allclose(
            jxb.data, [[j_y * b_z * 1e-9, 0.0, -j_y * b_x * 1e-9]] * 3, rtol=1e-9
        )
        np.testing.assert_allclose(
            div_t_shear.data,
            [[b_z * c * 1e-21 / constants.mu_0, 0.0, b_x * a * 1e-21 / constants.mu_0]]
            * 3,
            rtol=1e-9,
            atol=1e-30,
        )
        # grad(B^2 / 2) / mu_0 = (b_z a, 0, b_x c) and J x B = div T - grad Pb
        np.testing.assert_allclose(
            div_pb.data,
            [[b_z * a * 1e-21 / constants.mu_0, 0.0, b_x * c * 1e-21 / constants.mu_0]]
            * 3,
            rtol=1e-9,
            atol=1e-30,
        )


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

    def test_calc_ag_unequal_perp(self):
        # det P = 0.472 and det G = 1 * ((0.8 + 0.6) / 2) ** 2 = 0.49
        p_fac = np.array([[1.0, 0.1, 0.05], [0.1, 0.8, 0.0], [0.05, 0.0, 0.6]])
        result = pyrf.calc_ag(
            pyrf.ts_tensor_xyz(generate_timeline(1.0, 1), p_fac[None])
        )
        self.assertAlmostEqual(result.data[0], 0.018 / 0.962, places=12)

    def test_calc_ag_perp_rotation_invariant(self):
        # Rotating the perpendicular axes about B (first axis) must not change AG
        p_fac = np.array([[1.0, 0.1, 0.05], [0.1, 0.8, 0.0], [0.05, 0.0, 0.6]])
        angles = np.linspace(0.0, np.pi, 7)
        rot = np.zeros((len(angles), 3, 3))
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1], rot[:, 1, 2] = np.cos(angles), -np.sin(angles)
        rot[:, 2, 1], rot[:, 2, 2] = np.sin(angles), np.cos(angles)
        p_rot = np.einsum("nij,jk,nlk->nil", rot, p_fac, rot)
        result = pyrf.calc_ag(
            pyrf.ts_tensor_xyz(generate_timeline(1.0, len(angles)), p_rot)
        )
        np.testing.assert_allclose(result.data, 0.018 / 0.962, rtol=1e-12)

    def test_calc_ag_nan(self):
        p_fac = np.tile(np.eye(3), (3, 1, 1))
        p_fac[1, 0, 1] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pyrf.calc_ag(pyrf.ts_tensor_xyz(generate_timeline(1.0, 3), p_fac))
        np.testing.assert_array_equal(result.data, [0.0, np.nan, 0.0])


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

    @data("datetime64[ns]", "datetime64[us]", "datetime64[ms]")
    def test_calc_fs_time_unit(self, dtype):
        # The time was assumed in ns, so us times gave 1000 times f_s
        time = generate_timeline(100.0, 50).astype(dtype)
        inp = xr.DataArray(np.zeros(len(time)), coords=[time], dims=["time"])
        self.assertAlmostEqual(pyrf.calc_fs(inp), 100.0, places=9)
        self.assertAlmostEqual(pyrf.calc_fs(inp.to_dataset(name="x")), 100.0)


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
        n_c = random.randint(2, 100)
        result = _compress_cwt_1d.__wrapped__(
            np.random.random((1000, 100)),
            np.arange(1000 // n_c) * n_c,
            n_c,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1000 // n_c, 100))

    @staticmethod
    def _cwt(data):
        times = generate_timeline(1.0, len(data))
        coords = {"time": times, "f": np.arange(data.shape[1], dtype=float)}
        return xr.Dataset({c: (["time", "f"], data.copy()) for c in "xyz"}, coords)

    def test_compress_cwt_nan(self):
        data = np.ones((100, 2))
        data[3, 0] = np.nan
        data[10:20, 1] = np.nan
        _, cwt_x, _, _ = pyrf.compress_cwt(self._cwt(data), 10)

        # NaNs are ignored in the averages; only all-NaN blocks are NaN
        self.assertEqual(cwt_x[0, 0], 1.0)
        self.assertTrue(np.isnan(cwt_x[1, 1]))
        self.assertEqual(np.sum(np.isnan(cwt_x)), 1)

    @data(5, 10, 7)
    def test_compress_cwt_blocks(self, n_c):
        data = np.tile(np.arange(100, dtype=float)[:, None], (1, 2))
        cwt = self._cwt(data)
        cwt_t, cwt_x, _, _ = pyrf.compress_cwt(cwt, n_c)

        # All the full blocks of nc points, stamped at their centre
        n_b = 100 // n_c
        expected = np.arange(n_b) * n_c + (n_c - 1) / 2
        self.assertEqual(cwt_x.shape, (n_b, 2))
        np.testing.assert_array_almost_equal(cwt_x[:, 0], expected)

        d_t = (cwt_t - cwt.time.data[0]) / np.timedelta64(1, "s")
        np.testing.assert_array_almost_equal(d_t, expected)


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

    def test_convert_fac_values(self):
        time = generate_timeline(100.0, 10)
        k = np.arange(10.0)
        b_xyz = np.stack([0 * k + 1.0, 0.1 * k, 0 * k + 2.0], axis=1)
        b_bgd = pyrf.ts_vec_xyz(time, b_xyz)

        # Integer input was truncated
        inp = pyrf.ts_vec_xyz(time, np.tile([1, 2, 3], (10, 1)))
        result = pyrf.convert_fac(inp, b_bgd)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        expected = pyrf.convert_fac(inp.astype(np.float64), b_bgd)
        np.testing.assert_allclose(result.data, expected.data)
        b_hat = b_xyz / np.linalg.norm(b_xyz, axis=1, keepdims=True)
        np.testing.assert_allclose(result.data[:, 2], b_hat @ [1.0, 2.0, 3.0])

        # Same length on grids offset by 5 ms: the samples were paired by index
        b_off = pyrf.ts_vec_xyz(
            time + np.timedelta64(5, "ms"),
            np.stack([0 * k + 1.0, 0.1 * (k + 0.5), 0 * k + 2.0], axis=1),
        )
        result = pyrf.convert_fac(inp, b_off)
        np.testing.assert_allclose(result.data, expected.data, atol=1e-9)


@ddt
class CorrDerivTestCase(unittest.TestCase):
    @data(False, True)
    def test_corr_deriv_values(self, flag):
        # 1 Hz sines, the second 30 ms later, with no sample on a zero:
        # extrema at 0.25 + k / 2 s, inflections and zero crossings at k / 2 s
        t_sec = 0.0037 + np.arange(300) / 100
        t_0 = np.datetime64("2019-01-01T00:00:00", "ns")
        time = t_0 + (t_sec * 1e9).astype("timedelta64[ns]")
        inp0 = pyrf.ts_scalar(time, np.sin(2 * np.pi * t_sec))
        inp1 = pyrf.ts_scalar(time, np.sin(2 * np.pi * (t_sec - 0.03)))

        t1_d, t2_d, t1_dd, t2_dd = pyrf.corr_deriv(inp0, inp1, flag)

        def seconds(times):
            return (times - t_0) / np.timedelta64(1, "s")

        extrema = 0.25 + np.arange(6) / 2
        zeros = 0.5 + np.arange(5) / 2
        np.testing.assert_allclose(seconds(t1_d), extrema, atol=1e-3)
        np.testing.assert_allclose(seconds(t2_d), extrema + 0.03, atol=1e-3)
        np.testing.assert_allclose(seconds(t1_dd), zeros, atol=1e-3)
        np.testing.assert_allclose(seconds(t2_dd), zeros + 0.03, atol=1e-3)
        self.assertEqual(t1_d.dtype, np.dtype("datetime64[ns]"))


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

    def test_cotrans_flag_case(self):
        time = generate_timeline(1.0, 10)
        data_ = np.tile([1.0, 2.0, 3.0], (10, 1))
        inp = pyrf.ts_vec_xyz(time, data_)
        expected = pyrf.cotrans(inp, "gse>gsm")

        # Upper-case flags raised a KeyError
        result = pyrf.cotrans(inp, "GSE>GSM")
        np.testing.assert_allclose(result.data, expected.data)
        self.assertEqual(result.attrs["COORDINATE_SYSTEM"], "GSM")

        # ... or an AssertionError if inp has a COORDINATE_SYSTEM attribute
        inp = pyrf.ts_vec_xyz(time, data_, attrs={"COORDINATE_SYSTEM": "GSE"})
        np.testing.assert_allclose(pyrf.cotrans(inp, "GSE>GSM").data, expected.data)
        np.testing.assert_allclose(pyrf.cotrans(inp, "Gsm").data, expected.data)

        # The dipole direction failed if inp has a COORDINATE_SYSTEM attribute
        expected = pyrf.cotrans(pyrf.ts_vec_xyz(time, data_), "dipoledirectiongse")
        result = pyrf.cotrans(inp, "DipoleDirectionGSE")
        np.testing.assert_allclose(result.data, expected.data)

    def test_cotrans_errors(self):
        inp = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 10),
            np.tile([1.0, 2.0, 3.0], (10, 1)),
            attrs={"COORDINATE_SYSTEM": "gse"},
        )

        # Input frame in flag and in the attributes differ (was an assert)
        with self.assertRaises(ValueError):
            pyrf.cotrans(inp, "gsm>sm")

        # Unknown transformation (was a KeyError)
        with self.assertRaises(ValueError):
            pyrf.cotrans(inp, "gse>lmn")

        # No input frame
        with self.assertRaises(ValueError):
            pyrf.cotrans(pyrf.ts_vec_xyz(inp.time.data, inp.data), "gsm")

    @data(True, False)
    def test_cotrans_time_unit(self, hapgood):
        # The time was assumed in ns, so a time coordinate in us rotated by the
        # angles of 1970
        time = generate_timeline(1.0, 10)
        data_ = np.tile([1.0, 2.0, 3.0], (10, 1))
        expected = pyrf.cotrans(pyrf.ts_vec_xyz(time, data_), "gei>gsm", hapgood)

        inp = xr.DataArray(
            data_,
            coords=[time.astype("datetime64[us]"), ["x", "y", "z"]],
            dims=["time", "comp"],
        )
        result = pyrf.cotrans(inp, "gei>gsm", hapgood)
        np.testing.assert_allclose(result.data, expected.data, rtol=1e-12)

    @data(True, False)
    def test_cotrans_sidereal_time(self, hapgood):
        # GEI to GEO is a rotation by the Greenwich mean sidereal time (IAU 1982)
        time = np.array(
            [
                "2000-01-01T12:00:00",
                "2008-06-01T03:00:00",
                "2019-09-14T07:54:00",
                "2026-10-02T23:30:00",
            ],
            dtype="datetime64[ns]",
        )
        du = (time - np.datetime64("2000-01-01T12:00:00", "ns")) / np.timedelta64(
            1, "D"
        )
        du0 = np.floor(du - 0.5) + 0.5
        tu = du0 / 36525
        gmst = 24110.54841 + 8640184.812866 * tu + 0.093104 * tu**2 - 6.2e-6 * tu**3
        gmst = (gmst + 1.002737909350795 * (du - du0) * 86400) / 240

        inp = pyrf.ts_vec_xyz(time, np.tile([1.0, 0.0, 0.0], (len(time), 1)))
        result = pyrf.cotrans(inp, "gei>geo", hapgood=hapgood)
        theta = np.rad2deg(np.arctan2(-result.data[:, 1], result.data[:, 0]))

        # USNO formula was given TT instead of UT, so GEO was off by 0.29 deg
        # with hapgood=False
        np.testing.assert_allclose((theta - gmst + 180) % 360 - 180, 0, atol=1e-3)
        self.assertAlmostEqual(theta[0] % 360, 280.46061837, delta=1e-3)


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

    def test_cross_time_alignment(self):
        # Same length on grids offset by 5 ms: the samples were paired by index
        time = generate_timeline(100.0, 10)
        k = np.arange(10.0)
        e_xyz = pyrf.ts_vec_xyz(time, np.tile([1.0, 0.0, 0.0], (10, 1)))
        b_xyz = pyrf.ts_vec_xyz(
            time + np.timedelta64(5, "ms"), np.stack([0 * k, 0 * k, k], axis=1)
        )

        result = pyrf.cross(e_xyz, b_xyz)
        np.testing.assert_array_equal(result.time.data, time)
        np.testing.assert_allclose(result.data[:, 1], -(k - 0.5), atol=1e-9)

        # Same times: no resampling
        result = pyrf.cross(e_xyz, pyrf.ts_vec_xyz(time, b_xyz.data))
        np.testing.assert_array_equal(result.data[:, 1], -k)

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


class Unix2Datetime64TestCase(unittest.TestCase):
    def test_unix2datetime64_input(self):
        with self.assertRaises(TypeError):
            pyrf.unix2datetime64(1.0)

    def test_unix2datetime64_values(self):
        # Rounded to the nearest ns: 65 us * 1e9 = 64999.99999999999 used to be
        # truncated to 64999 ns
        result = pyrf.unix2datetime64([6.5e-5, 1.29e-4, 1.5])
        expected = np.array([65000, 129000, 1500000000], dtype="datetime64[ns]")
        np.testing.assert_array_equal(result, expected)

        # Round trip at the float64 resolution of Unix seconds (256 ns in 2026)
        time = np.datetime64("2026-09-29T12:34:56.123456", "ns") + np.arange(10)
        round_trip = pyrf.unix2datetime64(pyrf.datetime642unix(time))
        np.testing.assert_array_less(np.abs((round_trip - time).astype(np.int64)), 129)


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

    def test_dist_append_attrs_unchanged(self):
        # The stacked delta energies were written into the first VDF's attrs
        inp0 = generate_vdf(64.0, 10, [32, 32, 16])
        inp1 = generate_vdf(64.0, 10, [32, 32, 16])
        attrs = {k: np.array(v, copy=True) for k, v in inp0.attrs.items()}

        result = pyrf.dist_append(inp0, inp1)

        self.assertListEqual(list(inp0.attrs), list(attrs))
        for key, value in attrs.items():
            np.testing.assert_array_equal(inp0.attrs[key], value)
        self.assertTupleEqual(result.attrs["delta_energy_plus"].shape, (20, 32))


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

    def test_dynamic_press_values(self):
        # 5 cm^-3 at 400 km/s: 5e6 * m_p * (4e5) ** 2 = 1.338 nPa
        time = generate_timeline(1.0, 4)
        n_s = pyrf.ts_scalar(time, np.full(4, 5.0))
        v_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, -400.0, 0.0], (4, 1)))

        result = pyrf.dynamic_press(n_s, v_xyz, "ions")
        p_ion = 5e6 * constants.m_p * 4e5**2 * 1e9
        np.testing.assert_allclose(result.data, p_ion, rtol=1e-12)
        self.assertAlmostEqual(p_ion, 1.338, places=3)
        self.assertEqual(result.attrs["UNITS"], "nPa")

        result = pyrf.dynamic_press(n_s, v_xyz, "electrons")
        p_ele = p_ion * constants.m_e / constants.m_p
        np.testing.assert_allclose(result.data, p_ele, rtol=1e-12)

    def test_dynamic_press_time_alignment(self):
        # V_x = 400 + 10 t km/s sampled half a second later than n
        time = generate_timeline(1.0, 6)
        n_s = pyrf.ts_scalar(time, np.full(6, 2.0))
        t_v = np.arange(6) + 0.5
        v_xyz = pyrf.ts_vec_xyz(
            time + np.timedelta64(500, "ms"),
            np.column_stack([400.0 + 10 * t_v, np.zeros(6), np.zeros(6)]),
        )

        result = pyrf.dynamic_press(n_s, v_xyz)
        v_x = 400.0 + 10 * np.arange(6)
        expected = 2e6 * constants.m_p * (1e3 * v_x) ** 2 * 1e9
        np.testing.assert_allclose(result.data, expected, rtol=1e-9)
        np.testing.assert_array_equal(result.time.data, time)


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

    def test_eb_nrf_values(self):
        time = generate_timeline(1.0, 4)
        e_xyz = pyrf.ts_vec_xyz(time, np.tile([1.0, 2.0, 3.0], (4, 1)))
        v_xyz = pyrf.ts_vec_xyz(time, np.tile([1.0, 0.0, 0.0], (4, 1)))

        # "a": L along B = y, N closest to v = x, M = N x L = z
        b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 2.0, 0.0], (4, 1)))
        for flag in ["a", "A"]:
            result = pyrf.eb_nrf(e_xyz, b_xyz, v_xyz, flag)
            np.testing.assert_allclose(result.data, [[2.0, 3.0, 1.0]] * 4, atol=1e-12)

        # Default flag is "a", constant normal vector
        result = pyrf.eb_nrf(e_xyz, b_xyz, [2.0, 0.0, 0.0])
        np.testing.assert_allclose(result.data, [[2.0, 3.0, 1.0]] * 4, atol=1e-12)

        # "b": N along v, L along the mean B (y) perpendicular to N
        b_xyz = pyrf.ts_vec_xyz(
            time, np.column_stack([np.zeros(4), np.ones(4), [0.5, -0.5, 0.5, -0.5]])
        )
        result = pyrf.eb_nrf(e_xyz, b_xyz, v_xyz, "b")
        np.testing.assert_allclose(result.data, [[2.0, 3.0, 1.0]] * 4, atol=1e-12)

        # L closest to (0, 1, 1): M = (0, -1, 1) / sqrt(2), L = (0, 1, 1) / sqrt(2)
        result = pyrf.eb_nrf(e_xyz, b_xyz, v_xyz, np.array([0.0, 1.0, 1.0]))
        h = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(result.data, [[5 * h, h, 1.0]] * 4, atol=1e-12)

    def test_eb_nrf_v_resampled(self):
        # v on a coarser grid is resampled to the times of e
        e_xyz = pyrf.ts_vec_xyz(
            generate_timeline(4.0, 16), np.tile([1.0, 2.0, 3.0], (16, 1))
        )
        b_xyz = pyrf.ts_vec_xyz(
            generate_timeline(4.0, 16), np.tile([0.0, 1.0, 0.0], (16, 1))
        )
        v_xyz = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 4), np.tile([1.0, 0.0, 0.0], (4, 1))
        )
        result = pyrf.eb_nrf(e_xyz, b_xyz, v_xyz, "a")
        np.testing.assert_allclose(result.data, [[2.0, 3.0, 1.0]] * 16, atol=1e-12)
        np.testing.assert_array_equal(result.time.data, e_xyz.time.data)


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

    @data("e.b=0", "e_perp+nan", "e_par")
    def test_edb_input_unchanged(self, value):
        # Ez is recomputed in a copy: the caller's E (data and attrs) is unchanged
        e_xyz = generate_ts(64.0, 100, tensor_order=1)
        e_data, e_attrs = e_xyz.data.copy(), dict(e_xyz.attrs)
        b_bgd = generate_ts(64.0, 100, tensor_order=1)

        e_out, _ = pyrf.edb(e_xyz, b_bgd, 0.0 if value == "e.b=0" else 90.0, value)

        np.testing.assert_array_equal(e_xyz.data, e_data)
        self.assertDictEqual(e_xyz.attrs, e_attrs)
        self.assertFalse(np.array_equal(e_out.data[:, 2], e_data[:, 2], equal_nan=True))


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

    @data(
        # half length / radius, C / (4 pi eps0 radius) from a boundary element
        # solution (converged to 5 digits), relative tolerance
        (0.51, 0.9689, 0.03),
        (1.0, 1.1915, 0.03),
        (2.0, 1.5730, 0.03),
        (3.99, 2.2137, 0.03),
        (4.0, 2.2166, 0.08),
        (6.0, 2.7857, 0.03),
        (10.0, 3.8128, 0.03),
        (50.0, 11.8727, 0.03),
    )
    @unpack
    def test_estimate_capacitance_cylinder(self, h_a, c_bem, rtol):
        radius = 0.2
        result = pyrf.estimate("capacitance_cylinder", radius, h_a * radius)
        c_ref = 4 * np.pi * constants.epsilon_0 * radius * c_bem
        np.testing.assert_allclose(result, c_ref, rtol=rtol)

        # Bounded below by the inscribed disk and sphere
        self.assertGreater(result, pyrf.estimate("capacitance_disk", radius))
        if h_a >= 1:
            self.assertGreater(result, pyrf.estimate("capacitance_sphere", radius))


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


class FindClosestTestCase(unittest.TestCase):
    def test_find_closest_drop_t1(self):
        t_1, t_2, ind_1, ind_2 = pyrf.find_closest(
            np.arange(10.0), np.arange(0.0, 10.0, 3.0) + 0.1
        )
        np.testing.assert_array_equal(t_1, [0.0, 3.0, 6.0, 9.0])
        np.testing.assert_array_equal(t_2, [0.1, 3.1, 6.1, 9.1])
        np.testing.assert_array_equal(ind_1, [0, 3, 6, 9])
        np.testing.assert_array_equal(ind_2, [0, 1, 2, 3])
        self.assertTrue(np.issubdtype(ind_1.dtype, np.integer))

    def test_find_closest_drop_t2(self):
        # Every t1 is the nearest of some t2, but t2 = 1 is nobody's nearest
        t_1, t_2, ind_1, ind_2 = pyrf.find_closest(
            np.array([0.0, 10.0]), np.array([0.2, 1.0, 9.5])
        )
        np.testing.assert_array_equal(t_1, [0.0, 10.0])
        np.testing.assert_array_equal(t_2, [0.2, 9.5])
        np.testing.assert_array_equal(ind_1, [0, 1])
        np.testing.assert_array_equal(ind_2, [0, 2])

    def test_find_closest_datetime64(self):
        t_1 = np.datetime64("2020-01-01", "ns") + np.arange(5) * np.timedelta64(1, "s")
        t_2 = t_1[[1, 3]] + np.timedelta64(100, "ms")
        t_1_new, t_2_new, ind_1, ind_2 = pyrf.find_closest(t_1, t_2)
        np.testing.assert_array_equal(t_1_new, t_1[[1, 3]])
        np.testing.assert_array_equal(t_2_new, t_2)
        np.testing.assert_array_equal(ind_1, [1, 3])
        np.testing.assert_array_equal(ind_2, [0, 1])

    def test_find_closest_empty(self):
        t_1, t_2, ind_1, ind_2 = pyrf.find_closest(np.arange(3.0), np.array([]))
        for result in [t_1, t_2, ind_1, ind_2]:
            self.assertEqual(len(result), 0)


def _omni_response(header, rows, n_vars):
    # OMNIWeb listing (as returned by nx1.cgi), mocked for urllib.request.urlopen
    params = "".join(f" {i + 1} variable {i + 1}\n" for i in range(n_vars))
    lines = "\n".join(rows)
    text = (
        "<B>Listing for omni data</B><hr><pre>Selected parameters:\n"
        f"{params}\n{header}\n{lines}\n</pre><hr><HR>"
    )
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = text.encode()
    return response


@ddt
class GetOmniDataTestCase(unittest.TestCase):
    # 2019-09-14 (day 257) and 2019-09-15 (day 258), hourly: b, v, ae; with fill
    # values for b (999.9) and v (9999.) at 05:00, and a valid AE of 999 nT
    HOURS = [
        f"2019 {257 + h // 24} {h % 24:2d}"
        + (
            "  999.9  9999.   999"
            if h == 5
            else f"   {3 + h / 10:.1f}  {500 - h}.    {h}"
        )
        for h in range(48)
    ]

    def _get(self, rows, variables, tint, database="omni_hour", n_vars=None):
        header = "YYYY DOY HR MN" if database == "omni_min" else "YEAR DOY HR"
        n_vars = len(variables) if n_vars is None else n_vars
        response = _omni_response(f"{header} 1 2 3", rows, n_vars)

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            out = pyrf.get_omni_data(variables, tint, database=database)

        return out, urlopen.call_args.args[0]

    @data(
        # Hours overlapping tint (each one averages [t, t + 1 h))
        (["2019-09-14T07:30:00", "2019-09-14T10:30:00"], [7, 8, 9, 10]),
        (["2019-09-14T00:00:00", "2019-09-15T00:00:00"], list(range(25))),
        (["2019-09-14T07:54:00", "2019-09-14T08:11:00"], [7, 8]),
    )
    @unpack
    def test_get_omni_data_hourly_times(self, tint, hours):
        tint_in = list(tint)
        out, url = self._get(self.HOURS, ["b", "v", "ae"], tint)

        expected = np.datetime64("2019-09-14T00:00", "ns") + np.array(
            hours, dtype="timedelta64[h]"
        )
        np.testing.assert_array_equal(out.time.data, expected)
        self.assertListEqual(tint, tint_in)  # not modified
        self.assertIn("spacecraft=omni2&start_date=20190914&end_date=2019091", url)
        self.assertTrue(url.endswith("&vars=8&vars=24&vars=41"))

    def test_get_omni_data_fill_values(self):
        tint = ["2019-09-14T00:00:00", "2019-09-14T23:00:00"]
        out, _ = self._get(self.HOURS, ["b", "v", "ae"], tint)

        self.assertTrue(np.isnan(out.b.data[5]) and np.isnan(out.v.data[5]))
        self.assertEqual(out.ae.data[5], 999.0)  # AE fill value is 9999
        self.assertEqual(int(np.isnan(out.to_array()).sum()), 2)
        self.assertAlmostEqual(float(out.b.data[4]), 3.4)

    def test_get_omni_data_minute(self):
        # 1-minute data: minute codes, "YYYY DOY HR MN" header, minute times
        rows = [
            f"2019 257  {7 + m // 60} {m % 60:2d}   -1.{m:02d}   1.5"
            for m in range(120)
        ]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        out, url = self._get(rows, ["bzgsm", "bx", "bxgse"], tint, "omni_min", n_vars=2)

        self.assertEqual(out.sizes["time"], 18)
        self.assertEqual(out.time.data[0], np.datetime64("2019-09-14T07:54", "ns"))
        self.assertIn(
            "spacecraft=omni_min&start_date=2019091407&end_date=2019091408", url
        )
        self.assertTrue(url.endswith("&vars=18&vars=14"))  # bx and bxgse: same code
        np.testing.assert_array_equal(out.bx.data, out.bxgse.data)
        self.assertAlmostEqual(float(out.bzgsm.data[0]), -1.54)

    @data(
        (["b"], "omni_sec"),  # database
        (["vx"], "omni_hour"),  # not available hourly
        (["dst"], "omni_min"),  # not available in 1-minute data
        (["bmag"], "omni_hour"),  # unknown
    )
    @unpack
    def test_get_omni_data_input(self, variables, database):
        with self.assertRaises(ValueError):
            pyrf.get_omni_data(variables, ["2019-09-14", "2019-09-15"], database)

    def test_get_omni_data_no_data(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"<html>Error: no data"

        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(ValueError):
                pyrf.get_omni_data(["b"], ["2019-09-14", "2019-09-15"])

    @unittest.skipUnless(
        os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
    )
    def test_get_omni_data_omniweb(self):
        out = pyrf.get_omni_data(
            ["b", "bzgsm", "v"], ["2019-09-14T00:00:00", "2019-09-15T00:00:00"]
        )
        self.assertEqual(out.sizes["time"], 25)
        np.testing.assert_allclose(out.to_array().data[:, 0], [3.4, 1.6, 506.0])


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

    def test_increments_values(self):
        # Increments 1, -1, 1, -1, 2, -2: m2 = 2, m4 = 6, Pearson kurtosis 1.5
        time = generate_timeline(1.0, 8)
        inp = pyrf.ts_scalar(time, np.array([0, 1, 0, 1, 0, 2, 0, np.nan]))
        kurt, result = pyrf.increments(inp, 1)
        np.testing.assert_allclose(kurt, [1.5])
        np.testing.assert_array_equal(result.data, [1, -1, 1, -1, 2, -2, np.nan])
        np.testing.assert_array_equal(result.time.data, time[:7])

    def test_increments_vector(self):
        time = generate_timeline(1.0, 7)
        x_inp = np.array([0, 1, 0, 1, 0, 2, 0], dtype=float)
        inp = pyrf.ts_vec_xyz(time, np.column_stack([x_inp, 2 * x_inp, -x_inp]))
        kurt, result = pyrf.increments(inp, 1)
        np.testing.assert_allclose(kurt, [1.5, 1.5, 1.5])
        self.assertListEqual(list(result.shape), [6, 3])
        self.assertTupleEqual(result.dims, inp.dims)

    def test_increments_single(self):
        # One increment left: the time dimension must be kept
        inp = pyrf.ts_scalar(generate_timeline(1.0, 3), np.array([0.0, 1.0, 3.0]))
        _, result = pyrf.increments(inp, 2)
        np.testing.assert_array_equal(result.data, [3.0])
        self.assertTupleEqual(result.dims, ("time",))

    def test_increments_scale(self):
        inp = pyrf.ts_scalar(generate_timeline(1.0, 5), np.arange(5.0))
        for scale in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                pyrf.increments(inp, scale)


@ddt
class IntSphDistTestCase(unittest.TestCase):
    @data(
        {"projection_base": "pol", "projection_dim": "2d"},
        {"projection_base": "pol", "projection_dim": "3d"},
        {"projection_base": "cart", "projection_dim": "1d"},
        {"projection_base": "polar", "projection_dim": "1d"},
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
        {"velocity_edges": np.linspace(0.99, 2.01, 52)},  # brackets the speeds
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

    @data(
        ("pol", "1d", 20, None),
        ("pol", "1d", 20, "lin"),
        ("pol", "1d", 20, "log"),
        ("cart", "2d", 20, None),
        ("cart", "3d", 5, None),
    )
    @unpack
    def test_int_sph_dist_drifting_maxwellian(self, base, dim, n_mc, weight):
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
            weight=weight,
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

    def test_uniform_step(self):
        self.assertAlmostEqual(_uniform_step(np.linspace(-1.0, 1.0, 11)), 0.2)
        self.assertEqual(_uniform_step(np.array([1.0])), 0.0)  # no bin
        self.assertEqual(_uniform_step(np.array([0.0, 0.0, 1.0])), 0.0)  # zero step
        self.assertEqual(_uniform_step(np.array([0.0, 1.0, 3.0])), 0.0)  # non-uniform

    @staticmethod
    def _mc_kernel_args(kernel, speed, phi, theta, edges, v_lim, a_lim):
        # f = 1 in every instrument bin, with unit widths and a single
        # Monte-Carlo particle, which is at the bin centre (no random draws), so
        # that each bin carries dtau = speed ** 2 * cos(theta) to a known point.
        n_v, n_ph, n_th = len(speed), len(phi), len(theta)
        d_a_grid = np.ones(len(edges) - 1) if kernel is _mc_pol_1d else 1.0
        return (
            np.ones((n_v, n_ph, n_th)),
            np.array(speed, dtype=np.float64),
            np.array(phi, dtype=np.float64),
            np.array(theta, dtype=np.float64),
            np.ones(n_v),
            np.full(n_v, 0.5),
            np.ones(n_ph),
            np.ones(n_th),
            edges,
            d_a_grid,
            np.array(v_lim, dtype=np.float64),
            np.array(a_lim, dtype=np.float64),
            np.ones((n_v, n_ph, n_th), dtype=np.int64),
            np.eye(3),
        )

    def _mc_kernel_runs(self, kernel, *args, v_lim=(-np.inf, np.inf), a_lim=None):
        # Run the kernel as Python and jitted, and for the cartesian kernels with
        # both the arithmetic (fast) and the np.searchsorted bin index; with no
        # random draws, all must be identical.
        # (one accumulator row per thread, indexed by numba.get_thread_id())
        a_lim = (-np.pi, np.pi) if a_lim is None else a_lim
        k_args = self._mc_kernel_args(kernel, *args, v_lim=v_lim, a_lim=a_lim)
        n_threads = numba.get_num_threads()

        if kernel is _mc_pol_1d:
            funcs = [kernel.__wrapped__, kernel]
            results = [func(*k_args, n_threads) for func in funcs]
        else:
            v_step = _uniform_step(k_args[8])
            self.assertGreater(v_step, 0.0)
            results = [
                func(*k_args, step, n_threads)
                for func in [kernel.__wrapped__, kernel]
                for step in [v_step, 0.0]
            ]

        # Same bins; jitted values can differ by round-off, as the threads add
        # the particles landing in the same bin in a different order.
        for result in results[1:]:
            np.testing.assert_array_equal(result != 0, results[0] != 0)
            np.testing.assert_allclose(result, results[0], rtol=1e-12)

        if kernel is not _mc_pol_1d:
            np.testing.assert_array_equal(results[0], results[1])
            np.testing.assert_array_equal(results[2], results[3])

        return results[0]

    @data(
        # speed, phi, theta, bin index (None if outside the grid)
        (_mc_pol_1d, 0.05, 0.0, 0.0, None),
        (_mc_pol_1d, 0.1, 0.0, 0.0, (0,)),  # on the first edge
        (_mc_pol_1d, 0.25, 0.0, 0.0, (1,)),  # on an interior edge: upper bin
        (_mc_pol_1d, 0.3, 0.0, 0.0, (1,)),
        (_mc_pol_1d, 1.0, 0.0, 0.0, (2,)),  # on the last edge: last bin
        (_mc_pol_1d, 1.5, 0.0, 0.0, None),
        (_mc_cart_2d, 1.0, 0.0, 0.0, (7, 4)),
        (_mc_cart_2d, 1.0, np.pi, 0.0, (0, 4)),
        (_mc_cart_2d, 0.25, 0.0, 0.0, (5, 4)),
        (_mc_cart_2d, 0.25, np.pi, 0.0, (3, 4)),
        (_mc_cart_2d, 0.1, 0.0, 0.0, (4, 4)),
        (_mc_cart_2d, 1.0, np.pi / 2, 0.0, (4, 7)),
        (_mc_cart_2d, 1.5, 0.0, 0.0, None),  # vx outside
        (_mc_cart_2d, 1.5, np.pi / 2, 0.0, None),  # vy outside
        (_mc_cart_3d, 1.0, 0.0, 0.0, (7, 4, 4)),
        (_mc_cart_3d, 1.0, np.pi, 0.0, (0, 4, 4)),
        (_mc_cart_3d, 0.25, np.pi, 0.0, (3, 4, 4)),
        (_mc_cart_3d, 1.0, np.pi / 2, 0.0, (4, 7, 4)),
        (_mc_cart_3d, 1.0, 0.0, np.pi / 2, (4, 4, 7)),
        (_mc_cart_3d, 1.5, 0.0, 0.0, None),  # vx outside
        (_mc_cart_3d, 1.5, np.pi / 2, 0.0, None),  # vy outside
        (_mc_cart_3d, 1.5, 0.0, np.pi / 2, None),  # vz outside
    )
    @unpack
    def test_mc_kernels_bin_edges(self, kernel, speed, phi, theta, expected):
        # Bins are closed on the left and open on the right (MATLAB discretize),
        # except the last one, which is closed on both ends.
        if kernel is _mc_pol_1d:
            edges = np.array([0.1, 0.25, 0.5, 1.0])
        else:
            edges = np.arange(-1.0, 1.01, 0.25)  # exact, with an edge at 0

        result = self._mc_kernel_runs(kernel, [speed], [phi], [theta], edges)

        if expected is None:
            np.testing.assert_array_equal(result, 0.0)
        else:
            self.assertListEqual(np.argwhere(result).tolist(), [list(expected)])
            self.assertAlmostEqual(result[expected], speed**2 * np.cos(theta))

    @data(
        (_mc_cart_2d, np.linspace(-1.01, 1.01, 102)),
        (_mc_cart_2d, np.linspace(-1.0, 1.0, 11)),
        (_mc_cart_3d, np.linspace(-1.01, 1.01, 102)),
        (_mc_cart_3d, np.linspace(-1.0, 1.0, 11)),
    )
    @unpack
    def test_mc_cart_fast_index_on_edges(self, kernel, edges):
        # (edge - edges[0]) / step truncates to one bin too low or too high for
        # many edges of these grids: the fast index must still match
        # np.searchsorted for points on and next to every edge, in each direction.
        v_abs = np.unique(np.abs(edges[edges != 0]))
        speed = np.hstack([v_abs, np.nextafter(v_abs, 0), np.nextafter(v_abs, 2)])
        speed = np.sort(speed)
        phi = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        theta = [0.0, np.pi / 2] if kernel is _mc_cart_3d else [0.0]

        result = self._mc_kernel_runs(kernel, speed, phi, theta, edges)

        # Every particle within the grid is kept (none past the outer edges)
        v_in = speed[speed <= edges[-1]]
        expected = np.sum(v_in**2) * len(phi) * np.sum(np.cos(theta))
        np.testing.assert_allclose(np.sum(result), expected, rtol=1e-12)

    @data(
        # phi, theta, v_lim, a_lim, kept
        (_mc_pol_1d, np.pi / 2, 0.0, (-np.inf, 0.5), None, False),
        (_mc_pol_1d, np.pi / 2, 0.0, (0.5, np.inf), None, True),
        (_mc_pol_1d, np.pi / 2, 0.0, None, (0.0, np.pi / 4), False),
        (_mc_pol_1d, np.pi / 2, 0.0, None, (np.pi / 4, np.pi), True),
        (_mc_cart_2d, 0.0, np.pi / 6, (-0.25, 0.25), None, False),
        (_mc_cart_2d, 0.0, np.pi / 6, (0.25, 1.0), None, True),
        (_mc_cart_2d, 0.0, np.pi / 6, None, (-np.pi / 12, np.pi / 12), False),
        (_mc_cart_2d, 0.0, np.pi / 6, None, (np.pi / 12, np.pi / 4), True),
        (_mc_cart_3d, 0.0, np.pi / 6, (-0.25, 0.25), None, False),
        (_mc_cart_3d, 0.0, np.pi / 6, None, (np.pi / 12, np.pi / 4), True),
    )
    @unpack
    def test_mc_kernels_limits(self, kernel, phi, theta, v_lim, a_lim, kept):
        # v_lim/a_lim bound the transverse speed and its angle (1d), or the
        # out-of-plane speed and elevation (2d, 3d). Particle of unit speed.
        edges = np.arange(-1.0, 1.01, 0.25)
        v_lim = (-np.inf, np.inf) if v_lim is None else v_lim
        result = self._mc_kernel_runs(
            kernel, [1.0], [phi], [theta], edges, v_lim=v_lim, a_lim=a_lim
        )

        self.assertAlmostEqual(np.sum(result), np.cos(theta) if kept else 0.0)

    @data(
        ("pol", "1d", None, False),
        ("pol", "1d", "lin", False),
        ("pol", "1d", "log", False),
        ("cart", "2d", None, False),
        ("cart", "3d", None, False),
        ("pol", "1d", None, True),
        ("pol", "1d", "lin", True),
        ("pol", "1d", "log", True),
        ("cart", "2d", None, True),
    )
    @unpack
    def test_int_sph_dist_conservation(self, base, dim, weight, with_nan):
        # With a grid covering all the Monte-Carlo particles and no limits, the
        # projected distribution integrates to sum(f * dtau) over the instrument
        # bins, whatever the random draws and the weighting. NaNs are empty
        # bins (they used to give NaN bins, or with a weighting a zero output),
        # and log weighting gives every bin particles (log10(f + 1) was 0 for
        # f < 1e-16 s^3/m^6).
        vdf, _, speed, phi, theta = self._maxwellian()

        if with_nan:
            vdf_nan = vdf.copy()
            vdf_nan[[5, 12, 20], [3, 10, 30], [2, 8, 15]] = np.nan
            vdf = np.where(np.isnan(vdf_nan), 0.0, vdf)
        else:
            vdf_nan = vdf

        v_edges = _speed_bin_edges(speed)
        d_angle = np.deg2rad(11.25)
        dtau = speed[:, None, None] ** 2 * np.diff(v_edges)[:, None, None]
        dtau = dtau * np.cos(theta) * d_angle**2
        v_max = 1.01 * v_edges[-1]
        n_grid = {"1d": 200, "2d": 50, "3d": 20}[dim]
        d_phi_g = 2 * np.pi / 32
        phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, 32) + d_phi_g / 2

        out = pyrf.int_sph_dist(
            vdf_nan,
            speed,
            phi,
            theta,
            None,
            phi_grid,
            projection_base=base,
            projection_dim=dim,
            velocity_grid_edges=np.linspace(-v_max, v_max, n_grid + 1),
            n_mc=5,
            weight=weight,
        )
        self.assertTrue(np.all(np.isfinite(out["f"])))
        n, _ = self._moments(out, int(dim[0]))
        self.assertAlmostEqual(n / np.sum(vdf * dtau), 1.0, delta=1e-9)

    @data(
        ("cart", "2d", None),  # phi_grid is not used by cartesian projections
        ("cart", "3d", None),
        ("CART", "2D", np.linspace(0, 2 * np.pi, 32, endpoint=False)),
        ("Pol", "1D", None),
    )
    @unpack
    def test_int_sph_dist_projection_options(self, base, dim, phi_grid):
        vdf, _, speed, phi, theta = self._maxwellian()
        grid = np.linspace(-1500e3, 1500e3, 21)
        out = pyrf.int_sph_dist(
            vdf,
            speed,
            phi,
            theta,
            grid,
            phi_grid,
            projection_base=base,
            projection_dim=dim,
            n_mc=2,
        )
        self.assertTupleEqual(out["f"].shape, (21,) * int(dim[0]))

    @data(
        # 200 bins of 125 km/s, one of them split in two (passed the one-sided
        # check and was normalised with the mean width)
        np.sort(np.hstack([np.linspace(-1500e3, 1500e3, 201), 7.5e3])),
        np.geomspace(1e3, 1500e3, 50),
    )
    def test_int_sph_dist_non_uniform_cart_grid(self, grid_edges):
        vdf, _, speed, phi, theta = self._maxwellian()
        with self.assertRaises(ValueError):
            pyrf.int_sph_dist(
                vdf,
                speed,
                phi,
                theta,
                None,
                phi,
                projection_base="cart",
                projection_dim="2d",
                velocity_grid_edges=grid_edges,
            )

    @data(
        # below zero speed (Monte-Carlo particles reversed), negative widths,
        # wrong numbers of values
        lambda v: {"d_v_m": 2 * v, "d_v_p": 0.1 * v},
        lambda v: {"d_v_m": 0.1 * v, "d_v_p": -0.1 * v},
        lambda v: {"d_v_m": 0.1 * v[:-1], "d_v_p": 0.1 * v[:-1]},
        lambda v: {"velocity_edges": np.hstack([-1.0, v])},
        lambda v: {"velocity_edges": v},
    )
    def test_int_sph_dist_speed_widths_input(self, make_options):
        vdf, _, speed, phi, theta = self._maxwellian()
        options = make_options(speed)
        grid = np.linspace(-1500e3, 1500e3, 21)
        with self.assertRaises(ValueError):
            pyrf.int_sph_dist(vdf, speed, phi, theta, grid, None, **options)

    def test_int_sph_dist_partial_grid(self):
        # Particles outside the grid are dropped: the density is the fraction of
        # the Maxwellian with |vx| < 500 km/s.
        v_d = np.array([300e3, -200e3, 100e3])
        vdf, _, speed, phi, theta = self._maxwellian(v_d)
        v_th = np.sqrt(2 * 1.602176634e-19 * 1000.0 / 1.67262192369e-27)
        edges = np.linspace(-500e3, 500e3, 101)

        out = pyrf.int_sph_dist(
            vdf, speed, phi, theta, None, None, velocity_grid_edges=edges, n_mc=20
        )
        n, _ = self._moments(out, 1)
        fraction = 0.5 * (
            math.erf((500e3 - v_d[0]) / v_th) - math.erf((-500e3 - v_d[0]) / v_th)
        )
        self.assertAlmostEqual(n / 1e6, fraction, delta=0.01)

    def test_int_sph_dist_rotation(self):
        # Projection plane (x', y') = (y, z): the bulk velocity is (Vy, Vz)
        v_d = np.array([300e3, -200e3, 100e3])
        vdf, _, speed, phi, theta = self._maxwellian(v_d)
        d_phi_g = 2 * np.pi / 32
        phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, 32) + d_phi_g / 2
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        out = pyrf.int_sph_dist(
            vdf,
            speed,
            phi,
            theta,
            None,
            phi_grid,
            projection_base="cart",
            projection_dim="2d",
            velocity_grid_edges=np.linspace(-1500e3, 1500e3, 102),
            n_mc=20,
            xyz=xyz,
        )
        n, v = self._moments(out, 2)
        self.assertAlmostEqual(n / 1e6, 1.0, delta=0.01)
        np.testing.assert_allclose(v / 1e3, v_d[[1, 2]] / 1e3, atol=3.0)


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

    def test_iplasma_calc_defaults(self):
        # Empty answers use the defaults in the prompts: 10 nT, 1 cc, 100 eV, 1000 eV
        with mock.patch.object(builtins, "input", lambda _: ""):
            result = pyrf.iplasma_calc(True, False)

        gamma = 100.0 * constants.e / (constants.m_e * constants.c**2) + 1
        v_te = constants.c * np.sqrt(1 - 1 / gamma**2)
        self.assertAlmostEqual(result["v_te"] / v_te, 1.0, places=12)
        self.assertAlmostEqual(result["w_ce"], constants.e * 1e-8 / constants.m_e)
        self.assertAlmostEqual(
            result["w_pp"] ** 2 / (1e6 * constants.e**2),
            1 / (constants.m_p * constants.epsilon_0),
        )

        expected = pyrf.iplasma_calc(True, False, 10.0, 1.0, 100.0, 1000.0)
        for key, value in expected.items():
            self.assertAlmostEqual(result[key] / value, 1.0, places=12, msg=key)

    def test_iplasma_calc_values(self):
        def fail_input(_):
            raise AssertionError("input must not be called")

        with mock.patch.object(builtins, "input", fail_input):
            result = pyrf.iplasma_calc(True, False, 10.0, 1.0, 100.0, 1000.0)

        q_e, m_e, ep0 = constants.e, constants.m_e, constants.epsilon_0
        gamma = 100.0 * q_e / (m_e * constants.c**2) + 1
        v_te = constants.c * np.sqrt(1 - 1 / gamma**2)
        self.assertAlmostEqual(result["v_te"] / v_te, 1.0, places=12)

        # e-/ion collision frequency n e^4 / (16 pi eps0^2 me^2 Vte^3)
        f_col = 1e6 * q_e**4 / (16 * np.pi * ep0**2 * m_e**2 * v_te**3)
        self.assertAlmostEqual(result["f_col"] / f_col, 1.0, places=12)
        self.assertAlmostEqual(f_col, 9.66e-10, delta=1e-12)

        self.assertAlmostEqual(result["p_mag"], 1e-18 / (2 * constants.mu_0))


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

    def test_l_shell_values(self):
        # 3 R_E on the SM equator -> L = 3; 2 R_E at 60 deg latitude -> L = 8
        lat = np.deg2rad(60.0)
        r_sm = np.array(
            [
                [3 * R_E, 0.0, 0.0],
                [0.0, -3 * R_E, 0.0],
                [2 * R_E * np.cos(lat), 0.0, 2 * R_E * np.sin(lat)],
                [0.0, -2 * R_E * np.cos(lat), -2 * R_E * np.sin(lat)],
            ]
        )
        r_xyz = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 4), r_sm, {"COORDINATE_SYSTEM": "sm"}
        )
        result = pyrf.l_shell(r_xyz)
        np.testing.assert_allclose(result.data, [3.0, 3.0, 8.0, 8.0], rtol=1e-12)
        self.assertTupleEqual(result.dims, ("time",))
        self.assertListEqual(list(result.coords), ["time"])
        self.assertEqual(result.attrs["UNITS"], "R_E")


class MatchPhibeTestCase(unittest.TestCase):
    # Potential phi(x - v t) along k perpendicular to B = 40 nT z, with
    # E.k = (dphi / dt) / v, dB_par = phi n e mu0 / B and an unrelated E along
    # z x k. For B along z, k = cos(theta) (-y) + sin(theta) x.
    b_0, n_0, v_ph, theta = 40.0, 10.0, 300.0, np.deg2rad(31.0)

    def setUp(self):
        t_sec = np.arange(4096) / 1024.0
        time = generate_timeline(1024.0, 4096)
        window = np.hanning(len(t_sec))
        phi = 50.0 * np.sin(2 * np.pi * 40.0 * t_sec) * window
        e_k = np.gradient(phi, t_sec) / self.v_ph
        e_n = np.max(np.abs(e_k)) * np.sin(2 * np.pi * 25.0 * t_sec) * window
        self.k_vec = np.array([np.sin(self.theta), -np.cos(self.theta), 0.0])
        n_vec = np.cross([0.0, 0.0, 1.0], self.k_vec)
        d_b = phi * self.n_0 * 1e6 * constants.e * constants.mu_0 / self.b_0 * 1e18
        self.e_xyz = pyrf.ts_vec_xyz(
            time, np.outer(e_k, self.k_vec) + np.outer(e_n, n_vec)
        )
        self.b_xyz = pyrf.ts_vec_xyz(
            time, np.outer(d_b, [0.0, 0.0, 1.0]) + [0, 0, self.b_0]
        )

    def test_match_phibe_dir_values(self):
        b_data = self.b_xyz.data.copy()
        x_, _, z_, corr_vec, int_e_dt, b_z, b_0, *_ = pyrf.match_phibe_dir(
            self.b_xyz, self.e_xyz, f=10.0
        )
        k_best = np.argmax(corr_vec)
        self.assertEqual(np.arange(1, 360, 3)[k_best], 31)
        np.testing.assert_allclose(x_[k_best], self.k_vec, atol=1e-6)
        np.testing.assert_allclose(z_[0], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertGreater(corr_vec[k_best], 0.99)
        self.assertAlmostEqual(b_0, self.b_0, places=6)
        self.assertListEqual(list(int_e_dt.shape), [4096, 120])
        self.assertListEqual(list(b_z.shape), [4096])
        np.testing.assert_array_equal(self.b_xyz.data, b_data)

        # Given angles are used
        x_, *_ = pyrf.match_phibe_dir(self.b_xyz, self.e_xyz, [31.0], 10.0)
        np.testing.assert_allclose(x_[0], self.k_vec, atol=1e-6)

    def test_match_phibe_v_values(self):
        _, _, _, corr_vec, int_e_dt, b_z, b_0, *_ = pyrf.match_phibe_dir(
            self.b_xyz, self.e_xyz, f=10.0
        )
        k_best, sl_ = np.argmax(corr_vec), slice(1024, 3072)
        n_e = np.array([5.0, 10.0, 20.0])
        v_ph = np.array([100.0, 300.0, 1000.0])

        corr_mat, phi_b, phi_e = pyrf.match_phibe_v(
            b_0, b_z[sl_], int_e_dt[sl_, k_best], n_e, v_ph
        )

        # Best amplitude match at the true density and speed, n unchanged
        self.assertTupleEqual(
            np.unravel_index(np.argmin(np.abs(corr_mat)), (3, 3)), (1, 1)
        )
        self.assertListEqual(list(phi_b.shape), [2048, 3])
        self.assertListEqual(list(phi_e.shape), [2048, 3])
        np.testing.assert_array_equal(n_e, [5.0, 10.0, 20.0])


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

    def test_mean_dipole_sign_per_sample(self):
        # B.r > 0 for the first three samples and < 0 for the last three, so
        # the sign of Y = (z x b) sign(b.r) must flip in the middle.
        time = generate_timeline(1.0, 6)
        r_xyz = pyrf.ts_vec_xyz(time, np.tile([1.0, 0.0, 0.0], (6, 1)))
        b_x = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        b_xyz = pyrf.ts_vec_xyz(time, np.column_stack([b_x, np.ones(6), np.zeros(6)]))
        z_dip = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 1.0], (6, 1)))
        inp = pyrf.ts_vec_xyz(time, np.tile([1.0, 0.0, 1.0], (6, 1)))

        result = pyrf.mean(inp, r_xyz, b_xyz, z_dip)

        h = 1.0 / np.sqrt(2.0)
        expected = np.array([[-1.0, -h, h]] * 3 + [[1.0, h, -h]] * 3)
        np.testing.assert_allclose(result.data, expected, atol=1e-12)


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

    @staticmethod
    def _field(seed=0):
        # Variances 25, 4 and 1 along a random right-handed frame, plus a mean
        rng = np.random.default_rng(seed)
        basis = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
        b_data = (rng.normal(size=(2000, 3)) * [5.0, 2.0, 1.0]) @ basis.T
        b_data += [3.0, -2.0, 10.0]
        b_xyz = pyrf.ts_vec_xyz(
            generate_timeline(16.0, 2000),
            b_data,
            attrs={"COORDINATE_SYSTEM": "GSE", "UNITS": "nT"},
        )
        return b_xyz, basis

    def test_mva_values(self):
        b_xyz, basis = self._field()
        b_lmn, lamb, lmn = pyrf.mva(b_xyz)

        np.testing.assert_allclose(lamb, [25.0, 4.0, 1.0], rtol=0.05)
        np.testing.assert_allclose(np.abs(np.sum(lmn * basis, axis=0)), 1, atol=1e-3)

        # Output keeps the attributes, in LMN coordinates (they used to be dropped)
        self.assertEqual(b_lmn.attrs["COORDINATE_SYSTEM"], "lmn")
        self.assertEqual(b_lmn.attrs["UNITS"], "nT")
        np.testing.assert_allclose(b_lmn.data, b_xyz.data @ lmn)

    def test_mva_flag_case(self):
        # "MVAR" passed the validation and was then computed as "td"
        b_xyz, _ = self._field()
        for lower, upper in [("mvar", "MVAR"), ("<bn>=0", "<BN>=0"), ("td", "TD")]:
            np.testing.assert_allclose(
                pyrf.mva(b_xyz, upper)[1], pyrf.mva(b_xyz, lower)[1]
            )

    @data("mvar", "<bn>=0", "td")
    def test_mva_real_output(self, flag):
        # np.linalg.eig returns complex arrays with NumPy >= 2.5: the frame, the
        # eigenvalues and the rotated field must stay real
        b_xyz = generate_ts(64.0, 100, tensor_order=1)
        b_lmn, lamb, lmn = pyrf.mva(b_xyz, flag)

        for value in [b_lmn.data, lamb, lmn]:
            self.assertTrue(np.isrealobj(value))

    @data("mvar", "<bn>=0", "td")
    def test_mva_right_handed(self, flag):
        # The "<bn>=0" frame used to be left-handed about a third of the time
        for seed in range(10):
            _, _, lmn = pyrf.mva(self._field(seed)[0], flag)
            handedness = np.dot(np.cross(lmn[:, 0], lmn[:, 1]), lmn[:, 2])
            self.assertAlmostEqual(handedness, 1.0, places=6)


@ddt
class NewXyzTestCase(unittest.TestCase):
    @data(
        generate_ts(64.0, 100, tensor_order=1), generate_ts(64.0, 100, tensor_order=2)
    )
    def test_new_xyz_output(self, inp):
        result = pyrf.new_xyz(inp, np.random.random((3, 3)))
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.ndim, inp.ndim)

    def test_new_xyz_coordinate_system(self):
        # New frame with unit vectors (y, z, x) of the original one as columns
        time = generate_timeline(1.0, 5)
        inp = pyrf.ts_vec_xyz(
            time, np.tile([1.0, 2.0, 3.0], (5, 1)), attrs={"COORDINATE_SYSTEM": "GSE"}
        )
        trans_mat = np.eye(3)[:, [1, 2, 0]]

        result = pyrf.new_xyz(inp, trans_mat, "lmn")
        np.testing.assert_allclose(result.data, np.tile([2.0, 3.0, 1.0], (5, 1)))
        self.assertEqual(result.attrs["COORDINATE_SYSTEM"], "lmn")

        # Default: the original label is removed (it used to be kept, so that
        # cotrans treated the rotated data as GSE)
        result = pyrf.new_xyz(inp, trans_mat)
        self.assertNotIn("COORDINATE_SYSTEM", result.attrs)
        with self.assertRaises(ValueError):
            pyrf.cotrans(result, "gsm")

        # The caller's attributes are unchanged
        self.assertEqual(inp.attrs["COORDINATE_SYSTEM"], "GSE")

        with self.assertRaises(TypeError):
            pyrf.new_xyz(inp, trans_mat, 1)


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

    def test_trace_input_unchanged(self):
        # The output got the input's attrs dict, so the input became TENSOR_ORDER 0
        inp = generate_ts(64.0, 100, tensor_order=2)
        attrs = dict(inp.attrs)
        result = pyrf.trace(inp)

        self.assertDictEqual(inp.attrs, attrs)
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)


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


def _repeated_times():
    # Samples 1 and 2 are identical, sample 4 is 50 ns after sample 3 and
    # sample 6 is exactly 100 ns after sample 5 (not a repeat)
    t_0 = np.datetime64("2019-01-01T00:00:00", "ns")
    offsets = np.array([0, 10**9, 10**9, 2 * 10**9, 2 * 10**9 + 50, 3 * 10**9])
    offsets = np.r_[offsets, offsets[-1] + 100]
    return t_0 + offsets.astype("timedelta64[ns]")


@ddt
class RemoveRepeatedPointsTestCase(unittest.TestCase):
    @data(0, 1, 2)
    def test_remove_repeated_points_dataarray(self, tensor_order):
        time = _repeated_times()
        data_ = np.arange(len(time) * 3**tensor_order, dtype=float)
        data_ = data_.reshape([len(time)] + [3] * tensor_order)
        inp = generate_ts(1.0, len(time), tensor_order=tensor_order)
        inp = inp.copy(data=data_).assign_coords(time=time)
        inp.attrs["UNITS"] = "nT"

        result = pyrf.remove_repeated_points(inp)

        # The later point of each repeated pair is kept
        keep = [0, 2, 4, 5, 6]
        np.testing.assert_array_equal(result.time.data, time[keep])
        np.testing.assert_array_equal(result.data, data_[keep])
        self.assertTupleEqual(result.dims, inp.dims)
        self.assertEqual(result.attrs["UNITS"], "nT")

    def test_remove_repeated_points_dataset(self):
        time = _repeated_times()
        inp = xr.Dataset(
            {"z_ra": ("time", np.arange(7.0)), "z_dec": ("time", -np.arange(7.0))},
            coords={"time": time},
        )
        result = pyrf.remove_repeated_points(inp)
        np.testing.assert_array_equal(result.z_ra.data, [0.0, 2.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result.time.data, time[[0, 2, 4, 5, 6]])

    def test_remove_repeated_points_dict(self):
        time = _repeated_times()
        inp = {"time": time, "data": np.arange(14.0).reshape(7, 2)}
        result = pyrf.remove_repeated_points(inp)
        np.testing.assert_array_equal(result["time"], time[[0, 2, 4, 5, 6]])
        np.testing.assert_array_equal(result["data"][:, 0], [0, 4, 8, 10, 12])

        # int64 time in ns, and the caller's dict is unchanged
        inp = {"time": time.astype(np.int64), "data": np.arange(7.0)}
        result = pyrf.remove_repeated_points(inp)
        np.testing.assert_array_equal(result["data"], [0.0, 2.0, 4.0, 5.0, 6.0])
        self.assertEqual(len(inp["data"]), 7)


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

    T_0 = np.datetime64("2019-09-14T07:54:00", "ns")

    def _ts(self, t_s, data):
        times = self.T_0 + np.round(np.asarray(t_s) * 1e9).astype("timedelta64[ns]")
        return pyrf.ts_scalar(times, np.asarray(data, dtype=np.float64))

    @staticmethod
    def _matlab_average(data, centres, half, thresh=0.0):
        # irf_resamp: samples in (t - dt/2, t + dt/2], points farther than
        # thresh * std (N - 1) from the mean disregarded
        idx = np.arange(len(data))
        out = []
        for i in centres:
            window = data[(idx > i - half) & (idx <= i + half)]
            if thresh:
                keep = np.abs(window - window.mean()) <= thresh * window.std(ddof=1)
                window = window[keep]
            out.append(window.mean())
        return np.array(out)

    def test_resample_average_window(self):
        # 128 Hz ramp to 32 Hz with samples on the window edges: each sample is in
        # one window (they were in two, or one, depending on round-off)
        inp = self._ts(np.arange(1280) / 128.0, np.arange(1280.0))
        ref = self._ts(np.arange(4, 316) / 32.0, np.zeros(312))
        result = pyrf.resample(inp, ref)

        expected = self._matlab_average(np.arange(1280.0), np.arange(4, 316) * 4, 2)
        np.testing.assert_allclose(result.data, expected)

    def test_resample_average_nan(self):
        # A NaN only affects its window; windows without samples are NaN
        data = np.arange(1280.0)
        data[400] = np.nan
        inp = self._ts(np.arange(1280) / 128.0, data)
        ref = self._ts(np.arange(4, 330) / 32.0, np.zeros(326))
        result = pyrf.resample(inp, ref)

        # window j holds samples 4 (j + 4) - 1 .. 4 (j + 4) + 2; the last sample
        # (1279) is in window 316
        self.assertTrue(np.isnan(result.data[96]))  # window of sample 400
        self.assertTrue(np.all(np.isnan(result.data[317:])))  # after the data
        self.assertEqual(int(np.isnan(result.data[:317]).sum()), 1)

    def test_resample_thresh(self):
        # thresh always raised AssertionError; points farther than thresh * std
        # are disregarded as in irf_resamp (32 samples per window)
        data = np.random.default_rng(1).standard_normal(1280)
        data[[10, 500, 501]] = [50.0, -40.0, 60.0]
        inp = self._ts(np.arange(1280) / 128.0, data)
        ref = self._ts(np.arange(1, 39) / 4.0, np.zeros(38))
        result = pyrf.resample(inp, ref, thresh=2.0)

        expected = self._matlab_average(data, np.arange(1, 39) * 32, 16, thresh=2.0)
        np.testing.assert_allclose(result.data, expected)
        self.assertLess(abs(result.data[0]), 1.0)  # the 50 outlier is removed

    def test_resample_irregular_first_interval(self):
        # The sampling frequency was guessed from the first interval
        inp = self._ts(np.arange(1280) / 128.0, np.arange(1280.0))
        ref = self._ts(np.r_[0.0, 0.0375 + np.arange(1, 300) / 32.0], np.zeros(300))
        result = pyrf.resample(inp, ref)

        np.testing.assert_allclose(result.data[1:4], [8.5, 12.5, 16.5])

    def test_resample_float_time(self):
        # Numeric times are seconds (their bits were read as integers)
        inp = xr.DataArray(
            np.arange(1280.0), coords=[np.arange(1280) / 128.0], dims=["time"]
        )
        ref = xr.DataArray(
            np.zeros(312), coords=[np.arange(4, 316) / 32.0], dims=["time"]
        )
        result = pyrf.resample(inp, ref)

        expected = self._matlab_average(np.arange(1280.0), np.arange(4, 316) * 4, 2)
        np.testing.assert_allclose(result.data, expected)

        with self.assertRaises(TypeError):
            pyrf.resample(inp, self._ts(np.arange(10) / 32.0, np.zeros(10)))

    def test_resample_extrapolation(self):
        # Linear extrapolation outside the time range of inp (as irf_resamp)
        inp = self._ts(np.arange(10.0), np.arange(10.0))
        ref = self._ts(np.arange(-2.0, 12.0, 0.5), np.zeros(28))
        result = pyrf.resample(inp, ref)

        np.testing.assert_allclose(result.data, np.arange(-2.0, 12.0, 0.5))


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

    # S for E = 1 mV/m and B = 1 nT (E perpendicular to B), in mW/m^2
    S_UNIT = 1e-9 / (4 * np.pi * 1e-7)

    @staticmethod
    def _fields(n_pts=100, f_s=10.0):
        # Constant E = (0, 0, 1) mV/m and B = (1, 1, 0) nT: S = (-1, 1, 0) S_UNIT
        times = pyrf.unix2datetime64(np.arange(n_pts) / f_s)
        e_xyz = pyrf.ts_vec_xyz(times, np.tile([0.0, 0.0, 1.0], (n_pts, 1)))
        b_xyz = pyrf.ts_vec_xyz(times, np.tile([1.0, 1.0, 0.0], (n_pts, 1)))
        return e_xyz, b_xyz

    def test_poynting_flux_integral(self):
        # Integral of each component (np.cumsum used to mix the components):
        # S * (k + 1) / f_s after k + 1 samples
        e_xyz, b_xyz = self._fields()
        s_xyz, int_s = pyrf.poynting_flux(e_xyz, b_xyz)

        np.testing.assert_allclose(
            s_xyz.data, np.tile([-1, 1, 0], (100, 1)) * self.S_UNIT
        )
        self.assertTupleEqual(int_s.dims, ("time", "comp"))
        np.testing.assert_array_equal(int_s.time.data, e_xyz.time.data)
        expected = np.arange(1, 101)[:, None] / 10.0 * np.array([-1, 1, 0])
        np.testing.assert_allclose(int_s.data, expected * self.S_UNIT, atol=1e-15)
        self.assertEqual(int_s.attrs["UNITS"], "mW s/m^2")

    def test_poynting_flux_nan(self):
        # NaNs count as zero in the integrals only; the returned S and S_z keep
        # them, and the valid components of S at that time
        e_xyz, b_xyz = self._fields()
        e_xyz.data[10, 0] = np.nan  # S = (-1, nan, nan)
        b_hat = pyrf.ts_vec_xyz(e_xyz.time.data, np.tile([-1.0, 1.0, 0.0], (100, 1)))

        s_xyz, int_s = pyrf.poynting_flux(e_xyz, b_xyz)
        np.testing.assert_allclose(s_xyz.data[10], [-self.S_UNIT, np.nan, np.nan])
        np.testing.assert_allclose(
            int_s.data[-1], np.array([-10.0, 9.9, 0.0]) * self.S_UNIT, atol=1e-15
        )

        s_xyz, s_z, int_s_z = pyrf.poynting_flux(e_xyz, b_xyz, b_hat)
        self.assertTrue(np.isnan(s_z.data[10]))
        np.testing.assert_allclose(s_z.data[11], np.sqrt(2) * self.S_UNIT)
        self.assertAlmostEqual(
            float(int_s_z.data[-1]) / (np.sqrt(2) * self.S_UNIT), 9.9, places=10
        )

    def test_poynting_flux_same_length_shifted(self):
        # Same number of samples but times shifted by half a sample: B is
        # resampled to the times of E. With Bx = 1 + t, Sy = (1 + t) S_UNIT at
        # the times t of E (the samples of B are 50 ms later).
        n_pts, f_s = 100, 10.0
        t_e = np.arange(n_pts) / f_s
        t_b = t_e + 0.05
        e_xyz = pyrf.ts_vec_xyz(
            pyrf.unix2datetime64(t_e), np.tile([0.0, 0.0, 1.0], (n_pts, 1))
        )
        b_data = np.stack([1 + t_b, np.ones(n_pts), np.zeros(n_pts)], axis=1)
        b_xyz = pyrf.ts_vec_xyz(pyrf.unix2datetime64(t_b), b_data)

        s_xyz, _ = pyrf.poynting_flux(e_xyz, b_xyz)

        t_s = (s_xyz.time.data - e_xyz.time.data[0]) / np.timedelta64(1, "s")
        self.assertTrue(np.all(np.isin(s_xyz.time.data, e_xyz.time.data)))
        np.testing.assert_allclose(s_xyz.data[:, 1], (1 + t_s) * self.S_UNIT)


class PsdTestCase(unittest.TestCase):
    F_S = 64.0

    def _sines(self, amplitudes, freqs, n_pts=8192):
        # Sum of sines, one column per component (bin-centred frequencies)
        t = np.arange(n_pts) / self.F_S
        data = np.stack(
            [a * np.sin(2 * np.pi * f * t) for a, f in zip(amplitudes, freqs)],
            axis=1,
        )
        return generate_timeline(self.F_S, n_pts), data

    def test_psd_scalar(self):
        # Peak at the sine frequency, and Parseval: int PSD df = variance
        time, data = self._sines([2.0], [5.0])
        result = pyrf.psd(pyrf.ts_scalar(time, data[:, 0]))

        self.assertTupleEqual(result.dims, ("f",))
        self.assertAlmostEqual(float(result.f[np.argmax(result.data)]), 5.0)
        power = np.sum(result.data) * float(result.f[1] - result.f[0])
        self.assertAlmostEqual(power, 2.0**2 / 2, delta=0.02)

    def test_psd_vector(self):
        # One spectrum per component (the spectrum was taken over the
        # component axis of the rectified signal and raised a ValueError)
        amplitudes, freqs = [1.0, 2.0, 3.0], [2.5, 5.0, 10.0]
        time, data = self._sines(amplitudes, freqs)
        result = pyrf.psd(pyrf.ts_vec_xyz(time, data))

        self.assertTupleEqual(result.dims, ("f", "comp"))
        self.assertListEqual(list(result.comp.data), ["x", "y", "z"])
        d_f = float(result.f[1] - result.f[0])
        for i, (amplitude, freq) in enumerate(zip(amplitudes, freqs)):
            spectrum = result.data[:, i]
            self.assertAlmostEqual(float(result.f[np.argmax(spectrum)]), freq)
            self.assertAlmostEqual(
                np.sum(spectrum) * d_f, amplitude**2 / 2, delta=0.02 * amplitude**2
            )

    def test_psd_tensor_and_options(self):
        time, data = self._sines([1.0] * 9, [5.0] * 9, n_pts=2048)
        result = pyrf.psd(pyrf.ts_tensor_xyz(time, data.reshape(-1, 3, 3)))
        self.assertEqual(result.shape[1:], (3, 3))

        # n_overlap=None: 256-point segments (used to raise with a float)
        result = pyrf.psd(pyrf.ts_scalar(time, data[:, 0]), n_overlap=None)
        self.assertEqual(len(result.f), 129)


class PresAnisTestCase(unittest.TestCase):
    def test_pres_anis_output(self):
        result = pyrf.pres_anis(
            generate_ts(64.0, 100, tensor_order=2),
            generate_ts(64.0, 100, tensor_order=1),
        )
        self.assertIsInstance(result, xr.DataArray)


class PviTestCase(unittest.TestCase):
    def test_pvi_scalar(self):
        # Increments 1, 2, 3, 4 and <|dx|^2> = 7.5
        time = generate_timeline(1.0, 5)
        inp = pyrf.ts_scalar(time, np.array([0.0, 1.0, 3.0, 6.0, 10.0]))
        result = pyrf.pvi(inp, 1)
        np.testing.assert_allclose(result.data, np.arange(1, 5) / np.sqrt(7.5))
        np.testing.assert_array_equal(result.time.data, time[:4])

    def test_pvi_vector(self):
        # Constant increments give PVI = 1, labelled at the first sample
        time = generate_timeline(1.0, 6)
        steps = np.arange(6.0)
        inp = pyrf.ts_vec_xyz(
            time, np.column_stack([steps, 2 * steps, 0 * steps]), {"UNITS": "nT"}
        )
        result = pyrf.pvi(inp, 2)
        np.testing.assert_allclose(result.data, np.ones(4))
        np.testing.assert_array_equal(result.time.data, time[:4])
        self.assertNotIn("UNITS", result.attrs)
        self.assertEqual(result.attrs["TENSOR_ORDER"], 0)
        self.assertEqual(inp.attrs["UNITS"], "nT")

    def test_pvi_nan(self):
        # Increments 1, 2, NaN, NaN, 5 and <|dx|^2> = 10 over the valid ones
        time = generate_timeline(1.0, 6)
        inp = pyrf.ts_scalar(time, np.array([0.0, 1.0, 3.0, np.nan, 10.0, 15.0]))
        result = pyrf.pvi(inp, 1)
        expected = np.array([1.0, 2.0, np.nan, np.nan, 5.0]) / np.sqrt(10.0)
        np.testing.assert_allclose(result.data, expected)

    def test_pvi_scale(self):
        inp = pyrf.ts_scalar(generate_timeline(1.0, 5), np.arange(5.0))
        for scale in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                pyrf.pvi(inp, scale)


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
        r_xyz = rot.T @ np.array([l_bs / (1 + eps), 0.0, 0.0]) * R_E
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

    def test_time_clip_values(self):
        time = generate_timeline(10.0, 20)
        data_ = np.arange(60.0).reshape(20, 3)
        tint = time[[5, 12]]

        # Dimension without coordinate (raised a ValueError)
        inp = xr.DataArray(data_, coords={"time": time}, dims=["time", "comp"])
        result = pyrf.time_clip(inp, tint)
        np.testing.assert_array_equal(result.data, data_[5:13])
        np.testing.assert_array_equal(result.time.data, time[5:13])

        # Non-dimension coordinates are kept (they were dropped) and clipped
        inp = inp.assign_coords(
            comp=["x", "y", "z"],
            r_xyz=(("time", "comp"), -data_),
            label=("comp", ["a", "b", "c"]),
        )
        inp.time.attrs["UNITS"] = "ns"
        result = pyrf.time_clip(inp, tint)
        np.testing.assert_array_equal(result.r_xyz.data, -data_[5:13])
        np.testing.assert_array_equal(result.label.data, ["a", "b", "c"])
        np.testing.assert_array_equal(result.comp.data, ["x", "y", "z"])
        self.assertEqual(result.time.attrs["UNITS"], "ns")

        # Time is not the first dimension (the first dimension was clipped)
        result = pyrf.time_clip(inp.transpose("comp", "time"), tint)
        np.testing.assert_array_equal(result.data, data_[5:13].T)

        # Strings and time series intervals
        result = pyrf.time_clip(inp, [str(t) for t in tint])
        np.testing.assert_array_equal(result.data, data_[5:13])
        result = pyrf.time_clip(inp, inp[5:13])
        np.testing.assert_array_equal(result.data, data_[5:13])

    def test_time_clip_dataset(self):
        time = generate_timeline(10.0, 20)
        energy = np.arange(4.0)
        data_ = np.arange(80.0).reshape(20, 4)
        delta = np.arange(40.0).reshape(20, 2)

        # A non-dimension coordinate (raised an IndexError)
        inp = xr.Dataset(
            {"x": (("time", "energy"), data_), "y": ("energy", -energy)},
            coords={"energy": energy, "time": time, "r": ("time", -data_[:, 0])},
            attrs={"delta": delta, "scalar": np.array(1.0), "name": "test"},
        )
        result = pyrf.time_clip(inp, time[[5, 12]])
        np.testing.assert_array_equal(result.time.data, time[5:13])
        np.testing.assert_array_equal(result.energy.data, energy)
        np.testing.assert_array_equal(result.r.data, -data_[5:13, 0])
        np.testing.assert_array_equal(result.x.data, data_[5:13])
        np.testing.assert_array_equal(result.y.data, -energy)

        # Time dependent attributes are clipped, the caller's are unchanged
        np.testing.assert_array_equal(result.attrs["delta"], delta[5:13])
        self.assertEqual(result.attrs["scalar"], 1.0)
        self.assertEqual(result.attrs["name"], "test")
        self.assertEqual(inp.attrs["delta"].shape, (20, 2))


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

    def test_ts_scalar_attrs_unchanged(self):
        # TENSOR_ORDER is set in a copy of attrs: the caller's dict is unchanged
        attrs = {"UNITS": "nT"}
        out = pyrf.ts_scalar(generate_timeline(64.0, 100), np.ones((100,)), attrs=attrs)

        self.assertDictEqual(attrs, {"UNITS": "nT"})
        self.assertDictEqual(out.attrs, {"UNITS": "nT", "TENSOR_ORDER": 0})


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

    def test_ts_vec_xyz_attrs_unchanged(self):
        # TENSOR_ORDER is set in a copy of attrs: the caller's dict is unchanged
        attrs = {"UNITS": "nT"}
        out = pyrf.ts_vec_xyz(
            generate_timeline(64.0, 100), np.ones((100, 3)), attrs=attrs
        )

        self.assertDictEqual(attrs, {"UNITS": "nT"})
        self.assertDictEqual(out.attrs, {"UNITS": "nT", "TENSOR_ORDER": 1})


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


def _waverage_loops(t_sec, data, n_pts):
    # Literal port of the irf_waverage.m loops (one column)
    weights = {
        5: np.array([0.1, 0.25, 0.3, 0.25, 0.1]),
        7: np.array([0.07, 0.15, 0.18, 0.2, 0.18, 0.15, 0.07]),
    }[n_pts]
    f_s = 1 / (t_sec[1] - t_sec[0])
    n_data = int(round((t_sec[-1] - t_sec[0]) * f_s))
    d_t = (t_sec[-1] - t_sec[0]) / n_data
    out = np.zeros(n_data + 1)
    ind = np.round((t_sec - t_sec[0]) / d_t).astype(int)
    out[ind] = data
    out[np.isnan(out)] = 0
    pad = np.r_[np.zeros(n_pts // 2), out, np.zeros(n_pts // 2)]
    for j in range(n_data + 1):
        x_ = pad[j : j + n_pts]
        cor = np.sum(weights[x_ == 0])
        out[j] = 0 if np.isclose(cor, 1) else np.sum(x_ * weights) / (1 - cor)
    return out[ind]


@ddt
class WaverageTestCase(unittest.TestCase):
    @data(5, 7)
    def test_waverage_loops(self, n_pts):
        # Random data with a gap (samples 20-22 missing) and a NaN
        rng = np.random.default_rng(0)
        keep = np.r_[0:20, 23:60]
        t_sec = np.arange(60)[keep] / 16.0
        time = np.datetime64("2019-01-01", "ns") + (t_sec * 1e9).astype(
            "timedelta64[ns]"
        )
        data_ = rng.normal(size=(len(t_sec), 3))
        data_[30, 1] = np.nan
        inp = pyrf.ts_vec_xyz(time, data_)

        result = pyrf.waverage(inp, n_pts=n_pts)

        self.assertTupleEqual(result.shape, inp.shape)
        np.testing.assert_array_equal(result.time.data, time)
        for col in range(3):
            expected = _waverage_loops(t_sec, data_[:, col], n_pts)
            np.testing.assert_allclose(result.data[:, col], expected, atol=1e-12)

    def test_waverage_constant(self):
        # A constant is unchanged, including at the edges and around a gap
        time = generate_timeline(32.0, 50)
        time = np.delete(time, [10, 11])
        inp = pyrf.ts_scalar(time, np.full(len(time), 2.5))
        result = pyrf.waverage(inp, 32.0, 5)
        np.testing.assert_allclose(result.data, 2.5)
        self.assertTupleEqual(result.dims, ("time",))


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

    @data(True, False)
    def test_vht_time_alignment(self, no_ez):
        # Same length on grids offset by 5 ms: the samples were paired by index
        time = generate_timeline(100.0, 50)
        k = np.arange(50.0)
        e = pyrf.ts_vec_xyz(time, np.stack([np.sin(k), np.cos(k), 0.1 * k], axis=1))
        b_xyz = np.stack([0 * k + 10.0, 0.2 * k, 5.0 - 0.1 * k], axis=1)
        b_on_e = pyrf.ts_vec_xyz(time, b_xyz)
        b_off = pyrf.ts_vec_xyz(
            time + np.timedelta64(5, "ms"), b_xyz + 0.5 * np.array([0, 0.2, -0.1])
        )

        expected = pyrf.vht(e, b_on_e, no_ez)
        result = pyrf.vht(e, b_off, no_ez)
        np.testing.assert_allclose(result[0], expected[0], rtol=1e-9)
        np.testing.assert_allclose(result[1].data, expected[1].data, atol=1e-9)

    @data(True, False)
    def test_vht_input_unchanged(self, no_ez):
        # Ez is set to 0 in a copy with no_ez: the caller's E is unchanged, and the
        # result equals that of E with Ez = 0 given explicitly
        e = generate_ts(64.0, 100, tensor_order=1)
        b = generate_ts(64.0, 100, tensor_order=1)
        e_data = e.data.copy()

        v_ht, _, _ = pyrf.vht(e, b, no_ez)

        np.testing.assert_array_equal(e.data, e_data)

        if no_ez:
            e_0 = e.copy(deep=True)
            e_0.data[:, 2] = 0.0
            np.testing.assert_allclose(v_ht, pyrf.vht(e_0, b, True)[0])


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

    @staticmethod
    def _field(t_sec, data):
        time = np.datetime64("2020-01-01T00:00:00", "ns") + np.round(
            t_sec * 1e9
        ).astype("timedelta64[ns]")
        return pyrf.ts_vec_xyz(time, data)

    def test_mean_field_long_series(self):
        # More than 65 535 samples (uint16 index overflow)
        n_pts = 70000
        t_sec = np.arange(n_pts) / 128.0
        trend = np.stack([10.0 + 2.0 * t_sec, -t_sec, 0.5 * t_sec**2 / 500.0], 1)
        wave = np.sin(2 * np.pi * 20.0 * t_sec)[:, None] * np.ones(3)
        b_mean, b_wave = pyrf.mean_field(self._field(t_sec, trend + wave), 2)

        np.testing.assert_allclose(b_mean.data, trend, atol=1e-3)
        np.testing.assert_allclose(b_wave.data, wave, atol=1e-3)

    def test_mean_field_nan_and_uneven_sampling(self):
        # The fit uses the sample times, and ignores NaNs
        t_sec = np.sort(np.random.default_rng(0).uniform(0.0, 100.0, 1000))
        trend = np.stack([1.0 + 0.2 * t_sec, 3.0 - 0.1 * t_sec, 0.0 * t_sec], 1)
        data = trend.copy()
        data[10, 0] = np.nan
        b_mean, b_wave = pyrf.mean_field(self._field(t_sec, data), 1)

        np.testing.assert_allclose(b_mean.data, trend, atol=1e-6)
        self.assertEqual(np.sum(np.isnan(b_wave.data)), 1)
        self.assertTrue(np.isnan(b_wave.data[10, 0]))


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

    @data(2, 5, 10, 11)
    def test_movmean_values(self, window_size):
        inp = generate_ts(64.0, 100, tensor_order=1)
        result = pyrf.movmean(inp, window_size)
        data = inp.data.astype(np.float64)

        # Window of window_size points around each time (one more after than
        # before for even windows), at the times with a full window
        i_start = (window_size - 1) // 2
        expected = [
            np.mean(data[i - i_start : i - i_start + window_size], axis=0)
            for i in range(i_start, 100 - window_size // 2)
        ]
        np.testing.assert_allclose(result.data, expected, rtol=1e-12)
        np.testing.assert_array_equal(
            result.time.data, inp.time.data[i_start : 100 - window_size // 2]
        )
        self.assertEqual(result.dtype, np.float64)

    def test_movmean_nan(self):
        inp = pyrf.ts_scalar(generate_timeline(64.0, 100), np.ones(100))
        inp.data[10] = np.nan
        inp.data[20:30] = np.nan
        result = pyrf.movmean(inp, 5)

        # NaNs are ignored; only the windows with no finite value are NaN
        self.assertEqual(np.sum(np.isnan(result.data)), 6)
        self.assertTrue(np.all(np.isnan(result.data[20:26])))
        np.testing.assert_array_equal(result.data[:20], 1.0)
        np.testing.assert_array_equal(result.data[26:], 1.0)


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
