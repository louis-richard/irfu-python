#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

# Built-in imports
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, unpack

from pyrfu import mms

from . import generate_data, generate_spectr, generate_ts, generate_vdf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


@ddt
class CalcEpsilonTestCase(unittest.TestCase):
    @data(
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_ts(32.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
        ),
    )
    @unpack
    def test_calc_epsilon_input(self, vdf, model_vdf, n_s, sc_pot):
        with self.assertRaises(ValueError):
            mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot)

    @data(
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            {},
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="electrons"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="electrons"),
            {},
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=True, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=True, species="ions"),
            {},
        ),
    )
    @unpack
    def test_calc_epsilon_output(self, vdf, model_vdf, kwargs):
        mms.calculate_epsilon(
            vdf,
            model_vdf,
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
            **kwargs,
        )


@ddt
class Dsl2GseTestCase(unittest.TestCase):
    def test_dsl2gse_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gse(generate_ts(64.0, 42, "vector"), np.random.random((42, 3)), 1)

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gse_output(self, value):
        result = mms.dsl2gse(generate_ts(64.0, 42, "vector"), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gse(generate_ts(64.0, 42, "vector"), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dsl2GsmTestCase(unittest.TestCase):
    def test_dsl2gsm_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gsm(generate_ts(64.0, 42, "vector"), np.random.random((42, 3)), 1)

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gsm_output(self, value):
        result = mms.dsl2gsm(generate_ts(64.0, 42, "vector"), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gsm(generate_ts(64.0, 42, "vector"), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class MakeModelVDFTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), False),
        (generate_vdf(64.0, 100, (32, 16, 16), species="electrons"), False),
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), True),
    )
    @unpack
    def test_make_Model_vdf_output(self, vdf, isotropic):
        result = mms.make_model_vdf(
            vdf,
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "tensor"),
            isotropic,
        )
        self.assertIsInstance(result, xr.Dataset)


@ddt
class MakeModelKappaTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="bazinga"), random.random()),
    )
    @unpack
    def test_make_model_kappa_input(self, vdf, kappa):
        with self.assertRaises(ValueError):
            mms.make_model_kappa(
                vdf,
                generate_ts(64.0, 100, "scalar"),
                generate_ts(64.0, 100, "vector"),
                generate_ts(64.0, 100, "scalar"),
                kappa,
            )

    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), random.random()),
        (generate_vdf(64.0, 100, (32, 16, 16), species="electrons"), random.random()),
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), random.random()),
    )
    @unpack
    def test_make_model_kappa_output(self, vdf, kappa):
        result = mms.make_model_kappa(
            vdf,
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "vector"),
            generate_ts(64.0, 100, "scalar"),
            kappa,
        )

        self.assertIsInstance(result, xr.Dataset)


@ddt
class ReduceTestCase(unittest.TestCase):
    @data("s^3/cm^6", "s^3/m^6", "s^3/km^6")
    def test_reduce_units(self, value):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species="ions")
        vdf.data.attrs["UNITS"] = value
        result = mms.reduce(vdf, np.eye(3), "1d", "cart")
        self.assertIsInstance(result, xr.DataArray)

    @data(
        (False, "ions", np.eye(3), "1d", "cart"),
        (False, "electrons", np.eye(3), "1d", "cart"),
        (True, "ions", np.eye(3), "1d", "cart"),
        (True, "electrons", np.eye(3), "1d", "cart"),
        (False, "electrons", generate_ts(64.0, 42, "tensor"), "1d", "cart"),
        (False, "ions", np.eye(3), "1d", "pol"),
        # (False, "ions", np.eye(3), "2d", "pol"), mc_pol_2d NotImplementedError
    )
    @unpack
    def test_reduce_output(self, energy01, species, xyz, dim, base):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01, species)
        result = mms.reduce(vdf, xyz, dim, base)
        self.assertIsInstance(result, xr.DataArray)

    @data(
        ("1d", "cart", {"vg": np.linspace(-1, 1, 42)}),
        ("1d", "cart", {"lower_e_lim": generate_ts(64.0, 42)}),
        ("1d", "cart", {"vg_edges": np.linspace(-1.01, 1.01, 102)}),
    )
    @unpack
    def test_reduce_options(self, dim, base, options):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=False, species="ions")
        xyz = np.eye(3)
        result = mms.reduce(vdf, xyz, dim, base, **options)
        self.assertIsInstance(result, xr.DataArray)

    @data(
        ("ions", "s^3/m^6", np.array([1, 0, 0]), "1d", "pol", {}),
        ("I AM GROOT", "s^3/m^6", np.eye(3), "1d", "pol", {}),
        ("ions", "bazinga", np.eye(3), "1d", "pol", {}),
        ("ions", "s^3/m^6", np.eye(3), "2d", "pol", {}),
        ("ions", "s^3/m^6", np.eye(3), "1d", "pol", {"lower_e_lim": generate_data(42)}),
    )
    @unpack
    def test_reduce_nonononono(self, species, units, xyz, dim, base, options):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species=species)
        vdf.data.attrs["UNITS"] = units
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            mms.reduce(vdf, xyz, dim, base, **options)


@ddt
class PsdRebinTestCase(unittest.TestCase):
    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_output(self, vdf):
        result = mms.psd_rebin(
            vdf,
            vdf.phi,
            vdf.attrs["energy0"],
            vdf.attrs["energy1"],
            vdf.attrs["esteptable"],
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(len(result[0]), 50)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertListEqual(list(result[1].shape), [50, 64, 32, 16])
        self.assertIsInstance(result[2], np.ndarray)
        self.assertEqual(len(result[2]), 64)
        self.assertIsInstance(result[3], np.ndarray)
        self.assertListEqual(list(result[3].shape), [50, 32])


@ddt
class SpectrToDatasetTestCase(unittest.TestCase):
    @data(generate_spectr(64.0, 100, 10))
    def test_spectr_to_dataset_output(self, spectr):
        result = mms.spectr_to_dataset(spectr)
        self.assertIsInstance(result, xr.Dataset)
