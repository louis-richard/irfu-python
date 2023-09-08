#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os
import random
import string

# Built-in imports
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, idata, unpack

from pyrfu import mms, pyrf
from pyrfu.mms.psd_moments import _moms

from . import (
    generate_data,
    generate_spectr,
    generate_timeline,
    generate_ts,
    generate_vdf,
)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def generate_feeps(f_s, n_pts, data_rate, dtype, lev, mms_id):
    var = {"tmmode": data_rate, "dtype": dtype, "lev": lev}
    tint = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]
    eyes = mms.feeps_active_eyes(var, tint, mms_id)
    keys = [f"{k}-{eyes[k][i]}" for k in eyes for i in range(len(eyes[k]))]
    feeps_dict = {k: generate_spectr(f_s, n_pts, 16, f"energy-{k}") for k in keys}
    feeps_dict["spinsectnum"] = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts), np.tile(np.arange(12), n_pts // 12 + 1)[:n_pts]
    )

    feeps_alle = xr.Dataset(feeps_dict)
    feeps_alle.attrs = {"mmsId": mms_id, **var}

    return feeps_alle


def generate_eis(f_s, n_pts, data_rate, dtype, lev, specie, data_unit, mms_id):
    pref = f"mms{mms_id:d}_epd_eis"
    pref = f"{pref}_{data_rate}_{lev}_{dtype}"

    if data_rate == "brst":
        pref = f"{pref}_{data_rate}_{dtype}"
    else:
        pref = f"{pref}_{dtype}"

    suf = f"{specie}_P1_{data_unit.lower()}_t"

    keys = [f"{pref}_{suf}{t:d}" for t in range(6)]

    spin_nums = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts),
        np.sort(np.tile(np.arange(n_pts // 12 + 1), (12,)))[1 : n_pts + 1],
    )
    sectors = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts),
        np.tile(np.arange(12), n_pts // 12 + 1)[1 : n_pts + 1],
    )

    if dtype.lower() == "extof":
        energies = np.array(
            [
                47.645324,
                54.928681,
                62.419454,
                70.833554,
                80.315371,
                91.00098,
                103.018894,
                116.554129,
                131.801143,
                148.970297,
                168.295534,
                190.060874,
                214.590996,
                242.245343,
                273.466432,
                308.768669,
                348.73539,
                394.035378,
                445.404668,
                503.597543,
                569.429005,
                643.764143,
                727.683404,
                822.660211,
                930.654627,
            ]
        )
    else:
        energies = np.array(
            [
                10.51516,
                11.509144,
                12.612351,
                13.817409,
                15.111664,
                16.55435,
                18.134081,
                19.857029,
                21.774935,
                23.807037,
                26.021971,
                28.526016,
                31.215776,
                34.228877,
                37.604494,
                41.116729,
                45.29041,
                51.412368,
                58.570702,
                65.951929,
                75.09237,
            ]
        )

    eis_dict = {"spin": spin_nums, "sector": sectors}

    for i, k in enumerate(keys):
        eis_dict[f"t{i:d}"] = generate_spectr(f_s, n_pts, len(energies), "energy")
        eis_dict[f"look_t{i:d}"] = generate_ts(f_s, n_pts, "vector")

    # glob_attrs = {**outdict["spin"].attrs["GLOBAL"], **var}
    glob_attrs = {
        "delta_energy_plus": 0.5 * np.ones(len(energies)),
        "delta_energy_minus": 0.5 * np.ones(len(energies)),
        "species": specie,
        "randattrs": "".join(random.choice(string.ascii_lowercase) for _ in range(10)),
    }

    # Build Dataset
    eis = xr.Dataset(eis_dict, attrs=glob_attrs)
    eis = eis.assign_coords(energy=energies)

    return eis


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


class DbInitTestCase(unittest.TestCase):
    def test_db_init_inpput(self):
        with self.assertRaises(AssertionError):
            mms.db_init("bazinga!")

    def test_db_init_output(self):
        self.assertIsNone(mms.db_init(os.getcwd()))


@ddt
class Def2PsdTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_def2psd_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.def2psd(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["keV/(cm^2 s sr keV)", "eV/(cm^2 s sr eV)", "1/(cm^2 s sr)"],
        )
    )
    @unpack
    def test_def2psd_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.def2psd(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.def2psd(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dpf2PsdTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_dpf2psd_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.dpf2psd(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["1/(cm^2 s sr keV)", "1/(cm^2 s sr eV)"],
        )
    )
    @unpack
    def test_dpf2psd_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.dpf2psd(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.dpf2psd(spectr)
        self.assertIsInstance(result, xr.DataArray)


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
class EisCombineProtonPadTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_combine_proton_pad_output(self, tmmode, specie, unit):
        mms_id = random.randint(1, 5)
        phxtof_allt = generate_eis(
            64.0, 100, tmmode, "phxtof", "l2", specie, unit, mms_id
        )
        extof_allt = generate_eis(
            64.0, 100, tmmode, "extof", "l2", specie, unit, mms_id
        )
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.DataArray)

    @idata(itertools.product([99, 100], repeat=2))
    @unpack
    def test_eis_combine_proton_pad_input(self, n_phxtof, n_extof):
        phxtof_allt = generate_eis(
            64.0, n_phxtof, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(
            64.0, n_extof, "brst", "extof", "l2", "proton", "flux", 1
        )
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.DataArray)

    @data(None, [1, 0, 0], generate_ts(64.0, 10, "vector"))
    def test_eis_combine_proton_pad_vec(self, vec):
        phxtof_allt = generate_eis(
            64.0, 100, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, vec)
        self.assertIsInstance(result, xr.DataArray)

    def test_eis_combine_proton_pad_options(self):
        phxtof_allt = generate_eis(
            64.0, 100, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, None, despin=True)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisCombineProtonSpecTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_combine_proton_spec_output(self, tmmode, specie, unit):
        mms_id = random.randint(1, 5)
        phxtof_allt = generate_eis(
            64.0, 100, tmmode, "phxtof", "l2", specie, unit, mms_id
        )
        extof_allt = generate_eis(
            64.0, 100, tmmode, "extof", "l2", specie, unit, mms_id
        )
        result = mms.eis_combine_proton_spec(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.Dataset)

    @idata(itertools.product([99, 100], repeat=2))
    @unpack
    def test_eis_combine_proton_spec_ctimes(self, n_phxtof, n_extof):
        phxtof_allt = generate_eis(
            64.0, n_phxtof, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(
            64.0, n_extof, "brst", "extof", "l2", "proton", "flux", 1
        )
        result = mms.eis_combine_proton_spec(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class EisOmniTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_omni_output(self, tmmode, dtype, specie, unit):
        eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_omni(eis, "mean")
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisPadTestCase(unittest.TestCase):
    @data(None, [1, 0, 0], generate_ts(64.0, 10, "vector"))
    def test_eis_pad_output(self, vec):
        eis = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_pad(eis, vec)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisPadSpinAvgTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_pad_spin_avg_output(self, tmmode, dtype, specie, unit):
        eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        eis_pad = mms.eis_pad(eis)
        result = mms.eis_pad_spinavg(eis_pad, eis.spin)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisProtonCorrectionTestCase(unittest.TestCase):
    def test_eis_proton_correction_dataarray(self):
        flux_eis = generate_spectr(64, 100, 16, "energy")
        result = mms.eis_proton_correction(flux_eis)
        self.assertIsInstance(result, xr.DataArray)

    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_proton_correction_dataset(self, tmmode, dtype, specie, unit):
        flux_eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_proton_correction(flux_eis)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class EisSpinAvgTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["mean", "sum"],
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_spin_avg_output(self, method, tmmode, dtype, specie, unit):
        eis_allt = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_spin_avg(eis_allt, method)
        self.assertIsInstance(result, xr.Dataset)


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


class HpcaEnergiesTestCase(unittest.TestCase):
    def test_hpca_energies_output(self):
        result = mms.hpca_energies()
        self.assertIsInstance(result, list)


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
class Psd2DefTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_psd2def_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.psd2def(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["s^3/cm^6", "s^3/m^6", "s^3/km^6"],
        )
    )
    @unpack
    def test_psd2def_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.psd2def(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.psd2def(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Psd2DpfTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_psd2dpf_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.psd2dpf(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["s^3/cm^6", "s^3/m^6", "s^3/km^6"],
        )
    )
    @unpack
    def test_psd2dpf_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.psd2dpf(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.psd2dpf(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class PsdMomentsTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 32, 16)), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), species="electrons"), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=False), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=False), "fast"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=True), "brst"),
    )
    @unpack
    def test_psd_moments_input(self, vdf, data_rate):
        delta_theta = 0.5 * np.ones(vdf.data.shape[3])
        vdf.attrs["delta_theta_minus"] = delta_theta
        vdf.attrs["delta_theta_plus"] = delta_theta

        delta_phi = 0.5 * np.ones((vdf.data.shape[0], vdf.data.shape[2]))
        vdf.attrs["delta_phi_minus"] = delta_phi
        vdf.attrs["delta_phi_plus"] = delta_phi
        vdf.data.attrs["FIELDNAM"] = f"MMS1 FPI/DIS {data_rate}SkyMap dist"
        mms.psd_moments(vdf, generate_ts(64.0, 100, "scalar"))

    @data({"energy_range": [1, 1000]}, {"no_sc_pot": True})
    def test_psd_moments_options(self, options):
        vdf = generate_vdf(64.0, 100, (32, 32, 16))
        vdf.data.attrs["FIELDNAM"] = "MMS1 FPI/DIS brstSkyMap dist"
        mms.psd_moments(vdf, generate_ts(64.0, 100, "scalar"), **options)

    @data(
        (
            np.random.random((100, 32)),  # energy
            np.random.random((10000, 32)),  # delta_v
            random.random(),  # q_e
            np.random.random(100),  # sc_pot
            random.random(),  # p_mass
            random.choice([True, False]),  # flag_inner_electron
            random.random(),  # w_inner_electron
            np.random.random((100, 32, 16)),  # phi
            np.random.random((100, 32, 16)),  # theta
            np.arange(32),  # int_energies
            np.random.random((100, 32, 32, 16)),  # vdf
            np.random.random((100, 32, 16)),  # delta_ang
        )
    )
    def test_moms(self, value):
        result = _moms.__wrapped__(*value)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], np.ndarray)


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
class FeepsActiveEyesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"], ["sitl", "l2"]))
    @unpack
    def test_feeps_active_eyes_output(self, data_rate, dtype, lev):
        tint = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]
        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            tint,
            random.randint(1, 4),
        )
        self.assertIsInstance(result, dict)

        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            pyrf.iso86012datetime64(np.array(tint)),
            str(random.randint(1, 4)),
        )
        self.assertIsInstance(result, dict)


@ddt
class FeepsCorrectEnergiesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_correct_energies_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )

        result = mms.feeps_correct_energies(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsFlatFieldCorrectionsTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_flat_field_corrections_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )

        result = mms.feeps_flat_field_corrections(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsOmniTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_omni_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_omni(feeps_alle)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPadTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_ouput(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, "vector"))
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPadSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        feeps_pad = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, "vector"))
        result = mms.feeps_pad_spinavg(feeps_pad, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPitchAnglesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pitch_angles_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pitch_angles(feeps_alle, generate_ts(64.0, 100, "vector"))
        self.assertIsInstance(result[0], xr.DataArray)


@ddt
class FeepsRemoveBadDataTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_bad_data_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )

        result = mms.feeps_remove_bad_data(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsRemoveSunTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_sun(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_remove_sun(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_omni = mms.feeps_omni(feeps_alle)

        result = mms.feeps_spin_avg(feeps_omni, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsSplitIntegralChTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_split_integral_ch(self, data_rate, dtype):
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", random.randint(1, 4)
        )
        mms.feeps_split_integral_ch(feeps_alle)


@ddt
class FkPowerSpectrum4scTestCase(unittest.TestCase):
    @data((None, None), (random.random(), None), (None, [0.1, 1]))
    @unpack
    def test_fk_power_spectrum_4sc(self, df, f_range):
        e_mms = [generate_ts(64.0, 100, "scalar") for _ in range(4)]
        r_mms = [generate_ts(64.0, 100, "vector") for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, "vector") for _ in range(4)]
        tint = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]

        result = mms.fk_power_spectrum_4sc(
            e_mms, r_mms, b_mms, tint, df=df, f_range=f_range
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
    def test_reduce_input(self, species, units, xyz, dim, base, options):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species=species)
        vdf.data.attrs["UNITS"] = units
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            mms.reduce(vdf, xyz, dim, base, **options)


@ddt
class RotateTensorTestCase(unittest.TestCase):
    @data(("rot", generate_data(100, "vector")), ("gse", None))
    @unpack
    def test_rotate_tensor_input(self, flag, vec):
        with self.assertRaises((TypeError, NotImplementedError)):
            mms.rotate_tensor(generate_ts(64.0, 100, "tensor"), flag, vec)

    @data(
        ("fac", generate_ts(64.0, 100, "vector"), "pp"),
        ("fac", generate_ts(64.0, 100, "vector"), "qq"),
        ("rot", np.random.random(3), "pp"),
        ("rot", np.random.random((3, 3)), "pp"),
    )
    @unpack
    def test_rotate_tensor_output(self, rot_flag, vec, perp):
        result = mms.rotate_tensor(
            generate_ts(64.0, 100, "tensor"), rot_flag, vec, perp
        )
        self.assertIsInstance(result, xr.DataArray)


@ddt
class SpectrToDatasetTestCase(unittest.TestCase):
    @data(generate_spectr(64.0, 100, 10))
    def test_spectr_to_dataset_output(self, spectr):
        result = mms.spectr_to_dataset(spectr)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class Scpot2NeTestCase(unittest.TestCase):
    @data(None, generate_ts(64.0, 100, "scalar"))
    def test_scpot2ne_output(self, i_aspoc):
        result = mms.scpot2ne(
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "scalar"),
            generate_ts(64.0, 100, "tensor"),
            i_aspoc,
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)


@ddt
class TokenizeTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            mms.tokenize.__globals__["PARAMS_SCALARS"],
            mms.tokenize.__globals__["INSTRUMENTS"],
            mms.tokenize.__globals__["SAMPLING_RATES"],
            mms.tokenize.__globals__["DATA_LVLS"],
        )
    )
    def test_tokenize_scalar(self, var):
        result = mms.tokenize("_".join(var))
        self.assertIsInstance(result, dict)
        self.assertListEqual(
            list(result.keys()), ["param", "to", "cs", "inst", "tmmode", "lev"]
        )
        self.assertEqual(result["to"], 0)

    @idata(
        itertools.product(
            mms.tokenize.__globals__["PARAMS_VECTORS"],
            mms.tokenize.__globals__["COORDINATE_SYSTEMS"],
            mms.tokenize.__globals__["INSTRUMENTS"],
            mms.tokenize.__globals__["SAMPLING_RATES"],
            mms.tokenize.__globals__["DATA_LVLS"],
        )
    )
    def test_tokenize_vector(self, var):
        result = mms.tokenize("_".join(var))
        self.assertIsInstance(result, dict)
        self.assertListEqual(
            list(result.keys()), ["param", "to", "cs", "inst", "tmmode", "lev"]
        )
        self.assertEqual(result["to"], 1)

    @idata(
        itertools.product(
            mms.tokenize.__globals__["PARAMS_TENSORS"],
            mms.tokenize.__globals__["COORDINATE_SYSTEMS"],
            mms.tokenize.__globals__["INSTRUMENTS"],
            mms.tokenize.__globals__["SAMPLING_RATES"],
            mms.tokenize.__globals__["DATA_LVLS"],
        )
    )
    def test_tokenize_tensor(self, var):
        result = mms.tokenize("_".join(var))
        self.assertIsInstance(result, dict)
        self.assertListEqual(
            list(result.keys()), ["param", "to", "cs", "inst", "tmmode", "lev"]
        )


@ddt
class VdfElimTestCase(unittest.TestCase):
    @data(
        random.randint(0, 15),
        random.randint(0, 15) + 0.4,
        [random.randint(0, 15), random.randint(16, 31)],
    )
    def test_vdf_elim_output(self, e_int):
        result = mms.vdf_elim(
            generate_vdf(64.0, 42, [32, 32, 16], energy01=True), e_int
        )
        self.assertIsInstance(result, xr.Dataset)


@ddt
class VdfOmniTestCase(unittest.TestCase):
    @data("mean", "sum")
    def test_vdf_omni_output(self, method):
        result = mms.vdf_omni(generate_vdf(64.0, 100, (32, 32, 16)), method)
        self.assertIsInstance(result, xr.DataArray)


if __name__ == "__main__":
    unittest.main()
