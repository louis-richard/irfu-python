#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib

# Built-in imports
import datetime
import importlib
import io
import itertools
import json
import os
import random
import re
import string
import sys
import tempfile
import types
import unittest
import urllib.parse
import warnings
from contextlib import nullcontext
from unittest import mock

# 3rd party imports
import keyring
import keyring.credentials
import keyring.errors
import numpy as np
import pycdfpp
import pytest
import requests
import xarray as xr
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from ddt import data, ddt, idata, unpack
from scipy import constants
from scipy.special import gammainc

# Local imports
from .. import mms, pyrf
from ..mms.feeps_flat_field_corrections import g_corr
from ..mms.get_ts import get_ts
from ..mms.psd_moments import _moms
from . import (
    generate_data,
    generate_defatt,
    generate_spectr,
    generate_timeline,
    generate_ts,
    generate_vdf,
)

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Module (pyrfu.mms.list_files_aws is also the name of the function)
list_files_aws_module = importlib.import_module("..mms.list_files_aws", __package__)

TEST_TINT = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]

data_units_keys = {
    "flux": "1/(cm^2 s sr keV)",
    "counts": "counts",
    "cps": "1/s",
    "mask": "sector_mask",
}


def generate_feeps(f_s, n_pts, data_rate, dtype, lev, units_name, mms_id):

    units_key = data_units_keys[units_name.lower()]

    var = {
        "tmmode": data_rate,
        "dtype": dtype,
        "lev": lev,
        "units_name": units_name,
        "species": dtype[0],
        "mmsId": mms_id,
    }

    eyes = mms.feeps_active_eyes(var, TEST_TINT, mms_id)
    keys = [f"{k}-{eyes[k][i]}" for k in eyes for i in range(len(eyes[k]))]
    feeps_dict = {
        k: generate_spectr(f_s, n_pts, 16, dict(sensor=f"energy-{k}", UNITS=units_key))
        for k in keys
    }

    feeps_dict["spinsectnum"] = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts), np.tile(np.arange(12), n_pts // 12 + 1)[:n_pts]
    )

    feeps_alle = xr.Dataset(feeps_dict)
    feeps_alle.attrs = {**var}

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
        eis_dict[f"look_t{i:d}"] = generate_ts(f_s, n_pts, tensor_order=1)

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


def _mms_keys():
    test_path = os.path.dirname(os.path.abspath(__file__))

    root_path = os.path.join(os.path.split(test_path)[0], "mms")

    with open(
        os.sep.join([root_path, "mms_keys.json"]), "r", encoding="utf-8"
    ) as json_file:
        keys_ = json.load(json_file)

    all_keys = list(
        np.hstack([list(instrument.keys()) for instrument in keys_.values()])
    )
    return all_keys


@ddt
class CalcEpsilonTestCase(unittest.TestCase):
    @data(
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_ts(32.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
        ),
    )
    @unpack
    def test_calc_epsilon_input(self, vdf, model_vdf, n_s, sc_pot):
        with self.assertRaises(ValueError):
            mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot)

    @data(
        (
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/cm^6",
            ),
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/cm^6",
            ),
            {},
        ),
        (
            generate_vdf(
                64.0, 100, (32, 16, 16), energy01=False, species="ions", units="s^3/m^6"
            ),
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/km^6",
            ),
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
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            **kwargs,
        )

    @staticmethod
    def _maxwellian(widths="attrs", alternating=False):
        # FPI-like skymap of a drifting proton Maxwellian (n = 1 cm^-3)
        energy1 = None
        if alternating:
            energy1 = 10.0 * 3000.0 ** ((np.arange(32) + 0.5) / 31)

        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(
            1.0, 1000.0, [200.0, 0.0, 0.0], n_t=4, energy1=energy1
        )
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        if widths == "missing":
            del vdf.attrs["delta_energy_minus"], vdf.attrs["delta_energy_plus"]
        elif widths == "None":
            # get_dist sets missing widths to None
            vdf.attrs["delta_energy_minus"] = vdf.attrs["delta_energy_plus"] = None

        n_s = pyrf.ts_scalar(vdf.time.data, np.ones(4))
        return vdf, n_s

    @data(
        ("attrs", False),
        ("missing", False),
        ("None", False),
        ("missing", True),
    )
    @unpack
    def test_calc_epsilon_values(self, widths, alternating):
        # epsilon = int |f - f_model| d3v / (2 n): 0 for identical distributions
        # and 1/2 for a zero model. A single energy table without energy widths
        # (IndexError) and widths set to None (TypeError) used to crash.
        vdf, n_s = self._maxwellian(widths, alternating)
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(4))

        eps = mms.calculate_epsilon(vdf, vdf.copy(deep=True), n_s, sc_pot)
        np.testing.assert_allclose(eps.data, 0.0, atol=1e-12)

        model = vdf.copy(deep=True)
        model.data.data[...] = 0.0
        eps = mms.calculate_epsilon(vdf, model, n_s, sc_pot)
        np.testing.assert_allclose(eps.data, 0.5, rtol=0.01)

    def test_calc_epsilon_phi_1d(self):
        # phi as (phi,) (IndexError): as with phi as (time, phi)
        vdf, n_s = self._maxwellian()
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(4))
        model = vdf.copy(deep=True)
        model.data.data[...] = 0.0
        expected = mms.calculate_epsilon(vdf, model, n_s, sc_pot)

        vdf["phi"] = ("idx1", vdf.phi.data[0])
        eps = mms.calculate_epsilon(vdf, model, n_s, sc_pot)
        np.testing.assert_allclose(eps.data, expected.data, rtol=1e-12)

    def test_calc_epsilon_straddling_sc_pot(self):
        # Channels whose lower edge is below the spacecraft potential used to be
        # dropped (sqrt of a negative energy is NaN); compare with a direct sum
        vdf, n_s = self._maxwellian()
        v_sc = -25.0  # ions: the corrected energy is E + V = E - 25 eV
        model = vdf.copy(deep=True)
        model.data.data[...] = 0.0
        eps = mms.calculate_epsilon(
            vdf, model, n_s, pyrf.ts_scalar(vdf.time.data, np.full(4, v_sc))
        )

        energy = vdf.energy.data[0]
        e_minus = vdf.attrs["delta_energy_minus"][0]
        e_plus = vdf.attrs["delta_energy_plus"][0]
        self.assertTrue(np.any((energy - e_minus < 25.0) & (energy + e_plus > 25.0)))

        def speed(e_kin):
            e_corr = np.clip(e_kin - 25.0, 0.0, None)
            return np.sqrt(2 * constants.elementary_charge * e_corr / constants.m_p)

        w_v = speed(energy) ** 2 * (speed(energy + e_plus) - speed(energy - e_minus))
        w_ang = np.sin(np.deg2rad(vdf.theta.data)) * np.deg2rad(11.25) ** 2
        f_si = vdf.data.data[0] * 1e12
        expected = np.sum(f_si * w_v[:, None, None] * w_ang[None, None, :]) / 2e6
        np.testing.assert_allclose(eps.data, expected, rtol=1e-10)


class CorrectEdpProbeTimingTestCase(unittest.TestCase):
    def test_correct_edp_probe_timing_values(self):
        # Probe potentials linear in time, sampled as in the EDP processing:
        # V3 and V5 7.629 and 15.259 us late, V2, V4, V6 reconstructed from
        # E12, E34, E56 sampled 26.703, 30.518 and 34.332 us late
        time = generate_timeline(8192.0, 100)
        t_sec = (time - time[0]) / np.timedelta64(1, "s")
        offset = np.array([1.0, 1.5, -2.0, 0.5, 3.0, 2.5])
        rate = np.array([1e3, -2e3, 3e3, 1e3, -1e3, 4e3])

        def pot(i, delay=0.0):
            return offset[i] + rate[i] * (t_sec + delay)

        v_odd = [pot(0), pot(2, 7.629e-6), pot(4, 15.259e-6)]
        v_l2 = []
        for i, (delay, length) in enumerate(
            zip([26.703e-6, 30.518e-6, 34.332e-6], [0.120, 0.120, 0.0292])
        ):
            e_meas = (pot(2 * i, delay) - pot(2 * i + 1, delay)) / length
            v_l2 += [v_odd[i], v_odd[i] - e_meas * length]

        sc_pot = xr.DataArray(
            np.column_stack(v_l2),
            coords=[time, np.arange(1, 7)],
            dims=["time", "probe"],
            attrs={"UNITS": "V"},
        )
        sc_pot_data = sc_pot.data.copy()

        result = mms.correct_edp_probe_timing(sc_pot)

        expected = np.column_stack([pot(i) for i in range(6)])
        np.testing.assert_allclose(result.data, expected, atol=1e-9)
        self.assertTupleEqual(result.dims, ("time", "probe"))
        np.testing.assert_array_equal(result.probe.data, np.arange(1, 7))
        self.assertEqual(result.attrs["UNITS"], "V")
        np.testing.assert_array_equal(sc_pot.data, sc_pot_data)


class ConfigCacheTestCase(unittest.TestCase):
    # The configuration and the SDC session are cached until the configuration
    # file changes (e.g., mms.db_init)

    @staticmethod
    def _write_config(path, default, rights="public", mtime_ns=None):
        config = {
            "default": default,
            "local": ".",
            "sdc": {"rights": rights, "username": "u"},
            "aws": "",
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file)
        if mtime_ns is not None:
            os.utime(path, ns=(mtime_ns, mtime_ns))

    def test_load_config_refresh(self):
        module = importlib.import_module("pyrfu.mms.db_get_ts")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.json")
            self._write_config(path, "local", mtime_ns=10**18)
            with mock.patch.object(module, "MMS_CFG_PATH", path):
                module._load_config_cached.cache_clear()
                config = module._load_config()
                config["default"] = "changed by the caller"
                self.assertEqual(module._load_config()["default"], "local")

                self._write_config(path, "aws", mtime_ns=10**18 + 10**9)
                self.assertEqual(module._load_config()["default"], "aws")
                module._load_config_cached.cache_clear()

    def test_load_config_other_path(self):
        # Another configuration file with the same modification time is read
        # (the cache was keyed by the time only)
        module = importlib.import_module("pyrfu.mms.db_get_ts")
        with tempfile.TemporaryDirectory() as root:
            module._load_config_cached.cache_clear()
            for default in ["local", "aws"]:
                path = os.path.join(root, f"config_{default}.json")
                self._write_config(path, default, mtime_ns=10**18)
                with mock.patch.object(module, "MMS_CFG_PATH", path):
                    self.assertEqual(module._load_config()["default"], default)
            module._load_config_cached.cache_clear()

    def test_login_lasp_refresh(self):
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.json")
            self._write_config(path, "sdc", mtime_ns=10**18)
            with (
                mock.patch.object(module, "MMS_CFG_PATH", path),
                mock.patch.object(module.requests, "Session") as session_cls,
                mock.patch.object(module, "_get_credential", return_value=None),
            ):
                module._login_lasp_cached.cache_clear()
                first = module._login_lasp()
                self.assertIs(module._login_lasp()[0], first[0])
                self.assertEqual(session_cls.call_count, 1)

                self._write_config(path, "sdc", mtime_ns=10**18 + 10**9)
                module._login_lasp()
                self.assertEqual(session_cls.call_count, 2)
                module._login_lasp_cached.cache_clear()


class LoginLaspCredentialsTestCase(unittest.TestCase):
    # The SDC credentials are only read and sent for the SITL access

    def _login(self, rights, credential):
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.json")
            ConfigCacheTestCase._write_config(path, "sdc", rights=rights)
            with (
                mock.patch.object(module, "MMS_CFG_PATH", path),
                mock.patch.object(module.requests, "Session") as session_cls,
                mock.patch.object(
                    module, "_get_credential", return_value=credential
                ) as get_credential,
            ):
                module._login_lasp_cached.cache_clear()
                try:
                    _, _, lasp_url = module._login_lasp()
                finally:
                    module._login_lasp_cached.cache_clear()

        return session_cls.return_value, get_credential, lasp_url

    def test_login_lasp_public(self):
        # The SITL password was sent with every public request
        credential = keyring.credentials.SimpleCredential("u", "secret")
        session, get_credential, lasp_url = self._login("public", credential)
        get_credential.assert_not_called()
        self.assertNotIsInstance(session.auth, tuple)
        self.assertIn("/public/", lasp_url)

    def test_login_lasp_sitl(self):
        credential = keyring.credentials.SimpleCredential("u", "secret")
        session, get_credential, lasp_url = self._login("sitl", credential)
        get_credential.assert_called_once_with("u")
        self.assertTupleEqual(session.auth, ("u", "secret"))
        self.assertIn("/sitl/", lasp_url)

        with self.assertRaises(EnvironmentError):
            self._login("sitl", None)


class _MemoryKeyring:
    # In-memory keyring backend with a given priority (>= 1: system keyring)
    def __init__(self, priority):
        self.priority = priority
        self.passwords = {}
        self.file_path = "/nowhere/keyring_pass.cfg"

    def get_credential(self, service, username):
        password = self.passwords.get((service, username))
        return (
            keyring.credentials.SimpleCredential(username, password)
            if password
            else None
        )

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        if self.passwords.pop((service, username), None) is None:
            raise keyring.errors.PasswordDeleteError("not found")


class DbInitCredentialsTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_init")
        self.plaintext = _MemoryKeyring(0.5)

    def _db_init(self, system, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.object(
                    self.module, "MMS_CFG_PATH", os.path.join(root, "c.json")
                ),
                mock.patch.object(
                    self.module.keyring, "get_keyring", return_value=system
                ),
                mock.patch.object(
                    self.module, "PlaintextKeyring", return_value=self.plaintext
                ),
                mock.patch.object(self.module.keyring, "set_keyring") as set_keyring,
            ):
                self.module.db_init(local=root, sdc="sitl", **kwargs)
                credential = self.module._get_credential(kwargs["sdc_username"])

        set_keyring.assert_not_called()  # the process keyring is not changed
        return credential

    def test_db_init_system_keyring(self):
        # Saved in the system keyring, and the old plaintext copy is removed
        system = _MemoryKeyring(5)
        self.plaintext.set_password("mms-sdc", "louis", "old")

        credential = self._db_init(system, sdc_username="louis", sdc_password="new")

        self.assertEqual(credential.password, "new")
        self.assertDictEqual(system.passwords, {("mms-sdc", "louis"): "new"})
        self.assertDictEqual(self.plaintext.passwords, {})

    def test_db_init_no_system_keyring(self):
        # Without system keyring, saved in plain text with a warning
        system = _MemoryKeyring(0)
        with self.assertLogs("pyrfu", level="WARNING"):
            credential = self._db_init(system, sdc_username="louis", sdc_password="pw")

        self.assertEqual(credential.password, "pw")
        self.assertDictEqual(system.passwords, {})
        self.assertDictEqual(self.plaintext.passwords, {("mms-sdc", "louis"): "pw"})

    def _patched(self, root, system):
        # Temporary configuration file and in-memory keyrings
        return [
            mock.patch.object(
                self.module, "MMS_CFG_PATH", os.path.join(root, "c.json")
            ),
            mock.patch.object(self.module.keyring, "get_keyring", return_value=system),
            mock.patch.object(
                self.module, "PlaintextKeyring", return_value=self.plaintext
            ),
        ]

    def _config(self, root):
        with open(os.path.join(root, "c.json"), encoding="utf-8") as file:
            return json.load(file)

    def test_db_init_keeps_other_settings(self):
        # Only the given settings change; the others used to be reset (rights
        # to public, username to "username", local to ".", aws to "")
        system = _MemoryKeyring(5)
        system.set_password("mms-sdc", "louis", "secret")
        with tempfile.TemporaryDirectory() as root:
            other = os.path.join(root, "other")
            os.makedirs(other)
            with contextlib.ExitStack() as stack:
                for patch in self._patched(root, system):
                    stack.enter_context(patch)
                self.module.db_init("aws", root, "sitl", "louis", aws="b/p")
                self.module.db_init(local=other)
                self.module.db_init(default="local")
                config = self._config(root)

        self.assertDictEqual(
            config,
            {
                "default": "local",
                "local": other,
                "sdc": {"rights": "sitl", "username": "louis"},
                "aws": "b/p",
            },
        )

    def test_db_init_keyring_untouched_without_password(self):
        # No placeholder is stored (used to store "username"/"password"), and
        # an existing credential is kept
        system = _MemoryKeyring(5)
        system.set_password("mms-sdc", "louis", "secret")
        with tempfile.TemporaryDirectory() as root:
            with contextlib.ExitStack() as stack:
                for patch in self._patched(root, system):
                    stack.enter_context(patch)
                self.module.db_init(local=root)
                self.module.db_init(sdc_username="louis")
                self.assertDictEqual(system.passwords, {("mms-sdc", "louis"): "secret"})
                self.assertDictEqual(self.plaintext.passwords, {})

                # SITL rights without credentials: a warning
                with self.assertLogs("pyrfu", level="WARNING"):
                    self.module.db_init(sdc="sitl", sdc_username="nobody")
                with self.assertRaises(ValueError):
                    self.module.db_init(sdc_password="pw")

        self.assertDictEqual(system.passwords, {("mms-sdc", "louis"): "secret"})
        self.assertDictEqual(self.plaintext.passwords, {})

    def test_db_init_reset(self):
        system = _MemoryKeyring(5)
        system.set_password("mms-sdc", "louis", "secret")
        with tempfile.TemporaryDirectory() as root:
            with contextlib.ExitStack() as stack:
                for patch in self._patched(root, system):
                    stack.enter_context(patch)
                self.module.db_init("aws", root, "sitl", "louis", aws="b/p")
                self.module.db_init(local=root, reset=True)
                config = self._config(root)

        self.assertDictEqual(
            config,
            {
                "default": "local",
                "local": root,
                "sdc": {"rights": "public", "username": "username"},
                "aws": "",
            },
        )
        self.assertDictEqual(system.passwords, {("mms-sdc", "louis"): "secret"})

    def test_get_credential_previous_versions(self):
        # Credentials saved in plain text by previous versions are still found
        self.plaintext.set_password("mms-sdc", "louis", "old")
        with (
            mock.patch.object(
                self.module.keyring, "get_keyring", return_value=_MemoryKeyring(5)
            ),
            mock.patch.object(
                self.module, "PlaintextKeyring", return_value=self.plaintext
            ),
        ):
            self.assertEqual(self.module._get_credential("louis").password, "old")


class MmsConfigPathTestCase(unittest.TestCase):
    # The MMS configuration lives in the user configuration directory, created
    # from the package configuration the first time
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_init")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.package = os.path.join(self.tmp_dir.name, "package_config.json")
        with open(self.package, "w", encoding="utf-8") as file:
            json.dump({"default": "aws"}, file)
        self.user_dir = os.path.join(self.tmp_dir.name, "user", "pyrfu")

        for patch in [
            mock.patch.object(self.module, "_PACKAGE_CFG_PATH", self.package),
            mock.patch.object(
                self.module.platformdirs, "user_config_dir", return_value=self.user_dir
            ),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def test_config_path_migration(self):
        # Created from the package configuration the first time, then kept
        path = self.module._user_config_path()
        self.assertEqual(path, os.path.join(self.user_dir, "mms_config.json"))
        with open(path, encoding="utf-8") as file:
            self.assertDictEqual(json.load(file), {"default": "aws"})

        with open(path, "w", encoding="utf-8") as file:
            json.dump({"default": "sdc"}, file)
        self.assertEqual(self.module._user_config_path(), path)
        with open(path, encoding="utf-8") as file:
            self.assertDictEqual(json.load(file), {"default": "sdc"})

    def test_config_path_not_writable(self):
        # The package configuration is used if the user directory is not writable
        with mock.patch.object(self.module.shutil, "copyfile", side_effect=OSError):
            self.assertEqual(self.module._user_config_path(), self.package)


class DbGetTsTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_get_ts")

    def _ts(self, i_file, value):
        # 3 samples per file, 1 s apart, starting 10 s after each other
        times = generate_timeline(
            1.0, 3, ref_time=f"2019-01-01T00:00:{10 * i_file:02d}"
        )
        return pyrf.ts_scalar(times, np.full(3, value, dtype=float))

    def test_db_get_ts_dict_one_read_per_file(self):
        # All the variables are read from each file once (one download from the
        # SDC or AWS), and appended in time
        session = mock.Mock()
        sources = (["file0", "file1"], session, {})
        names = ["var_a", "var_b", "var_c"]

        def get_ts(content, cdf_name, tint):
            return self._ts(int(content[-1:]), names.index(cdf_name))

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with mock.patch.object(
                self.module,
                "_get_file_content_sources",
                side_effect=lambda r, f, *_: f.encode(),
            ) as download:
                with mock.patch.object(self.module, "get_ts", side_effect=get_ts):
                    out = self.module._db_get_ts_dict(
                        "mms1_feeps_brst_l2_electron",
                        names,
                        ["a", "b"],
                        False,
                        "",
                        "SDC",
                    )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0].args[:2], ("sdc", "file0"))
        session.close.assert_not_called()  # shared, cached session

        for value, name in enumerate(names):
            self.assertEqual(len(out[name]), 6)
            np.testing.assert_array_equal(out[name].data, value)

    def test_db_get_ts_dict_support(self):
        # Non-record-varying support variables: the first record of the first
        # file that has them, without time; missing ones are left out
        sources = (["file0", "file1"], mock.Mock(), {})
        cdfs = {
            b"file0": {"var_s": types.SimpleNamespace(values=np.array([[1, 2, 3]]))},
            b"file1": {
                "var_s": types.SimpleNamespace(values=np.array([[9, 9, 9]])),
                "var_t": types.SimpleNamespace(values=np.array([[4, 5]])),
            },
        }

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with mock.patch.object(
                self.module,
                "_get_file_content_sources",
                side_effect=lambda r, f, *_: f.encode(),
            ):
                with mock.patch.object(self.module, "load", side_effect=cdfs.get):
                    with mock.patch.object(
                        self.module, "get_ts", return_value=self._ts(0, 1.0)
                    ):
                        out = self.module._db_get_ts_dict(
                            "mms1_feeps_brst_l2_electron",
                            ["var_a"],
                            ["a", "b"],
                            False,
                            "",
                            "sdc",
                            support_names=["var_s", "var_t", "var_u"],
                        )

        np.testing.assert_array_equal(out["var_s"], [1, 2, 3])
        np.testing.assert_array_equal(out["var_t"], [4, 5])
        self.assertNotIn("var_u", out)
        self.assertIsInstance(out["var_a"], xr.DataArray)

    def test_db_get_ts_no_file(self):
        session = mock.Mock()

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=([], session, {})
        ):
            with self.assertRaises(FileNotFoundError):
                mms.db_get_ts("mms1_fgm_srvy_l2", "b", ["a", "b"], source="sdc")

        session.close.assert_not_called()  # shared, cached session

    def test_db_get_ts_value_verbose(self):
        # The time series of the variable, the loading logged when verbose
        ts = self._ts(0, 1.0)
        with mock.patch.object(
            self.module, "_list_files_sources", return_value=(["file"], None, {})
        ):
            with mock.patch.object(
                self.module, "_get_file_content_sources", return_value=b"cdf"
            ):
                with mock.patch.object(self.module, "get_ts", return_value=ts):
                    with self.assertLogs("pyrfu", level="INFO") as logs:
                        out = mms.db_get_ts(
                            "mms1_fgm_srvy_l2", "b", ["a", "b"], verbose=True
                        )

        xr.testing.assert_identical(out, ts)
        self.assertTrue(any("Loading b" in line for line in logs.output))

    def test_db_get_ts_read_error(self):
        # A variable that cannot be read is logged with its file and re-raised
        with mock.patch.object(
            self.module, "_list_files_sources", return_value=(["file.cdf"], None, {})
        ):
            with mock.patch.object(
                self.module, "_get_file_content_sources", return_value=b"cdf"
            ):
                with mock.patch.object(
                    self.module, "get_ts", side_effect=KeyError("b")
                ):
                    with self.assertLogs("pyrfu", level="ERROR") as logs:
                        with self.assertRaises(KeyError):
                            mms.db_get_ts(
                                "mms1_fgm_srvy_l2", "b", ["a", "b"], verbose=False
                            )

        self.assertTrue(any("file.cdf" in line for line in logs.output))

    def test_resolve_source(self):
        with mock.patch.object(
            self.module, "_load_config", return_value={"default": "local"}
        ):
            self.assertEqual(self.module._resolve_source(""), "local")
            self.assertEqual(self.module._resolve_source(None), "local")
            self.assertEqual(self.module._resolve_source("Default"), "local")

        self.assertEqual(self.module._resolve_source("AWS"), "aws")

        with self.assertRaises(ValueError):
            self.module._resolve_source("cdaweb")

    def test_tokenize_no_dtype(self):
        mms_id, var = self.module._tokenize("mms1_fgm_srvy_l2")
        self.assertEqual(mms_id, "1")
        self.assertDictEqual(
            var, {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        )


class DbGetVariableTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_get_variable")

    def test_db_get_variable_source(self):
        # Read from the first file of the resource, the session is kept
        session = mock.Mock()
        sources = (["file0", "file1"], session, {})

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ) as list_files:
            with mock.patch.object(
                self.module, "_get_file_content_sources", return_value=b"cdf"
            ) as download:
                with mock.patch.object(self.module, "get_variable") as get_variable:
                    mms.db_get_variable(
                        "mms2_feeps_brst_l2_electron",
                        "sensor_ids",
                        ["a", "b"],
                        verbose=False,
                        source="aws",
                    )

        self.assertEqual(list_files.call_args.args[0], "aws")
        self.assertEqual(list_files.call_args.args[3]["dtype"], "electron")
        download.assert_called_once()
        self.assertEqual(download.call_args.args[:2], ("aws", "file0"))
        get_variable.assert_called_once_with(b"cdf", "sensor_ids")
        session.close.assert_not_called()  # shared, cached session

    def test_db_get_variable_no_file(self):
        session = mock.Mock()

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=([], session, {})
        ):
            with self.assertRaises(FileNotFoundError):
                mms.db_get_variable(
                    "mms1_fgm_srvy_l2", "label", ["a", "b"], source="sdc"
                )

        session.close.assert_not_called()  # shared, cached session

    def test_get_variable_ndim(self):
        # N-D variables (only 1-D variables were supported): dims x, y, ...
        module = importlib.import_module("pyrfu.mms.get_variable")
        variable = mock.Mock(values=np.arange(12).reshape(1, 12), attributes={})

        with mock.patch.object(module, "load", return_value={"ids": variable}):
            out = module.get_variable(b"cdf", "ids")

        self.assertTupleEqual(out.dims, ("x", "y"))
        np.testing.assert_array_equal(out.data, variable.values)


class DbInitTestCase(unittest.TestCase):
    # db_init writes to a temporary configuration file and in-memory keyrings,
    # never to the user's configuration or keyring
    def setUp(self):
        module = importlib.import_module("pyrfu.mms.db_init")
        self.tmp_dir = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(
                module, "MMS_CFG_PATH", os.path.join(self.tmp_dir.name, "c.json")
            ),
            mock.patch.object(
                module.keyring, "get_keyring", return_value=_MemoryKeyring(5)
            ),
            mock.patch.object(
                module, "PlaintextKeyring", return_value=_MemoryKeyring(0.5)
            ),
            mock.patch.object(module.keyring, "set_keyring"),
            mock.patch.object(module.keyring, "get_credential", return_value=None),
            mock.patch.object(module.keyring, "set_password"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp_dir.cleanup)

    def test_db_init_input(self):
        with self.assertRaises(NotImplementedError):
            mms.db_init(default="bazinga!", local=os.getcwd(), sdc="public")

        with self.assertRaises(FileNotFoundError):
            mms.db_init(default="local", local="bazinga!", sdc="public")

        with self.assertRaises(ValueError):
            mms.db_init(default="sdc", local=os.getcwd(), sdc="bazinga!")

    def test_db_init_output(self):
        self.assertIsNone(mms.db_init(local=os.getcwd()))


@ddt
class Def2PsdTestCase(unittest.TestCase):
    @data(np.random.random((100, 32, 32, 16)))
    def test_def2psd_input(self, value):
        with self.assertRaises(TypeError):
            mms.def2psd(value)

    @data(generate_vdf(64.0, 100, (32, 32, 16), False, "I AM GROOT!!", "s^3/cm^6"))
    def test_def2psd_input_mass_ratio(self, value):
        with self.assertRaises(ValueError):
            mms.def2psd(value)

    @data(generate_vdf(64.0, 100, (32, 32, 16), False, "ions", "bazinga"))
    def test_def2psd_input_convert(self, value):
        with self.assertRaises(ValueError):
            mms.def2psd(value)

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


class UnitConversionsTestCase(unittest.TestCase):
    def setUp(self):
        # Energies 10 to 41 eV (generate_vdf starts at 0 eV)
        vdf = generate_vdf(64.0, 5, [32, 32, 16], species="ions", units="s^3/m^6")
        self.vdf = vdf.assign(energy=vdf.energy + 10.0)
        self.energy = self.vdf.energy.data

    def _inputs(self):
        # Skymap with one time step, spectrum with (time, energy) energies and
        # pitch-angle distribution (time, energy, angle)
        attrs = {"species": "ions", "UNITS": "s^3/m^6"}
        spectr = xr.DataArray(
            np.random.random((5, 32)),
            coords={
                "time": self.vdf.time.data,
                "energy": (("time", "idx"), self.energy),
            },
            dims=["time", "idx"],
            attrs=attrs,
        )
        pad = xr.Dataset(
            {
                "data": (
                    ("time", "idx0", "idx1"),
                    np.random.random((5, 32, 12)),
                    {"UNITS": "s^3/m^6"},
                ),
                "energy": (("time", "idx0"), self.energy),
            },
            coords={"time": self.vdf.time.data},
            attrs={"species": "ions"},
        )
        return [self.vdf.isel(time=[0]), spectr, pad]

    @staticmethod
    def _values(inp):
        return inp.data.data if isinstance(inp, xr.Dataset) else inp.data

    def test_unit_conversions_round_trip(self):
        for inp in self._inputs():
            for forward, backward in [
                (mms.psd2def, mms.def2psd),
                (mms.psd2dpf, mms.dpf2psd),
            ]:
                result = backward(forward(inp))
                self.assertEqual(self._values(result).shape, self._values(inp).shape)
                np.testing.assert_allclose(self._values(result), self._values(inp))

    def test_unit_conversions_values(self):
        # 1 s^3/km^6 of protons at E: DEF = E^2 / (1e6 * 0.53707) and
        # DPF = 1e3 E / (1e6 * 0.53707)
        vdf = self.vdf.copy()
        vdf["data"] = vdf.data.copy(data=np.ones(vdf.data.shape))
        vdf.data.attrs = {"UNITS": "s^3/km^6"}
        e_mat = self.energy[:, :, None, None]
        np.testing.assert_allclose(
            mms.psd2def(vdf).data.data,
            np.broadcast_to(e_mat**2 / 5.3707e5, vdf.data.shape),
        )
        np.testing.assert_allclose(
            mms.psd2dpf(vdf).data.data,
            np.broadcast_to(1e3 * e_mat / 5.3707e5, vdf.data.shape),
        )

    def test_unit_conversions_species(self):
        # DEF scales as 1 / m^2, for any spelling of the species
        def_p = mms.psd2def(self.vdf).data.data
        m_e = constants.electron_mass / constants.proton_mass

        for species, mass in [
            ("electron", m_e),
            ("Electrons", m_e),
            ("ion", 1.0),
            ("alpha", 4.0),
        ]:
            vdf = self.vdf.copy()
            vdf.attrs = {**self.vdf.attrs, "species": species}

            for func in [mms.psd2def, mms.psd2dpf]:
                func(vdf)

            np.testing.assert_allclose(mms.psd2def(vdf).data.data, def_p / mass**2)

    def test_unit_conversions_input_unchanged(self):
        values = self.vdf.data.data.copy()

        for func in [mms.psd2def, mms.psd2dpf]:
            func(self.vdf)

        np.testing.assert_array_equal(self.vdf.data.data, values)
        self.assertEqual(self.vdf.data.attrs["UNITS"], "s^3/m^6")


@ddt
class DftTimeShiftTestCase(unittest.TestCase):
    @staticmethod
    def _signal(n_pts, f_s=8192.0):
        # Sine and cosine periodic in the window, on a time line in ns
        time = np.datetime64("2016-10-07T12:00:00", "ns")
        time = time + np.round(np.arange(n_pts) / f_s * 1e9).astype("timedelta64[ns]")

        def sig(s):
            return np.sin(2 * np.pi * 5 * s / n_pts) + 0.3 * np.cos(
                2 * np.pi * 17 * s / n_pts + 0.4
            )

        return pyrf.ts_scalar(time, sig(np.arange(n_pts))), sig

    @data(
        (128, -7.629e-6),
        (128, 2.5 / 8192),
        (127, -26.703e-6),
        (127, 2.5 / 8192),
    )
    @unpack
    def test_dft_time_shift_values(self, n_pts, tau):
        # The output at t + tau is the input at t + tau, for fractions of a
        # sample, as irfu-matlab (2e-15 against mms.dft_timeshift). It was
        # shifted by whole samples, the wrong way, and complex
        inp, sig = self._signal(n_pts)
        result = mms.dft_time_shift(inp, tau, f_s=8192.0)
        expected = sig(np.arange(n_pts) + tau * 8192.0)
        self.assertFalse(np.iscomplexobj(result.data))
        np.testing.assert_allclose(result.data, expected, atol=1e-12)
        d_t = (result.time.data - inp.time.data) / np.timedelta64(1, "ns")
        np.testing.assert_array_equal(d_t, np.round(tau * 1e9))

    def test_dft_time_shift_default_f_s(self):
        # Default sampling frequency from the mean time step
        inp, sig = self._signal(8192)
        result = mms.dft_time_shift(inp, 2.5 / 8192)
        np.testing.assert_allclose(result.data, sig(np.arange(8192) + 2.5), atol=1e-8)

    def test_dft_time_shift_vector(self):
        # Each component is shifted as a scalar; the input is unchanged and
        # the attributes are kept
        inp, _ = self._signal(128)
        vec = pyrf.ts_vec_xyz(
            inp.time.data,
            np.stack([inp.data, 2 * inp.data, -inp.data], axis=1),
            attrs={"UNITS": "mV/m"},
        )
        vec_copy = vec.copy(deep=True)
        result = mms.dft_time_shift(vec, 0.3 / 8192, f_s=8192.0)
        scalar = mms.dft_time_shift(inp, 0.3 / 8192, f_s=8192.0)
        np.testing.assert_allclose(result.data[:, 0], scalar.data, atol=1e-13)
        np.testing.assert_allclose(result.data[:, 1], 2 * scalar.data, atol=1e-13)
        self.assertEqual(result.dims, vec.dims)
        self.assertEqual(result.attrs["UNITS"], "mV/m")
        xr.testing.assert_identical(vec, vec_copy)


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

    def test_dpf2psd_relativistic(self):
        # f = gamma^3 m J / v^2 with the relativistic speed: 2 gamma^5 /
        # (gamma + 1) times the non-relativistic f = m J / v^2
        energy = np.array([1e3, 5e4, 5e5])  # eV
        attrs = {"species": "electrons", "UNITS": "1/(cm^2 s sr keV)"}
        spectr = pyrf.ts_spectr(generate_timeline(1.0, 2), energy, np.ones((2, 3)))
        spectr.attrs = attrs
        ratio = mms.dpf2psd(spectr, relativistic=True).data / mms.dpf2psd(spectr).data
        np.testing.assert_allclose(ratio[0], [1.0098, 1.5205, 20.36], rtol=1e-3)

        gamma = 1 + energy / 510998.95
        np.testing.assert_allclose(ratio[0], 2 * gamma**5 / (gamma + 1), rtol=1e-6)

        # Protons: 4.9 % at 10 MeV
        spectr.attrs = {**attrs, "species": "ions"}
        spectr = spectr.assign_coords(energy=energy * 20)
        ratio = mms.dpf2psd(spectr, relativistic=True).data / mms.dpf2psd(spectr).data
        self.assertAlmostEqual(float(ratio[0, -1]), 1.0488, delta=1e-4)


@ddt
class Dsl2GseTestCase(unittest.TestCase):
    def test_dsl2gse_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gse(
                generate_ts(64.0, 42, tensor_order=1), np.random.random((42, 3)), 1
            )

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gse_output(self, value):
        result = mms.dsl2gse(generate_ts(64.0, 42, tensor_order=1), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gse(generate_ts(64.0, 42, tensor_order=1), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dsl2GsmTestCase(unittest.TestCase):
    def test_dsl2gsm_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gsm(
                generate_ts(64.0, 42, tensor_order=1), np.random.random((42, 3)), 1
            )

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gsm_output(self, value):
        result = mms.dsl2gsm(generate_ts(64.0, 42, tensor_order=1), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gsm(generate_ts(64.0, 42, tensor_order=1), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class DslTransformationValuesTestCase(unittest.TestCase):
    @data((mms.dsl2gse, "GSE"), (mms.dsl2gsm, "GSM"))
    @unpack
    def test_dsl_transformation_values(self, func, frame):
        b_xyz = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 5), np.tile([1.0, 2.0, 3.0], (5, 1))
        )

        # Spin axis along z: DSL and the target frame coincide
        np.testing.assert_allclose(
            func(b_xyz, np.array([0.0, 0.0, 1.0])).data, b_xyz.data, atol=1e-12
        )

        spin_axis = np.array([0.1, -0.2, 0.97])
        forward = func(b_xyz, spin_axis / np.linalg.norm(spin_axis))
        backward = func(forward, spin_axis / np.linalg.norm(spin_axis), -1)
        np.testing.assert_allclose(backward.data, b_xyz.data)

        # The GSE/GSM -> DSL output used to be labelled GSE/GSM
        self.assertEqual(forward.attrs["COORDINATE_SYSTEM"], frame)
        self.assertEqual(backward.attrs["COORDINATE_SYSTEM"], "DSL")

        # A spin axis that isn't a unit vector gives the same rotation (it used
        # to scale and distort the field)
        np.testing.assert_allclose(func(b_xyz, 3.0 * spin_axis).data, forward.data)
        np.testing.assert_allclose(
            np.linalg.norm(forward.data, axis=1), np.linalg.norm(b_xyz.data, axis=1)
        )


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
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, [0, 0, 1])
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
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, [0, 0, 1])
        self.assertIsInstance(result, xr.DataArray)

    @data([1, 0, 0], generate_ts(64.0, 10, tensor_order=1))
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
        result = mms.eis_combine_proton_pad(
            phxtof_allt, extof_allt, [0, 0, 1], despin=True
        )
        self.assertIsInstance(result, xr.DataArray)

    @staticmethod
    def _allt(energy, factor, n_t=12):
        # 6 telescopes looking in different directions, flux factor * time
        # index + channel index, spins of 4 samples, widths 0.1 E and 0.2 E
        time = generate_timeline(1.0, n_t)
        flux = factor * np.arange(n_t, dtype=float)[:, None] + np.arange(len(energy))
        look = [[0, 0, -1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        allt = {}
        for i in range(6):
            allt[f"t{i:d}"] = xr.DataArray(
                flux * (i + 1), coords=[time, energy], dims=["time", "energy"]
            )
            allt[f"look_t{i:d}"] = pyrf.ts_vec_xyz(time, np.tile(look[i], (n_t, 1)))
        allt["spin"] = pyrf.ts_scalar(time, np.arange(n_t) // 4)
        attrs = {"delta_energy_minus": 0.1 * energy, "delta_energy_plus": 0.2 * energy}
        return xr.Dataset(allt, attrs=attrs)

    def test_eis_combine_proton_pad_channels(self):
        # The PADs of the combined spectra, labelled with their energies (on
        # the 2021 tables, the columns were stitched and mislabelled)
        e_phx = np.array([11.9, 17.1, 23.4, 30.7, 40.5, 53.0, 68.4])
        e_ex = np.array([56.3, 81.9, 118.0, 180.7, 294.5, 480.2, 783.3])
        phx, ext = self._allt(e_phx, 100), self._allt(e_ex, 1000)
        result = mms.eis_combine_proton_pad(phx, ext, [0, 0, 1])

        np.testing.assert_array_equal(result.energy.data, np.hstack([e_phx, e_ex[1:]]))
        np.testing.assert_array_equal(
            result.data[..., :7], mms.eis_pad(phx, [0, 0, 1]).data
        )
        np.testing.assert_array_equal(
            result.data[..., 7:], mms.eis_pad(ext, [0, 0, 1]).data[..., 1:]
        )

        # Spin average of the complete spins (the last one was dropped, the
        # spins of PHxTOF used on the times of ExTOF)
        despun = mms.eis_combine_proton_pad(phx, ext, [0, 0, 1], despin=True)
        np.testing.assert_array_equal(despun.time.data, phx.time.data[[0, 4, 8]])
        xr.testing.assert_identical(despun, mms.eis_pad_spinavg(result, ext.spin))


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

    @staticmethod
    def _allt(energy, seconds, factor):
        # 6 telescopes with flux factor * second + channel index, widths 0.1 E
        # below and 0.2 E above
        time = np.datetime64("2021-07-11T00:00:00", "ns") + (
            np.asarray(seconds) * 1e9
        ).astype("timedelta64[ns]")
        flux = factor * np.asarray(seconds, dtype=float)[:, None] + np.arange(
            len(energy)
        )
        allt = {
            f"t{i:d}": xr.DataArray(
                flux, coords=[time, energy], dims=["time", "energy"]
            )
            for i in range(6)
        }
        allt["spin"] = pyrf.ts_scalar(time, np.zeros(len(time)))
        attrs = {"delta_energy_minus": 0.1 * energy, "delta_energy_plus": 0.2 * energy}
        return xr.Dataset(allt, attrs=attrs)

    def test_eis_combine_proton_spec_channels(self):
        # Channels of IDL 6.1 on the 2021 tables: all the PHxTOF channels and
        # the ExTOF ones but the lowest (it was kept and the top PHxTOF one
        # left out), with their widths (they were stored as edges)
        e_phx = np.array([11.9, 17.1, 23.4, 30.7, 40.5, 53.0, 68.4])
        e_ex = np.array([56.3, 81.9, 118.0, 180.7, 294.5, 480.2, 783.3])
        phx, ext = self._allt(e_phx, [0, 1, 2], 100), self._allt(e_ex, [0, 1, 2], 1000)
        result = mms.eis_combine_proton_spec(phx, ext)

        energy = np.hstack([e_phx, e_ex[1:]])
        np.testing.assert_array_equal(result.energy.data, energy)
        np.testing.assert_allclose(result.attrs["delta_energy_minus"], 0.1 * energy)
        np.testing.assert_allclose(result.attrs["delta_energy_plus"], 0.2 * energy)
        np.testing.assert_array_equal(result.t3.data[:, :7], phx.t3.data)
        np.testing.assert_array_equal(result.t3.data[:, 7:], ext.t3.data[:, 1:])

    def test_eis_combine_proton_spec_units(self):
        # Fluxes only, in the same units, as IDL (counts and cps were combined,
        # different units became a list)
        e_phx = np.array([11.9, 17.1, 23.4, 30.7, 40.5, 53.0, 68.4])
        e_ex = np.array([56.3, 81.9, 118.0, 180.7, 294.5, 480.2, 783.3])
        flux = "1/(cm^2 s sr keV)"

        for units in [("1/s", "1/s"), (" ", " "), (flux, "1/s")]:
            phx = self._allt(e_phx, [0, 1, 2], 100)
            ext = self._allt(e_ex, [0, 1, 2], 1000)

            for i in range(6):
                phx[f"t{i:d}"].attrs["UNITS"] = units[0]
                ext[f"t{i:d}"].attrs["UNITS"] = units[1]

            with self.subTest(units=units), self.assertRaises(ValueError):
                mms.eis_combine_proton_spec(phx, ext)

        for i in range(6):
            ext[f"t{i:d}"].attrs["UNITS"] = flux

        result = mms.eis_combine_proton_spec(phx, ext)
        self.assertEqual(result.t0.attrs["UNITS"], flux)

    def test_eis_combine_proton_spec_cross_over(self):
        # PHxTOF channels above the second ExTOF one averaged with the ExTOF
        # ones below the top PHxTOF one, at the geometric mean of their edges
        e_phx = np.array([20.0, 40.0, 60.0, 80.0])
        e_ex = np.array([45.0, 50.0, 70.0, 120.0])
        phx, ext = self._allt(e_phx, [0, 1], 100), self._allt(e_ex, [0, 1], 1000)
        result = mms.eis_combine_proton_spec(phx, ext)

        low = [min(0.9 * 60.0, 0.9 * 50.0), min(0.9 * 80.0, 0.9 * 70.0)]
        high = [max(1.2 * 60.0, 1.2 * 50.0), max(1.2 * 80.0, 1.2 * 70.0)]
        np.testing.assert_allclose(
            result.energy.data, [20.0, 40.0, *np.sqrt(np.multiply(low, high)), 120.0]
        )
        np.testing.assert_allclose(
            result.t0.data[:, 2:4],
            (phx.t0.data[:, 2:4] + ext.t0.data[:, 1:3]) / 2,
        )
        np.testing.assert_allclose(
            result.attrs["delta_energy_minus"][2:4], result.energy.data[2:4] - low
        )

    def test_eis_combine_proton_spec_times(self):
        # Only the times of both instruments, paired by time (samples of
        # different times could be averaged)
        e_phx = np.array([20.0, 40.0])
        e_ex = np.array([45.0, 50.0, 120.0])
        phx = self._allt(e_phx, [0, 1, 2, 3, 5], 100)
        ext = self._allt(e_ex, [1, 2, 3, 4, 5], 1000)
        result = mms.eis_combine_proton_spec(phx, ext)

        np.testing.assert_array_equal(result.time.data, ext.time.data[[0, 1, 2, 4]])
        np.testing.assert_array_equal(result.t0.data[:, 0], [100, 200, 300, 500])
        np.testing.assert_array_equal(result.t0.data[:, 2], [1001, 2001, 3001, 5001])
        self.assertEqual(len(result.spin), 4)

        with self.assertRaises(ValueError):
            mms.eis_combine_proton_spec(phx.drop_vars("t5"), ext)


@ddt
class EisMomentsTestCase(unittest.TestCase):
    @staticmethod
    def _maxwellian_flux(mass, n_cc=1.0, kt_kev=10.0, n_t=3):
        # Omni-directional differential flux (1/(cm^2 s sr keV)) of an isotropic
        # Maxwellian, on a wide energy grid (keV) so the partial moments are full
        energy = np.geomspace(0.01, 2000.0, 400)
        k_t = kt_kev * 1e3 * constants.e
        e_j = energy * 1e3 * constants.e
        vdf = n_cc * 1e6 * (mass / (2 * np.pi * k_t)) ** 1.5 * np.exp(-e_j / k_t)
        flux = 2 * e_j / mass**2 * vdf * 1e3 * constants.e / 1e4
        times = generate_timeline(1.0, n_t)
        flux = np.tile(flux, (n_t, 1))
        # channel widths: to the geometric means of the neighbouring energies
        edges = np.sqrt(energy[1:] * energy[:-1])
        edges = np.hstack(
            [energy[0] ** 2 / edges[0], edges, energy[-1] ** 2 / edges[-1]]
        )
        attrs = {
            "delta_energy_minus": energy - edges[:-1],
            "delta_energy_plus": edges[1:] - energy,
        }
        return xr.DataArray(
            flux, coords=[times, energy], dims=["time", "energy"], attrs=attrs
        )

    @data(("proton", 1), ("alpha", 4), ("helium", 4), ("oxygen", 16))
    @unpack
    def test_eis_moments_maxwellian(self, specie, mass_number):
        # n = 1 cm^-3, kT = 10 keV: P = n kT = 1.602 nPa (the 2nd order integral
        # is the energy density, 1.5 times larger), <|v|> = sqrt(8 kT / (pi m))
        mass = mass_number * constants.proton_mass
        n_i, v_i, p_i, t_i = mms.eis_moments(self._maxwellian_flux(mass), specie)
        k_t = 1e4 * constants.e

        np.testing.assert_allclose(n_i.data, 1.0, rtol=1e-4)
        np.testing.assert_allclose(p_i.data, 1e15 * k_t, rtol=1e-4)
        np.testing.assert_allclose(t_i.data, 1e4, rtol=1e-4)
        np.testing.assert_allclose(
            v_i.data, np.sqrt(8 * k_t / (np.pi * mass)) / 1e3, rtol=1e-4
        )

    def test_eis_moments_nan_channel(self):
        flux = self._maxwellian_flux(constants.proton_mass)
        flux.data[1, 10] = np.nan
        n_i, _, p_i, _ = mms.eis_moments(flux)

        self.assertTrue(np.isnan(n_i.data[1]) and np.isnan(p_i.data[1]))
        np.testing.assert_allclose(n_i.data[[0, 2]], 1.0, rtol=1e-4)

    def test_eis_moments_input(self):
        flux = self._maxwellian_flux(constants.proton_mass)

        with self.assertRaises(ValueError):
            mms.eis_moments(flux, "electron")

        flux.attrs = {}
        with self.assertRaisesRegex(ValueError, "energy widths"):
            mms.eis_moments(flux)

    def test_eis_moments_units(self):
        # Fluxes only (counts and cps were accepted)
        flux = self._maxwellian_flux(constants.proton_mass)

        for units in ["1/s", " "]:
            flux.attrs["UNITS"] = units

            with self.subTest(units=units), self.assertRaisesRegex(ValueError, "flux"):
                mms.eis_moments(flux)

        flux.attrs["UNITS"] = "1/(cm^2 s sr keV)"
        np.testing.assert_allclose(mms.eis_moments(flux)[0].data, 1.0, rtol=1e-4)

    def test_eis_moments_channel_sum(self):
        # Sums over the channels and their widths (Simpson's rule from the first
        # to the last channel centre left out the half channels at both ends)
        energy = np.array([50.0, 80.0, 120.0])
        attrs = {
            "delta_energy_minus": np.array([10.0, 15.0, 20.0]),
            "delta_energy_plus": np.array([15.0, 20.0, 30.0]),
        }
        flux = xr.DataArray(
            np.array([[3.0, 2.0, 1.0]]),
            coords=[generate_timeline(1.0, 1), energy],
            dims=["time", "energy"],
            attrs=attrs,
        )
        n_i, v_i, p_i, t_i = mms.eis_moments(flux)

        kev = 1e3 * constants.e
        e_j, w_j = energy * kev, np.array([25.0, 35.0, 50.0]) * kev
        j_si = 1e4 * flux.data[0] / kev
        fact = 4 * np.pi * np.sqrt(constants.proton_mass / 2)
        n_si = fact * np.sum(j_si / np.sqrt(e_j) * w_j)
        p_si = 2 / 3 * fact * np.sum(j_si * np.sqrt(e_j) * w_j)

        np.testing.assert_allclose(n_i.data, 1e-6 * n_si, rtol=1e-12)
        np.testing.assert_allclose(p_i.data, 1e9 * p_si, rtol=1e-12)
        np.testing.assert_allclose(t_i.data, p_si / (n_si * constants.e), rtol=1e-12)
        np.testing.assert_allclose(
            v_i.data,
            1e-3
            * fact
            * np.sum(j_si * w_j)
            / (n_si * np.sqrt(constants.proton_mass / 2)),
            rtol=1e-12,
        )

    def test_eis_moments_background(self):
        # n - n_bg and P - p_bg, T from them (P was reduced by 3 p_bg)
        flux = self._maxwellian_flux(constants.proton_mass)
        n_bg = pyrf.ts_scalar(flux.time.data, np.full(len(flux.time), 0.2))
        p_bg = pyrf.ts_scalar(flux.time.data, np.full(len(flux.time), 0.4))
        n_0, _, p_0, _ = mms.eis_moments(flux)
        n_i, _, p_i, t_i = mms.eis_moments(flux, n_bg=n_bg, p_bg=p_bg)

        np.testing.assert_allclose(n_i.data, n_0.data - 0.2)
        np.testing.assert_allclose(p_i.data, p_0.data - 0.4)
        np.testing.assert_allclose(
            t_i.data, 1e-9 * p_i.data / (1e6 * n_i.data * constants.e)
        )


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

    def test_eis_omni_spin_sector_attrs(self):
        # The spin numbers and sectors as coordinates and the energy widths and
        # species in the attributes (left out: KeyError in eis_spec_combine_sc)
        eis = generate_eis(64.0, 100, "srvy", "extof", "l2", "proton", "flux", 2)
        result = mms.eis_omni(eis)

        np.testing.assert_array_equal(result.spin.data, eis.spin.data)
        np.testing.assert_array_equal(result.sector.data, eis.sector.data)
        np.testing.assert_array_equal(
            result.attrs["delta_energy_plus"], eis.attrs["delta_energy_plus"]
        )
        self.assertEqual(result.attrs["species"], "proton")

        # attributes not shared with the telescope
        result.attrs["new"] = 1
        self.assertNotIn("new", eis.t0.attrs)

        # ValueError (was AssertionError)
        with self.assertRaises(ValueError):
            mms.eis_omni(eis, "median")


class EisAngAngTestCase(unittest.TestCase):
    # Telescope polar angles (degrees from +z) and 8 samples per spin, the
    # look direction turning by 45 degrees per sample (sectors 0, 4, ..., 28)
    POLAR = np.array([28.9, 55.5, 81.0, 110.0, 135.6, 162.2])

    def _allt(self, n_spins=3):
        n_t = 8 * n_spins
        time = generate_timeline(2.5, n_t)
        azimuth = np.deg2rad(22.5 + 45.0 * (np.arange(n_t) % 8))
        energy = np.array([60.0, 120.0])
        allt = {}
        for i, polar in enumerate(np.deg2rad(self.POLAR)):
            look = np.stack(
                [
                    np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.full(n_t, np.cos(polar)),
                ],
                axis=1,
            )
            allt[f"look_t{i:d}"] = pyrf.ts_vec_xyz(time, look)
            flux = np.tile([[10.0 * (i + 1), 1.0]], (n_t, 1))
            allt[f"t{i:d}"] = xr.DataArray(
                flux, coords=[time, energy], dims=["time", "energy"]
            )
        allt["spin"] = pyrf.ts_scalar(time, np.arange(n_t) // 8)
        allt["sector"] = pyrf.ts_scalar(time, 4 * (np.arange(n_t) % 8))
        attrs = {
            "species": "proton",
            "delta_energy_minus": np.array([5.0, 10.0]),
            "delta_energy_plus": np.array([6.0, 12.0]),
        }
        return xr.Dataset(allt, attrs=attrs)

    def test_eis_ang_ang_look_directions(self):
        # Every sample in the cell of its look direction (azimuth in [0, 360),
        # polar angle): all cells filled (about 19 % of the samples were
        # binned, at the polar angle + 90 degrees)
        result = mms.eis_ang_ang(self._allt(), frame="gse")

        self.assertTupleEqual(result.shape, (3, 2, 8, 6))
        np.testing.assert_allclose(result.phi.data, 22.5 + 45.0 * np.arange(8))
        np.testing.assert_allclose(
            result.theta.data, 10.0 + 160.0 / 6 * (np.arange(6) + 0.5)
        )
        np.testing.assert_allclose(
            result.data[:, 0], np.broadcast_to(10.0 * np.arange(1, 7), (3, 8, 6))
        )
        self.assertEqual(result.phi.attrs["COORDINATE_SYSTEM"], "GSE")
        np.testing.assert_array_equal(result.time.data, self._allt().time.data[::8])

    def test_eis_ang_ang_mean_and_channels(self):
        # Two telescopes in the same cell: their mean (the last one was kept);
        # empty cells NaN (they were 0); widths of the channels used
        allt = self._allt()
        allt["look_t1"] = allt["look_t0"].copy()
        result = mms.eis_ang_ang(allt, en_chan=[0], frame="gse")

        np.testing.assert_allclose(result.data[:, 0, :, 0], 15.0)
        self.assertTrue(np.all(np.isnan(result.data[:, 0, :, 1])))
        np.testing.assert_array_equal(result.attrs["delta_energy_minus"], [5.0])

    def test_eis_ang_ang_dbcs(self):
        # DBCS: the GSE look directions rotated with the attitude
        allt = self._allt()
        time = allt.time.data
        defatt = xr.Dataset(
            {
                "z_ra": pyrf.ts_scalar(time, np.linspace(268.0, 272.0, len(time))),
                "z_dec": pyrf.ts_scalar(time, np.linspace(65.0, 67.0, len(time))),
            }
        )
        result = mms.eis_ang_ang(allt, defatt=defatt)

        rotated = allt.copy()
        for i in range(6):
            rotated[f"look_t{i:d}"] = mms.dsl2gse(allt[f"look_t{i:d}"], defatt, -1)
        expected = mms.eis_ang_ang(rotated, frame="gse")

        np.testing.assert_array_equal(result.data, expected.data)
        self.assertEqual(result.theta.attrs["COORDINATE_SYSTEM"], "DBCS")

        with self.assertRaisesRegex(ValueError, "defatt"):
            mms.eis_ang_ang(allt)

        with self.assertRaises(ValueError):
            mms.eis_ang_ang(allt, frame="gsm")


class EisSkymapTestCase(unittest.TestCase):
    @staticmethod
    def _ang_ang(seconds, value=1.0, units="1/(cm^2 s sr keV)"):
        # Angle-angle distributions at the given times (one per spin), 2
        # energies (keV) with their widths, 8 azimuths and 6 polar angles
        time = np.datetime64("2017-07-11T22:00:00", "ns") + (
            np.asarray(seconds) * 1e9
        ).astype("timedelta64[ns]")
        energy = np.array([60.0, 120.0])
        out = xr.DataArray(
            np.full((len(time), 2, 8, 6), float(value)),
            coords=[
                time,
                energy,
                22.5 + 45.0 * np.arange(8),
                23.3 + 26.7 * np.arange(6),
            ],
            dims=["time", "energy", "phi", "theta"],
            attrs={
                "species": "proton",
                "delta_energy_minus": np.array([5.0, 10.0]),
                "delta_energy_plus": np.array([6.0, 12.0]),
                "UNITS": units,
                "GLOBAL": {},
            },
        )
        out.energy.attrs = {"UNITS": "keV"}
        return out

    def test_eis_skymap_units(self):
        # Energies and widths in eV (the widths stayed in keV, labelled keV);
        # a single time (IndexError)
        result = mms.eis_skymap(self._ang_ang([0.0]), to_psd=False)

        np.testing.assert_array_equal(result.energy.data[0], [6e4, 1.2e5])
        self.assertEqual(result.energy.attrs["UNITS"], "eV")
        np.testing.assert_array_equal(result.attrs["delta_energy_minus"], [5e3, 1e4])
        np.testing.assert_array_equal(result.attrs["delta_energy_plus"], [6e3, 1.2e4])

    def test_eis_skymap_combine_sc(self):
        # Combined at the nearest spin (by index, a missing spin shifted the
        # next ones), with the widths per channel (repeated for every time) and
        # NaN where no spacecraft has data (the sum was 0)
        ref = mms.eis_skymap(self._ang_ang([0, 20, 40, 60]), to_psd=False)
        other = self._ang_ang([1, 21, 41, 61, 81], value=3.0)
        other.data[3:] = 5.0
        other = mms.eis_skymap(other.drop_isel(time=2), to_psd=False)
        ref.data.data[0, 0, 0, 0] = np.nan
        other.data.data[0, 0, 0, 0] = np.nan
        result = mms.eis_skymap_combine_sc([ref, other])

        np.testing.assert_array_equal(result.time.data, ref.time.data)
        np.testing.assert_allclose(result.data.data[1, 1], 2.0)
        np.testing.assert_allclose(result.data.data[2, 1], 1.0)
        np.testing.assert_allclose(result.data.data[3, 1], 3.0)
        np.testing.assert_array_equal(result.attrs["delta_energy_plus"], [6e3, 1.2e4])

        summed = mms.eis_skymap_combine_sc([ref, other], "sum")
        self.assertTrue(np.isnan(summed.data.data[0, 0, 0, 0]))
        np.testing.assert_allclose(summed.data.data[1, 1], 4.0)

        with self.assertRaises(ValueError):
            mms.eis_skymap_combine_sc([ref, other], "median")

        psd = mms.eis_skymap(self._ang_ang([0, 20], units="s^3/cm^6"), to_psd=False)
        with self.assertRaisesRegex(ValueError, "UNITS"):
            mms.eis_skymap_combine_sc([ref, psd])


class EisPadCombineScTestCase(unittest.TestCase):
    @staticmethod
    def _pad(t_0, n_pts, value, energies=(50.0, 80.0, 120.0)):
        # PADs at 2.5 s, 12 pitch angle bins, value at every bin and energy
        time = np.datetime64(t_0, "ns") + (np.arange(n_pts) * 2.5e9).astype(
            "timedelta64[ns]"
        )
        theta = 7.5 + 15.0 * np.arange(12)
        data = np.full((n_pts, 12, len(energies)), float(value))
        return xr.DataArray(
            data,
            coords=[time, theta, np.asarray(energies)],
            dims=["time", "theta", "energy"],
        )

    def test_eis_pad_combine_sc_channels(self):
        # Different numbers of channels (ValueError in both directions), at the
        # closest channels, on the energies of the spacecraft with the fewest
        ref = self._pad("2017-07-11T22:00:00", 10, 1.0)
        more = self._pad(
            "2017-07-11T22:00:00", 12, 3.0, energies=(48.0, 60.0, 81.0, 119.0)
        )
        more.data[..., 1] = np.nan  # 60 keV: not a closest channel

        for pads in [[ref, more], [more, ref]]:
            result = mms.eis_pad_combine_sc(pads)
            np.testing.assert_array_equal(result.energy.data, [50.0, 80.0, 120.0])
            np.testing.assert_array_equal(result.time.data, ref.time.data)
            np.testing.assert_allclose(result.data, 2.0)

    def test_eis_pad_combine_sc_times(self):
        # At the nearest time (by index, a spacecraft starting a sample later
        # was paired with the next times); none within half a step: left out
        ref = self._pad("2017-07-11T22:00:00", 10, 1.0)
        late = self._pad("2017-07-11T22:00:02.6", 10, 3.0)
        late.data[:] = 3.0 + np.arange(10)[:, None, None]
        result = mms.eis_pad_combine_sc([ref, late])

        np.testing.assert_allclose(result.data[0], 1.0)
        np.testing.assert_allclose(
            result.data[1:, 0, 0], (1.0 + late.data[:9, 0, 0]) / 2
        )

        with self.assertRaises(ValueError):
            mms.eis_pad_combine_sc([ref, ref.assign_coords(theta=ref.theta + 1)])


class EisSpecCombineScTestCase(unittest.TestCase):
    @staticmethod
    def _omni(t_0, n_pts, first_sector, value, energies=None, spin_avg=False):
        # Survey-like omni spectrum: a sample every 2.5 s, 8 sectors per spin
        # (0, 4, ..., 28), flux value(sector) at every energy
        energies = np.array([50.0, 80.0, 120.0]) if energies is None else energies
        time = np.datetime64(t_0, "ns") + (
            np.arange(n_pts) * 2.5e9 * (8 if spin_avg else 1)
        ).astype("timedelta64[ns]")
        sector = (first_sector + 4 * np.arange(n_pts)) % 32
        flux = np.tile(value(sector)[:, None], (1, len(energies)))
        out = xr.DataArray(
            flux,
            coords=[time, energies],
            dims=["time", "energy"],
            attrs={"delta_energy_plus": energies / 10, "UNITS": "1/(cm^2 s sr keV)"},
        )
        return out if spin_avg else out.assign_coords(sector=("time", sector))

    def test_eis_spec_combine_sc_documented_chain(self):
        # get_eis_allt -> eis_omni -> eis_spec_combine_sc (KeyError: 'energy_dplus')
        omnis = [
            mms.eis_omni(
                generate_eis(64.0, 100, "srvy", "extof", "l2", "proton", "flux", ic)
            )
            for ic in range(2, 5)
        ]
        result = mms.eis_spec_combine_sc(omnis)

        np.testing.assert_allclose(
            result.data, np.mean([o.data for o in omnis], axis=0), rtol=1e-12
        )
        self.assertIn("delta_energy_minus", result.attrs)

    def test_eis_spec_combine_sc_sectors(self):
        # Paired at the same sector (look direction), not by index: a flux that
        # depends on the sector only gives back the value of each sector
        def value(sector):
            return 1.0 + sector.astype(float)

        ref = self._omni("2017-07-11T22:00:01.0", 40, 4, value)
        other = self._omni("2017-07-11T22:00:00.1", 41, 28, value)
        result = mms.eis_spec_combine_sc([ref, other])

        np.testing.assert_array_equal(result.time.data, ref.time.data)
        np.testing.assert_array_equal(result.sector.data, ref.sector.data)
        np.testing.assert_allclose(result.data, ref.data)

        result = mms.eis_spec_combine_sc([ref, other], "sum")
        np.testing.assert_allclose(result.data[2:-2], 2 * ref.data[2:-2])

    def test_eis_spec_combine_sc_spin_averages(self):
        # Spin averages paired by time: a spin missing on one spacecraft is
        # left out there (by index, all the next ones were shifted by a spin)
        ref = self._omni(
            "2017-07-11T22:00:00", 10, 0, lambda s: 0 * s + 1.0, spin_avg=True
        )
        other = self._omni(
            "2017-07-11T22:00:00.5", 11, 0, lambda s: 0 * s + 3.0, spin_avg=True
        )
        other.data[np.arange(11) > 4] = 5.0
        other = other.drop_isel(time=4)
        result = mms.eis_spec_combine_sc([ref, other])

        np.testing.assert_allclose(result.data[:4], 2.0)
        np.testing.assert_allclose(result.data[4], 1.0)
        np.testing.assert_allclose(result.data[5:], 3.0)

    def test_eis_spec_combine_sc_energies_method(self):
        # Energies of the spacecraft with the fewest channels, the others at
        # their closest channels; ValueError for an unknown method
        ref = self._omni("2017-07-11T22:00:00", 16, 0, lambda s: 0 * s + 1.0)
        other = self._omni(
            "2017-07-11T22:00:00",
            16,
            0,
            lambda s: 0 * s + 3.0,
            energies=np.array([48.0, 60.0, 81.0, 119.0]),
        )
        result = mms.eis_spec_combine_sc([other, ref])

        np.testing.assert_array_equal(result.energy.data, ref.energy.data)
        np.testing.assert_array_equal(
            result.attrs["delta_energy_plus"], ref.attrs["delta_energy_plus"]
        )
        np.testing.assert_allclose(result.data, 2.0)

        with self.assertRaises(ValueError):
            mms.eis_spec_combine_sc([ref, other], "median")

    def test_eis_spec_combine_sc_sum_without_data(self):
        # NaN where no spacecraft has data, as the mean (the sum was 0)
        ref = self._omni("2017-07-11T22:00:00", 16, 0, lambda s: 0 * s + 1.0)
        other = self._omni("2017-07-11T22:00:00", 16, 0, lambda s: 0 * s + 3.0)
        ref.data[0], other.data[0] = np.nan, np.nan
        result = mms.eis_spec_combine_sc([ref, other], "sum")

        self.assertTrue(np.all(np.isnan(result.data[0])))
        np.testing.assert_allclose(result.data[1:], 4.0)


@ddt
class EisPadTestCase(unittest.TestCase):
    @data([1, 0, 0], np.array([1, 0, 0]), generate_ts(64.0, 10, tensor_order=1))
    def test_eis_pad_output(self, vec):
        # NumPy directions raised TypeError
        eis = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_pad(eis, vec)
        self.assertIsInstance(result, xr.DataArray)

    @staticmethod
    def _eis(flux, look, energies, delta_minus, delta_plus):
        # Telescope fluxes flux[telescope] (time x energy) and look directions
        time = generate_timeline(1.0, flux.shape[1])
        eis = {}
        for i in range(6):
            eis[f"t{i:d}"] = xr.DataArray(
                flux[i], coords=[time, energies], dims=["time", "energy"]
            )
            eis[f"look_t{i:d}"] = pyrf.ts_vec_xyz(
                time, np.tile(look[i], (flux.shape[1], 1))
            )
        attrs = {"delta_energy_minus": delta_minus, "delta_energy_plus": delta_plus}
        return xr.Dataset(eis, attrs=attrs)

    def test_eis_pad_values(self):
        # B along z: telescopes looking along -z (pitch angle 0), +z (180) and
        # x (90); the zeros are averaged, as IDL and pySPEDAS (they were NaN)
        look = np.array(
            [[0, 0, -1], [0, 0, -1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
        )
        flux = np.zeros((6, 3, 2))
        flux[0], flux[3] = 4.0, 6.0
        eis = self._eis(flux, look, np.array([100.0, 200.0]), 10.0, 10.0)
        result = mms.eis_pad(eis, [0, 0, 1])

        np.testing.assert_array_equal(result.theta.data[[0, 6, 11]], [7.5, 97.5, 172.5])
        np.testing.assert_allclose(result.data[:, 0, :], 2.0)
        np.testing.assert_allclose(result.data[:, 6, :], 2.0)
        np.testing.assert_allclose(result.data[:, 11, :], 0.0)
        self.assertTrue(np.all(np.isnan(result.data[:, 3, :])))

    def test_eis_pad_energy_edges(self):
        # Channels with a lower (E - delta_minus, it was E + delta_minus) or
        # upper edge in the range
        look = np.tile([0, 0, -1], (6, 1))
        eis = self._eis(np.ones((6, 3, 2)), look, np.array([100.0, 200.0]), 40.0, 5.0)
        result = mms.eis_pad(eis, [0, 0, 1], energy=[150, 180])

        np.testing.assert_array_equal(result.energy.data, [200.0])
        np.testing.assert_array_equal(
            mms.eis_pad(eis, [0, 0, 1]).energy.data, [100.0, 200.0]
        )

    def test_eis_pad_errors(self):
        eis = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)

        with self.assertRaisesRegex(ValueError, "magnetic field"):
            mms.eis_pad(eis)

        with self.assertRaises(ValueError):
            mms.eis_pad(eis, [1, 0])

        with self.assertRaisesRegex(ValueError, "No energy channel"):
            mms.eis_pad(eis, [1, 0, 0], energy=[5000, 6000])

        # Bins with gaps for a width that doesn't divide 180
        with self.assertRaisesRegex(ValueError, "pa_width"):
            mms.eis_pad(eis, [1, 0, 0], pa_width=25)

    def test_eis_pad_attrs(self):
        # Units, species, widths of the channels used and bin width (no attrs)
        eis = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        eis["t0"].attrs["UNITS"] = "1/(cm^2 s sr keV)"
        eis["t0"].attrs["LABLAXIS"] = "Intensity"
        result = mms.eis_pad(eis, [1, 0, 0], energy=[100, 200])
        i_e = np.flatnonzero(np.isin(eis.energy.data, result.energy.data))

        self.assertEqual(result.attrs["UNITS"], "1/(cm^2 s sr keV)")
        self.assertEqual(result.attrs["LABLAXIS"], "Intensity")
        self.assertEqual(result.attrs["species"], "proton")
        self.assertEqual(result.attrs["pa_width"], 15)
        np.testing.assert_array_equal(
            result.attrs["delta_energy_minus"], eis.attrs["delta_energy_minus"][i_e]
        )


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
        eis_pad = mms.eis_pad(eis, [0, 0, 1])
        result = mms.eis_pad_spinavg(eis_pad, eis.spin)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisProtonCorrectionTestCase(unittest.TestCase):
    def test_eis_proton_correction_type(self):
        # TypeError (was AssertionError)
        with self.assertRaises(TypeError):
            mms.eis_proton_correction(np.ones((3, 16)))

    def test_eis_proton_correction_units(self):
        # Fluxes only (counts and cps were corrected)
        eis = generate_eis(64.0, 100, "srvy", "extof", "l2", "proton", "cps", 1)

        for i in range(6):
            eis[f"t{i:d}"].attrs["UNITS"] = "1/s"

        with self.assertRaisesRegex(ValueError, "flux"):
            mms.eis_proton_correction(eis)

        with self.assertRaisesRegex(ValueError, "flux"):
            mms.eis_proton_correction(mms.eis_omni(eis))

        for i in range(6):
            eis[f"t{i:d}"].attrs["UNITS"] = "1/(cm^2 s sr keV)"

        self.assertIsInstance(mms.eis_proton_correction(eis), xr.Dataset)

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

    @staticmethod
    def _eis(spin):
        # One telescope with flux = sample index, at 1 s, with the spin numbers
        time = generate_timeline(1.0, len(spin))
        flux = np.tile(np.arange(len(spin), dtype=float)[:, None], (1, 2))
        eis = {
            "t0": xr.DataArray(
                flux, coords=[time, [50.0, 80.0]], dims=["time", "energy"]
            ),
            "spin": pyrf.ts_scalar(time, np.asarray(spin, dtype=np.uint16)),
        }
        return xr.Dataset(eis)

    def test_eis_spin_avg_complete_spins(self):
        # Complete spins only (4 samples), stamped at their first sample: the
        # partial first and last spins and the spin cut by a gap are left out
        spin = [10] * 2 + [11] * 4 + [12] * 4 + [13] * 2 + [14] * 4 + [15] * 3
        eis = self._eis(spin)
        result = mms.eis_spin_avg(eis)

        np.testing.assert_array_equal(result.time.data, eis.time.data[[2, 6, 12]])
        np.testing.assert_array_equal(result.spin.data, [11, 12, 14])
        np.testing.assert_allclose(result.t0.data[:, 0], [3.5, 7.5, 13.5])
        np.testing.assert_allclose(
            mms.eis_spin_avg(eis, "sum").t0.data[:, 0], [14, 30, 54]
        )

    def test_eis_spin_avg_counter_wrap(self):
        # The spins in time order across the wrap of the uint16 counter (they
        # were ordered by number)
        spin = [65533] * 2 + [65534] * 4 + [65535] * 4 + [0] * 4 + [1] * 4 + [2]
        result = mms.eis_spin_avg(self._eis(spin))

        np.testing.assert_array_equal(result.spin.data, [65534, 65535, 0, 1])
        self.assertTrue(np.all(np.diff(result.time.data) > np.timedelta64(0, "ns")))

    def test_eis_spin_avg_errors(self):
        with self.assertRaisesRegex(ValueError, "no complete spin"):
            mms.eis_spin_avg(self._eis([3] * 5 + [4] * 2))

        with self.assertRaises(ValueError):
            mms.eis_spin_avg(self._eis([1] * 2 + [2] * 4 + [3]), "median")

    def test_eis_pad_spinavg_spins(self):
        # Same spins as eis_spin_avg, stamped at their first sample (they were
        # stamped one spin early, partial spins kept)
        spin = [10] * 2 + [11] * 4 + [12] * 4 + [13] * 3
        eis = self._eis(spin)
        pad = xr.DataArray(
            np.tile(np.arange(13.0)[:, None, None], (1, 12, 2)),
            coords=[eis.time.data, 7.5 + 15 * np.arange(12), [50.0, 80.0]],
            dims=["time", "theta", "energy"],
        )
        result = mms.eis_pad_spinavg(pad, eis.spin)

        np.testing.assert_array_equal(result.time.data, eis.time.data[[2, 6]])
        np.testing.assert_allclose(result.data[:, 0, 0], [3.5, 7.5])

        with self.assertRaises(ValueError):
            mms.eis_pad_spinavg(pad, eis.spin[:-1])

        # Spin numbers of other times (misaligned silently)
        shifted = eis.spin.assign_coords(
            time=eis.spin.time.data + np.timedelta64(1, "s")
        )
        with self.assertRaisesRegex(ValueError, "times of inp"):
            mms.eis_pad_spinavg(pad, shifted)


def _lh_wave(v_ph, theta, f_s=1024.0, f_w=40.0):
    # Potential phi(x - v t) along d (perpendicular to B = 40 nT z): at the
    # spacecraft E.d = (dphi / dt) / v and dB_par = phi n e mu0 / B, so that
    # phi_E = v int(E.d) dt = phi_B = phi. An unrelated E along z x d picks
    # out the direction.
    n_t = int(4 * f_s)
    t_sec = np.arange(n_t) / f_s
    time = np.datetime64("2019-01-01T00:00:00", "ns")
    time = time + (t_sec * 1e9).astype("timedelta64[ns]")
    b_0, n_0, window = 40.0, 10.0, np.hanning(n_t)

    phi = 50.0 * np.sin(2 * np.pi * f_w * t_sec) * window  # V
    e_d = np.gradient(phi, t_sec) / v_ph  # mV/m
    e_p = np.max(np.abs(e_d)) * np.sin(2 * np.pi * 25.0 * t_sec) * window
    d_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
    d_perp = np.array([-np.sin(theta), np.cos(theta), 0.0])
    d_b = phi * n_0 * 1e6 * constants.e * constants.mu_0 / (b_0 * 1e-18)  # nT

    e_xyz = pyrf.ts_vec_xyz(time, np.outer(e_d, d_vec) + np.outer(e_p, d_perp))
    b_scm = pyrf.ts_vec_xyz(time, np.outer(d_b, [0.0, 0.0, 1.0]))
    b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, b_0], (n_t, 1)))
    n_e = pyrf.ts_scalar(time, np.full(n_t, n_0))
    tints = ["2019-01-01T00:00:01.300", "2019-01-01T00:00:02.700"]

    return tints, e_xyz, b_scm, b_xyz, n_e, d_vec


@ddt
class LhWaveAnalysisTestCase(unittest.TestCase):
    @data(300.0, 1200.0)
    def test_lh_wave_analysis_values(self, v_ph):
        tints, e_xyz, b_scm, b_xyz, n_e, d_vec = _lh_wave(v_ph, np.deg2rad(30.0))
        phi_eb, v_best, dir_best, thetas, corrs = mms.lh_wave_analysis(
            tints, e_xyz, b_scm, b_xyz, n_e, min_freq=10.0
        )

        self.assertAlmostEqual(np.rad2deg(thetas[np.argmax(corrs)]), 30.0)
        np.testing.assert_allclose(dir_best, d_vec, atol=1e-6)
        self.assertAlmostEqual(v_best, v_ph, delta=1.0)
        self.assertGreater(np.max(corrs), 0.99)
        self.assertListEqual(list(phi_eb.comp.data), ["Ebest", "Bs"])

    @data([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    def test_lh_wave_analysis_b_along_x(self, b_hat):
        # The wave of _lh_wave with B along +x or -x (axes permuted, z to x,
        # x reversed for -x): B x x vanished and everything was NaN; the
        # angles are then from y towards B x y
        tints, e_xyz, b_scm, b_xyz, n_e, d_vec = _lh_wave(300.0, np.deg2rad(30.0))
        sign = b_hat[0]
        e_xyz, b_scm, b_xyz = [
            pyrf.ts_vec_xyz(inp.time.data, inp.data[:, [2, 0, 1]] * [sign, 1, 1])
            for inp in [e_xyz, b_scm, b_xyz]
        ]
        phi_eb, v_best, dir_best, _, corrs = mms.lh_wave_analysis(
            tints, e_xyz, b_scm, b_xyz, n_e, min_freq=10.0
        )

        np.testing.assert_allclose(dir_best, d_vec[[2, 0, 1]], atol=1e-6)
        self.assertAlmostEqual(v_best, 300.0, delta=1.0)
        self.assertGreater(np.max(corrs), 0.99)
        self.assertFalse(np.any(np.isnan(phi_eb.data)))

    def test_lh_wave_analysis_vmax(self):
        tints, e_xyz, b_scm, b_xyz, n_e, _ = _lh_wave(1200.0, np.deg2rad(30.0))
        with self.assertLogs("pyrfu", level="WARNING"):
            _, v_best, _, _, _ = mms.lh_wave_analysis(
                tints, e_xyz, b_scm, b_xyz, n_e, min_freq=10.0, vmax=500.0
            )
        self.assertEqual(v_best, 500.0)


def _brst_segments_response(rows, status_code=200):
    header = (
        "DATASEGMENTID,TAISTARTTIME (TAI seconds since 1958-01-01),"
        "TAIENDTIME (TAI seconds since 1958-01-01),PARAMETERSETID,FOM,ISPENDING,"
        "INPLAYLIST,STATUS,NUMEVALCYCLES,SOURCEID,CREATETIME,FINISHTIME,DISCUSSION"
    )
    lines = [
        f'{i},{start},{end},p,1.0,0,0,{status},1,eva,t,t,"Bow shock, maybe"'
        for i, (start, end, status) in enumerate(rows)
    ]
    response = mock.MagicMock(status_code=status_code)
    response.text = "\n".join([header, *lines]) + "\n"
    if status_code != 200:
        response.raise_for_status.side_effect = requests.HTTPError("404")
    return response


class LoadBrstSegmentsTestCase(unittest.TestCase):
    # TAI 1829005990 s is 2015-12-17T01:12:34 UTC (36 leap seconds)
    tint = ["2015-12-17T01:13:00", "2015-12-17T01:20:00"]

    def test_load_brst_segments_values(self):
        rows = [
            (1829006400, 1829006500, "COMPLETE+FINISHED"),  # unsorted, kept
            (1829005990, 1829006120, "COMPLETE+FINISHED"),  # straddles tint[0]
            (1829006200, 1829006300, "DERELICT+FINISHED"),  # incomplete
            (1829005800, 1829005900, "COMPLETE+FINISHED"),  # ends before tint
            (1829006700, 1829006800, "COMPLETE+FINISHED"),  # starts after tint
        ]
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response(rows)
        ) as get:
            result = mms.load_brst_segments(self.tint)

        # End times get 10 s more
        expected = [
            ["2015-12-17T01:12:34.000000000", "2015-12-17T01:14:54.000000000"],
            ["2015-12-17T01:19:24.000000000", "2015-12-17T01:21:14.000000000"],
        ]
        self.assertListEqual(result, expected)

        # Server-side selection of the segments overlapping tint
        url = get.call_args.args[0]
        self.assertTrue(url.startswith("https://lasp.colorado.edu/mms/sdc/"))
        self.assertIn("TAISTARTTIME<=1829006436", url)
        self.assertIn("TAIENDTIME>=1829006006", url)

    def test_load_brst_segments_empty(self):
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response([])
        ):
            self.assertListEqual(mms.load_brst_segments(self.tint), [])

    def test_load_brst_segments_http_error(self):
        response = _brst_segments_response([], status_code=404)
        with mock.patch.object(requests, "get", return_value=response):
            with self.assertRaises(requests.HTTPError):
                mms.load_brst_segments(self.tint)

    def test_load_brst_segments_deprecated_args(self):
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response([])
        ):
            with self.assertWarns(FutureWarning):
                mms.load_brst_segments(self.tint, data_path="/tmp", download=True)


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
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=2),
            isotropic,
        )
        self.assertIsInstance(result, xr.Dataset)

    @staticmethod
    def _skymap(species, e_min, e_max, n_t=3):
        # FPI-like skymap: 32 log-spaced energies, 32 phi, 16 theta
        energy = np.tile(np.geomspace(e_min, e_max, 32), (n_t, 1))
        phi = np.tile(5.625 + 11.25 * np.arange(32), (n_t, 1))
        theta = 5.625 + 11.25 * np.arange(16)
        time = generate_timeline(1 / 0.15, n_t)
        return pyrf.ts_skymap(
            time,
            np.ones((n_t, 32, 32, 16)),
            energy,
            phi,
            theta,
            energy0=energy[0],
            energy1=energy[0],
            esteptable=np.zeros(n_t),
            attrs={"UNITS": "s^3/cm^6"},
            glob_attrs={"species": species},
        )

    @staticmethod
    def _moments(time, n_cc, v_kms, t_par, t_perp):
        # B along z, so that T = diag(T_perp, T_perp, T_par) in xyz
        n_t = len(time)
        t_xyz = np.zeros((n_t, 3, 3))
        t_xyz[:, 0, 0], t_xyz[:, 1, 1], t_xyz[:, 2, 2] = t_perp, t_perp, t_par
        return (
            pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 10.0], (n_t, 1))),
            pyrf.ts_scalar(time, np.full(n_t, n_cc)),
            pyrf.ts_vec_xyz(time, np.tile(v_kms, (n_t, 1))),
            pyrf.ts_tensor_xyz(time, t_xyz),
        )

    @staticmethod
    def _analytic(vdf, mass, sc_pot, n_cc, v_kms, t_par, t_perp):
        # Bi-Maxwellian (s^3/km^6) at the particle velocities (minus the look
        # directions) with B along z
        q_e = constants.elementary_charge
        speed = np.sqrt(2 * q_e * (vdf.energy.data - sc_pot) / mass)[..., None, None]
        phi = np.deg2rad(vdf.phi.data)[:, None, :, None]
        theta = np.deg2rad(vdf.theta.data)[None, None, None, :]
        v_x = -speed * np.sin(theta) * np.cos(phi)
        v_y = -speed * np.sin(theta) * np.sin(phi)
        v_z = -speed * np.cos(theta)
        v_d = np.array(v_kms) * 1e3
        vth_par2, vth_perp2 = [2 * q_e * t_ / mass for t_ in [t_par, t_perp]]
        f_ = n_cc * 1e6 / (np.pi**1.5 * np.sqrt(vth_par2) * vth_perp2)
        f_ = f_ * np.exp(
            -((v_x - v_d[0]) ** 2 + (v_y - v_d[1]) ** 2) / vth_perp2
            - (v_z - v_d[2]) ** 2 / vth_par2
        )
        return f_ * 1e18

    def test_make_model_vdf_phi_1d(self):
        # phi as (phi,) (IndexError): as with phi as (time, phi), phi unchanged
        vdf = self._skymap("ions", 10.0, 3e4)
        b_xyz, n_s, v_xyz, t_xyz = self._moments(
            vdf.time.data, 2.0, [100.0, 0.0, 0.0], 1000.0, 500.0
        )
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(3))
        expected = mms.make_model_vdf(vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz)

        vdf["phi"] = ("idx1", vdf.phi.data[0])
        result = mms.make_model_vdf(vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz)
        np.testing.assert_allclose(result.data.data, expected.data.data, rtol=1e-12)
        self.assertEqual(result.phi.dims, ("idx1",))

    def test_make_model_vdf_b_along_x(self):
        # B exactly along x gave an all-NaN model (through rotate_tensor). An
        # isotropic Maxwellian doesn't depend on the direction of B
        vdf = self._skymap("ions", 10.0, 3e4)
        b_z, n_s, v_xyz, t_xyz = self._moments(
            vdf.time.data, 2.0, [100.0, 0.0, 0.0], 1000.0, 1000.0
        )
        b_x = pyrf.ts_vec_xyz(vdf.time.data, np.tile([10.0, 0.0, 0.0], (3, 1)))
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(3))
        result = mms.make_model_vdf(vdf, b_x, sc_pot, n_s, v_xyz, t_xyz)
        expected = mms.make_model_vdf(vdf, b_z, sc_pot, n_s, v_xyz, t_xyz)
        self.assertTrue(np.any(np.isfinite(result.data.data)))
        np.testing.assert_allclose(result.data.data, expected.data.data, rtol=1e-10)

    def _check(self, species, mass, e_range, sc_pot, v_kms, t_par, t_perp, isotropic):
        vdf = self._skymap(species, *e_range)
        b_xyz, n_s, v_xyz, t_xyz = self._moments(
            vdf.time.data, 2.0, v_kms, t_par, t_perp
        )
        sc_pot_ts = pyrf.ts_scalar(vdf.time.data, np.full(len(vdf.time), sc_pot))

        # No warning (sqrt of negative energies, rotate_tensor "pp" 0/0 for a
        # gyrotropic tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = mms.make_model_vdf(
                vdf, b_xyz, sc_pot_ts, n_s, v_xyz, t_xyz, isotropic
            )

        self.assertEqual(result.data.attrs["UNITS"], "s^3/km^6")
        self.assertEqual(vdf.data.attrs["UNITS"], "s^3/cm^6")

        if isotropic:
            t_par = t_perp = (t_par + 2 * t_perp) / 3

        # Ions: the potential is subtracted with the opposite sign
        sc_pot = sc_pot if species == "electrons" else -sc_pot
        expected = self._analytic(vdf, mass, sc_pot, 2.0, v_kms, t_par, t_perp)
        above = (vdf.energy.data - sc_pot) >= 0
        np.testing.assert_allclose(result.data.data[above], expected[above], rtol=1e-10)

        return result, above

    def test_make_model_vdf_maxwellian(self):
        self._check(
            "ions",
            constants.proton_mass,
            (10.0, 3e4),
            0.0,
            [200.0, -100.0, 300.0],
            1000.0,
            1000.0,
            False,
        )

    def test_make_model_vdf_bi_maxwellian(self):
        for isotropic in [False, True]:
            self._check(
                "ions",
                constants.proton_mass,
                (10.0, 3e4),
                0.0,
                [200.0, -100.0, 300.0],
                2000.0,
                500.0,
                isotropic,
            )

    def test_make_model_vdf_velocity_along_b(self):
        # v_perp = 0 gave 0 / 0 and an all-NaN model
        for v_kms in [[0.0, 0.0, 300.0], [0.0, 0.0, 0.0]]:
            result, _ = self._check(
                "ions",
                constants.proton_mass,
                (10.0, 3e4),
                0.0,
                v_kms,
                2000.0,
                500.0,
                False,
            )
            self.assertTrue(np.all(np.isfinite(result.data.data)))

    def test_make_model_vdf_below_sc_pot(self):
        # Electrons with a 5 V potential: NaN below 5 eV (no velocity), without
        # warnings, and no effect on epsilon (zero weight in v^2 dv)
        result, above = self._check(
            "electrons",
            constants.electron_mass,
            (1.0, 1e3),
            5.0,
            [100.0, 0.0, 50.0],
            100.0,
            100.0,
            False,
        )
        self.assertTrue(np.all(np.isnan(result.data.data[~above])))
        self.assertTrue(np.any(~above))

        # Data equal to the model above the potential, anything below it
        vdf = result.copy()
        data = result.data.data.copy()
        data[~above] = 1e3 * np.nanmax(data)
        vdf["data"] = result.data.copy(data=data)
        n_s = pyrf.ts_scalar(vdf.time.data, np.full(len(vdf.time), 2.0))
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.full(len(vdf.time), 5.0))
        eps = mms.calculate_epsilon(vdf, result, n_s, sc_pot)
        np.testing.assert_allclose(eps.data, 0.0, atol=1e-12)

    def test_make_model_vdf_invalid(self):
        vdf = self._skymap("bazinga", 10.0, 3e4)
        b_xyz, n_s, v_xyz, t_xyz = self._moments(
            vdf.time.data, 2.0, [0.0] * 3, 1e3, 1e3
        )
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(len(vdf.time)))

        with self.assertRaises(ValueError):
            mms.make_model_vdf(vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz)

        vdf.attrs["species"] = "ions"
        with self.assertRaises(ValueError):
            mms.make_model_vdf(vdf, b_xyz, sc_pot, n_s[:2], v_xyz, t_xyz)


def _gyro_skymap(values=None, n_t=2):
    # FPI-like skymap (2 energies, 32 phi, 16 theta) of the look directions,
    # filled with values(phi, theta) of the look directions (degrees)
    phi = 5.625 + 11.25 * np.arange(32)
    theta = 5.625 + 11.25 * np.arange(16)
    data = np.ones((n_t, 2, 32, 16))
    if values is not None:
        data *= values(phi[:, None], theta[None, :])[None, None]
    time = generate_timeline(1.0, n_t)
    vdf = pyrf.ts_skymap(
        time,
        data,
        np.tile([100.0, 200.0], (n_t, 1)),
        np.tile(phi, (n_t, 1)),
        theta,
        attrs={"UNITS": "s^3/cm^6"},
        glob_attrs={"species": "ions"},
    )
    return vdf


def _b_const(b_xyz, n_t=2):
    return pyrf.ts_vec_xyz(generate_timeline(1.0, n_t), np.tile(b_xyz, (n_t, 1)))


class GetGyrophaseDistTestCase(unittest.TestCase):
    def test_get_gyrophase_dist_gyrotropic(self):
        # f(pitch angle) with B along z: the same in every gyrophase bin. The
        # particle velocity is minus the look direction: pitch = 180 - theta
        vdf = _gyro_skymap(lambda p, t: 1.0 + np.cos(np.deg2rad(t)) ** 2 + 0 * p)
        result = mms.get_gyrophase_dist(vdf, _b_const([0.0, 0.0, 10.0]))

        self.assertTupleEqual(result.data.shape, (2, 2, 24))
        np.testing.assert_allclose(result.gyrophase.data[0], 7.5 + 15.0 * np.arange(24))
        flat = np.broadcast_to(result.data.data[..., :1], result.data.shape)
        np.testing.assert_allclose(result.data.data, flat)
        self.assertGreater(result.data.data[0, 0, 0], 1.0)

    def test_get_gyrophase_dist_sense(self):
        # A particle moving close to +y (look direction phi = 275.6, theta =
        # 95.6) has a gyrophase of 95.6 degrees right-handed about B along +z
        # from x (bin 6), and 264.4 degrees about B along -z (bin 17)
        def beam(p, t):
            return ((np.abs(p - 275.625) < 1) & (np.abs(t - 95.625) < 1)) * 1.0

        vdf = _gyro_skymap(beam)
        for b_z, i_bin in [(10.0, 6), (-10.0, 17)]:
            result = mms.get_gyrophase_dist(vdf, _b_const([0.0, 0.0, b_z]))
            nonzero = np.flatnonzero(result.data.data[0, 0] > 0)
            self.assertListEqual(nonzero.tolist(), [i_bin])

        # Reference along y: the gyrophase is rotated by -90 degrees, also
        # with a time series reference
        r_xyz = _b_const([0.0, 1.0, 0.0])
        for ref in [[0.0, 1.0, 0.0], r_xyz]:
            result = mms.get_gyrophase_dist(vdf, _b_const([0.0, 0.0, 10.0]), r_xyz=ref)
            self.assertListEqual(np.flatnonzero(result.data.data[0, 0]).tolist(), [0])

    def test_get_gyrophase_dist_convert_fac(self):
        # Oblique B: the bin of a beam is that of the gyrophase of its velocity
        # in the field-aligned coordinates of convert_fac
        def beam(p, t):
            return ((np.abs(p - 140.625) < 1) & (np.abs(t - 61.875) < 1)) * 1.0

        b_xyz = _b_const([3.0, -4.0, 5.0])
        result = mms.get_gyrophase_dist(_gyro_skymap(beam), b_xyz, angles=36)

        look = np.deg2rad([140.625, 61.875])
        u_xyz = -np.array(
            [
                np.cos(look[0]) * np.sin(look[1]),
                np.sin(look[0]) * np.sin(look[1]),
                np.cos(look[1]),
            ]
        )
        u_fac = pyrf.convert_fac(_b_const(u_xyz), b_xyz, [1, 0, 0]).data[0]
        gyrophase = np.rad2deg(np.arctan2(u_fac[1], u_fac[0])) % 360.0
        nonzero = np.flatnonzero(result.data.data[0, 0] > 0)
        self.assertListEqual(nonzero.tolist(), [int(gyrophase // 10)])

    def test_get_gyrophase_dist_pitch_angle_range(self):
        # Beams at pitch angles 95 and 30 degrees (theta 84.375 and 151.875)
        def beams(p, t):
            perp = (np.abs(p - 5.625) < 1) & (np.abs(t - 84.375) < 1)
            obl = (np.abs(p - 95.625) < 1) & (np.abs(t - 151.875) < 1)
            return perp * 1.0 + obl * 2.0

        vdf = _gyro_skymap(beams)
        b_z = _b_const([0.0, 0.0, 10.0])
        all_pa = mms.get_gyrophase_dist(vdf, b_z, angles=12)
        perp = mms.get_gyrophase_dist(vdf, b_z, angles=12, pitch_angle_range=[80, 100])

        # look phi 5.625 -> velocity at 185.6 (bin 6), 95.625 -> 275.6 (bin 9)
        self.assertListEqual(np.flatnonzero(all_pa.data.data[0, 0]).tolist(), [6, 9])
        self.assertListEqual(np.flatnonzero(perp.data.data[0, 0] > 0).tolist(), [6])
        np.testing.assert_array_equal(perp.attrs["pitch_angle_range"], [80, 100])

    def test_get_gyrophase_dist_input(self):
        vdf = _gyro_skymap()

        # Reference along B: no gyrophase
        result = mms.get_gyrophase_dist(
            vdf, _b_const([10.0, 0.0, 0.0]), r_xyz=[1.0, 0.0, 0.0]
        )
        self.assertTrue(np.isnan(result.data.data).all())

        for kwargs in [
            {"angles": "15"},
            {"pitch_angle_range": [0.0]},
            {"r_xyz": [1.0, 0.0]},
        ]:
            with self.assertRaises(ValueError):
                mms.get_gyrophase_dist(vdf, _b_const([0.0, 0.0, 10.0]), **kwargs)

    def test_get_gyrophase_dist_options(self):
        # Bin edges as an array, azimuths without time, and a time interval
        def beam(p, t):
            return ((np.abs(p - 275.625) < 1) & (np.abs(t - 95.625) < 1)) * 1.0

        vdf = _gyro_skymap(beam, n_t=4)
        b_z = _b_const([0.0, 0.0, 10.0], n_t=4)
        expected = mms.get_gyrophase_dist(vdf, b_z, angles=12)

        result = mms.get_gyrophase_dist(vdf, b_z, angles=np.linspace(0, 360, 13))
        np.testing.assert_array_equal(result.data.data, expected.data.data)

        vdf_1d = vdf.assign(phi=(["idx1"], vdf.phi.data[0]))
        result = mms.get_gyrophase_dist(vdf_1d, b_z, angles=12)
        np.testing.assert_array_equal(result.data.data, expected.data.data)

        tint = list(pyrf.datetime642iso8601(vdf.time.data[[1, 2]]))
        result = mms.get_gyrophase_dist(vdf, b_z, tint=tint, angles=12)
        np.testing.assert_array_equal(result.time.data, vdf.time.data[1:3])
        np.testing.assert_array_equal(result.data.data, expected.data.data[1:3])

    def test_get_gyrophase_dist_feeps(self):
        # FEEPS skymap (4 rings with angular widths in the attributes)
        feeps_alle = _feeps_skymap_case("brst", 128)
        sky = mms.feeps_skymap(feeps_alle, to_psd=False)
        b_z = pyrf.ts_vec_xyz(sky.time.data, np.tile([0.0, 0.0, 10.0], (2, 1)))

        result = mms.get_gyrophase_dist(sky, b_z, angles=8)
        np.testing.assert_allclose(result.data.data, 2.0)


class GetPitchAngleDistTestCase(unittest.TestCase):
    def setUp(self):
        # f = (i_energy + 1) (i_theta + 1) on FPI-like angles with B along z:
        # the velocity (minus the look direction) has pitch angle 180 - theta
        self.vdf = generate_vdf(64.0, 4, [32, 32, 16], units="s^3/cm^6")
        self.vdf["phi"] = 5.625 + 11.25 * self.vdf.phi
        self.vdf["theta"] = 5.625 + 11.25 * self.vdf.theta
        e_idx, t_idx = np.arange(32.0)[:, None, None], np.arange(16.0)[None, None, :]
        self.vdf.data.data[:] = (e_idx + 1) * (t_idx + 1) * np.ones((32, 32, 16))
        self.b_xyz = pyrf.ts_vec_xyz(
            self.vdf.time.data, np.tile([0.0, 0.0, 10.0], (4, 1))
        )

        # Default bins: [0, 15], [15, 30], ..., [165, 180] degrees
        self.pitch = 180.0 - self.vdf.theta.data
        self.in_bin = [
            (self.pitch >= lo) & (self.pitch <= lo + 15) for lo in range(0, 180, 15)
        ]

    def _expected(self, method):
        g_theta = np.arange(16.0) + 1
        sin_t = np.sin(np.deg2rad(self.vdf.theta.data))
        per_bin = {
            "mean": [np.mean(g_theta[m]) for m in self.in_bin],
            "sum": [32 * np.sum(g_theta[m]) for m in self.in_bin],
            "sum_weighted": [
                np.sum(g_theta[m] * sin_t[m]) / np.sum(sin_t[m]) for m in self.in_bin
            ],
        }[method]
        return (np.arange(32.0) + 1)[:, None] * np.array(per_bin)[None, :]

    def test_get_pitch_angle_dist_values(self):
        for method in ["mean", "sum", "sum_weighted"]:
            pad = mms.get_pitch_angle_dist(
                self.vdf, self.b_xyz, verbose=False, meanorsum=method
            )

            self.assertEqual(pad.data.shape, (4, 32, 12))
            np.testing.assert_allclose(
                pad.data.data, np.broadcast_to(self._expected(method), (4, 32, 12))
            )
            np.testing.assert_allclose(pad.theta.data[0], np.arange(7.5, 180, 15))
            self.assertEqual(pad.attrs["mean_or_sum"], method)

        self.assertNotIn("mean_or_sum", self.vdf.attrs)

    def test_get_pitch_angle_dist_1d_phi(self):
        # 1-D phi gave a coordinate with the length of time
        vdf = self.vdf.copy()
        vdf["phi"] = xr.DataArray(self.vdf.phi.data[0], dims=["idx1"])
        pad = mms.get_pitch_angle_dist(vdf, self.b_xyz, verbose=False)
        np.testing.assert_allclose(pad.data.data[0], self._expected("mean"))

    def test_get_pitch_angle_dist_one_time(self):
        # np.squeeze dropped the time dimension
        tint = list(pyrf.datetime642iso8601(self.vdf.time.data[[1, 1]]))
        for vdf, tint_ in [(self.vdf.isel(time=[0]), None), (self.vdf, tint)]:
            pad = mms.get_pitch_angle_dist(vdf, self.b_xyz, tint_, verbose=False)
            self.assertEqual(pad.data.shape, (1, 32, 12))
            np.testing.assert_allclose(pad.data.data[0], self._expected("mean"))

    def test_get_pitch_angle_dist_energy_tables(self):
        # Each time step keeps its energy table (they were averaged)
        vdf = generate_vdf(64.0, 4, [32, 32, 16], energy01=True, units="s^3/cm^6")
        vdf["phi"], vdf["theta"] = self.vdf.phi, self.vdf.theta
        pad = mms.get_pitch_angle_dist(vdf, self.b_xyz, verbose=False)
        np.testing.assert_array_equal(pad.energy.data, vdf.energy.data)

    def test_get_pitch_angle_dist_invalid(self):
        with self.assertRaises(ValueError):
            mms.get_pitch_angle_dist(self.vdf, self.b_xyz, meanorsum="bazinga")

        with self.assertRaises(ValueError):
            mms.get_pitch_angle_dist(self.vdf, self.b_xyz, angles="bazinga")


class HpcaEnergiesTestCase(unittest.TestCase):
    def test_hpca_energies_output(self):
        result = mms.hpca_energies()
        self.assertIsInstance(result, list)


@ddt
class HpcaPadTestCase(unittest.TestCase):
    N_AZ, N_PO, N_EN, D_T = 16, 16, 4, 0.625

    def _inputs(self, aze_deg, n_spins=2, centred=True, b_dir=None):
        # vdf = 1 at 16 samples per half-spin (start azimuths 0..15), all anodes
        # at 90 deg (equatorial), and aze records at the start or centre of the
        # half-spins (pyrfu.mms.get_data centres the times: 7.5 samples later)
        n_ti = n_spins * self.N_AZ
        t_0 = np.datetime64("2019-09-14T07:54:00", "ns")
        times = t_0 + (np.arange(n_ti) * self.D_T * 1e9).astype("timedelta64[ns]")
        energy = np.arange(1.0, self.N_EN + 1)
        vdf = xr.DataArray(
            np.ones((n_ti, self.N_PO, self.N_EN)),
            coords=[times, np.arange(self.N_PO), energy],
            dims=["time", "rcomp", "ccomp"],
        )
        saz = xr.DataArray(np.arange(n_ti) % self.N_AZ, coords=[times], dims=["time"])
        offset = 7.5 * self.D_T * 1e9 if centred else 0.0
        t_aze = times[:: self.N_AZ] + np.timedelta64(int(offset), "ns")
        aze = xr.DataArray(
            np.broadcast_to(
                aze_deg[..., None, None], (n_spins, self.N_AZ, self.N_PO, self.N_EN)
            ).copy(),
            coords=[t_aze, np.arange(self.N_AZ), np.arange(self.N_PO), energy],
            dims=["time", "az", "po", "en"],
        )

        # B at 128 Hz, along b_dir(t) (default +x)
        t_b = generate_timeline(
            128.0, int(n_ti * self.D_T * 128) + 256, ref_time="2019-09-14T07:53:59"
        )
        t_s = (t_b - t_0) / np.timedelta64(1, "s")
        b_data = (
            b_dir(t_s) if b_dir is not None else np.tile([1.0, 0.0, 0.0], (len(t_b), 1))
        )
        b_xyz = pyrf.ts_vec_xyz(t_b, b_data)

        return vdf, saz, aze, b_xyz

    def _pad(self, *args, **kwargs):
        return mms.hpca_pad(*args, elevation=np.full(self.N_PO, 90.0), **kwargs)

    @data(True, False)
    def test_hpca_pad_look_direction_and_pairing(self, centred):
        # Looking along -x (azimuth 180) with B along +x: particles along B,
        # pitch angle 0; looking along +x (azimuth 0): 180. The look direction
        # alternates with the azimuth step in half-spin 0 and is 0 in half-spin 1,
        # so each sample must get the azimuths of its own half-spin and step.
        aze_deg = np.zeros((2, self.N_AZ))
        aze_deg[0, ::2] = 180.0
        pad = self._pad(*self._inputs(aze_deg, centred=centred))

        is_par = np.zeros(32, dtype=bool)
        is_par[0:16:2] = True
        np.testing.assert_array_equal(pad.data[is_par, 0], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[is_par, 1:])))
        np.testing.assert_array_equal(pad.data[~is_par, -1], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[~is_par, :-1])))

    @data((0.0, 1.0, 0), (180.0, 1.0, -1), (0.0, -1.0, -1))
    @unpack
    def test_hpca_pad_polar_angle(self, polar, b_z, pa_bin):
        # The elevations are the polar angles of the particle velocities (not
        # flipped as the azimuths): polar 0 with B along +z is pitch angle 0
        def b_dir(t_s):
            return np.stack([0 * t_s, 0 * t_s, b_z + 0 * t_s], axis=1)

        inputs = self._inputs(np.full((2, self.N_AZ), 37.0), b_dir=b_dir)
        pad = mms.hpca_pad(*inputs, elevation=np.full(self.N_PO, polar))

        np.testing.assert_array_equal(pad.data[:, pa_bin], 1.0)

    def test_hpca_pad_b_time(self):
        # B along +x in half-spin 0 and along -x in half-spin 1, looking along -x:
        # pitch angle 0 then 180 (B was paired with the wrong samples)
        def b_dir(t_s):
            sign = np.where(t_s < 16 * self.D_T, 1.0, -1.0)
            return np.stack([sign, np.zeros_like(t_s), np.zeros_like(t_s)], axis=1)

        aze_deg = np.full((2, self.N_AZ), 180.0)
        pad = self._pad(*self._inputs(aze_deg, b_dir=b_dir))

        np.testing.assert_array_equal(pad.data[1:15, 0], 1.0)
        np.testing.assert_array_equal(pad.data[17:31, -1], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[1:15, -1])))

    def test_hpca_pad_energy_range(self):
        # Both ends of elim are included (the top channel was dropped)
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))
        vdf.data[...] = vdf.ccomp.data  # value = energy
        pad = self._pad(vdf, saz, aze, b_xyz, elim=[2.0, 3.0])

        np.testing.assert_allclose(pad.data[:, 0], 2.5)

    def test_hpca_pad_mismatch(self):
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))

        # aze shifted by 3 samples: not the start nor the centre of a half-spin
        aze_shifted = aze.assign_coords(
            time=aze.time.data + np.timedelta64(int(3 * self.D_T * 1e9), "ns")
        )
        with self.assertRaises(ValueError):
            self._pad(vdf, saz, aze_shifted, b_xyz)

        # start azimuths of a half-spin not 0..15
        saz_bad = saz.copy()
        saz_bad.data[20] = 7
        with self.assertRaises(ValueError):
            self._pad(vdf, saz_bad, aze, b_xyz)

    def test_hpca_pad_elevation_from_file(self):
        # mms?_hpca_centroid_elevation_angle read from the dataset of vdf
        module = importlib.import_module("pyrfu.mms.hpca_pad")
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))
        vdf.attrs["GLOBAL"] = {"Logical_source": "mms1_hpca_brst_l2_ion"}
        elevation = xr.DataArray(np.full((1, self.N_PO), 90.0), dims=["x", "y"])

        with mock.patch.object(
            module, "db_get_variable", return_value=elevation
        ) as read:
            pad = mms.hpca_pad(vdf, saz, aze, b_xyz, source="aws")

        self.assertEqual(
            read.call_args.args[:2],
            ("mms1_hpca_brst_l2_ion", "mms1_hpca_centroid_elevation_angle"),
        )
        self.assertEqual(read.call_args.kwargs["source"], "aws")
        np.testing.assert_array_equal(pad.data[:, 0], 1.0)

        del vdf.attrs["GLOBAL"]
        with self.assertRaises(ValueError):
            mms.hpca_pad(vdf, saz, aze, b_xyz)


@ddt
class GetHpcaDistTestCase(unittest.TestCase):
    # 4 azimuth steps per half-spin, 3 anodes, 2 energies, 1 s samples
    N_AZ, N_PO, N_EN = 4, 3, 2
    ELEVATION = np.array([123.75, 101.25, 78.75])

    def _inputs(self, centred=True):
        # 20 samples, half-spins from samples 2, 6, 10, 14 and 18 (incomplete),
        # without azimuths in half-spin 2; value 100 * sample + 10 * anode +
        # energy, azimuth 90 * step + half-spin + 1
        t_0 = np.datetime64("2019-09-14T07:54:00", "ns")
        offset = 0.5 if centred else 0.0  # middle of the accumulations
        t_vdf = t_0 + ((np.arange(20) + offset) * 1e9).astype("timedelta64[ns]")
        sample, anode, energy = np.meshgrid(
            np.arange(20), np.arange(self.N_PO), np.arange(self.N_EN), indexing="ij"
        )
        vdf = xr.DataArray(
            100.0 * sample + 10 * anode + energy,
            coords=[t_vdf, np.arange(self.N_PO), [1.0, 2.0]],
            dims=["time", "rcomp", "ccomp"],
            attrs={"FIELDNAM": "Hydrogen+ Phase Space Density"},
        )

        # Half-spin centres (1.5 samples after the first one) or starts
        offset = 1.5 if centred else 0.0
        t_aze = t_vdf[2::4] + np.timedelta64(int(offset * 1e9), "ns")
        aze_deg = 90.0 * np.arange(self.N_AZ)[None, :] + np.arange(1, 6)[:, None]
        aze_deg[2] = 0.0
        aze = xr.DataArray(
            np.broadcast_to(
                aze_deg[..., None, None], (5, self.N_AZ, self.N_PO, self.N_EN)
            ).copy(),
            coords=[t_aze, np.arange(self.N_AZ), np.arange(self.N_PO), [0, 1]],
            dims=["time", "az", "po", "en"],
        )

        return vdf, aze

    @data(True, False)
    def test_get_hpca_dist_half_spins(self, centred):
        # The data of a half-spin are the samples from its first one (they were
        # shifted by one sample); NaN without azimuths or not complete in vdf
        vdf, aze = self._inputs(centred)
        out = mms.get_hpca_dist(vdf, aze, elevation=self.ELEVATION)

        self.assertEqual(out["n_times"], 5)
        np.testing.assert_array_equal(out["time"], aze.time.data)

        for i, i_start in zip([0, 1, 3], [2, 6, 14]):
            np.testing.assert_array_equal(
                out["data"][i], np.transpose(vdf.data[i_start : i_start + 4], [2, 0, 1])
            )
            np.testing.assert_array_equal(
                out["phi"][i, 0, :, 0], 90.0 * np.arange(self.N_AZ) + i + 1
            )

        np.testing.assert_array_equal(out["dphi"][[0, 1, 3]], 90.0)

        for key in ["data", "phi", "dphi"]:
            self.assertEqual(out[key].shape, (5, self.N_EN, self.N_AZ, self.N_PO))
            self.assertTrue(np.all(np.isnan(out[key][[2, 4]])))

    def test_get_hpca_dist_angles(self):
        # Elevations of the anodes (they were the anode numbers), with the
        # dimensions of a distribution (energy, azimuth, elevation)
        vdf, aze = self._inputs()
        out = mms.get_hpca_dist(vdf, aze, elevation=self.ELEVATION)
        dim = (self.N_EN, self.N_AZ, self.N_PO)

        for key in ["theta", "energy", "dtheta", "bins", "denergy"]:
            self.assertEqual(out[key].shape, dim)

        np.testing.assert_array_equal(out["theta"][0, 0], [-33.75, -11.25, 11.25])
        np.testing.assert_array_equal(out["energy"][:, 0, 0], [1.0, 2.0])
        np.testing.assert_array_equal(out["bins"], 1.0)
        self.assertEqual((out["species"], out["charge"]), ("hydrogen+", 1))

    def test_get_hpca_dist_elevation_from_file(self):
        module = importlib.import_module("pyrfu.mms.get_hpca_dist")
        vdf, aze = self._inputs()

        with mock.patch.object(
            module, "_hpca_elevations", return_value=self.ELEVATION
        ) as read:
            out = mms.get_hpca_dist(vdf, aze, source="aws")

        self.assertIs(read.call_args.args[0], vdf)
        self.assertEqual(read.call_args.args[1:], ("aws", ""))
        np.testing.assert_array_equal(out["theta"][0, 0], [-33.75, -11.25, 11.25])

    def test_get_hpca_dist_errors(self):
        vdf, aze = self._inputs()

        # not monotonic
        aze_bad = aze.assign_coords(time=aze.time.data[[1, 0, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            mms.get_hpca_dist(vdf, aze_bad, elevation=self.ELEVATION)

        # without azimuths
        with self.assertRaisesRegex(ValueError, "no complete half-spin"):
            mms.get_hpca_dist(vdf, aze * 0, elevation=self.ELEVATION)

        # times out of vdf
        aze_bad = aze.assign_coords(time=aze.time.data + np.timedelta64(100, "s"))
        with self.assertRaisesRegex(ValueError, "no complete half-spin"):
            mms.get_hpca_dist(vdf, aze_bad, elevation=self.ELEVATION)


@ddt
class MakeModelKappaTestCase(unittest.TestCase):
    @staticmethod
    def _moments(vdf, n_s, v_kms, t_ev):
        # Constant moments on the times of vdf
        time = vdf.time.data
        n_t = len(time)
        return (
            pyrf.ts_scalar(time, np.full(n_t, n_s)),
            pyrf.ts_vec_xyz(time, np.tile(np.asarray(v_kms, dtype=float), (n_t, 1))),
            pyrf.ts_scalar(time, np.full(n_t, t_ev)),
        )

    @data(("bazinga", 7.0), ("ions", 1.5), ("ions", 1.0))
    @unpack
    def test_make_model_kappa_input(self, species, kappa):
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 500.0, [0.0] * 3)
        vdf.attrs["species"] = species

        with self.assertRaises(ValueError):
            mms.make_model_kappa(vdf, *self._moments(vdf, 1.0, [0.0] * 3, 500.0), kappa)

    @data(5.0, 20.0)
    def test_make_model_kappa_moments(self, kappa):
        # psd_moments of the model gives back n, V and T. The model was
        # centred at V / 1000 (km/s taken as m/s), 1e6 too small (cm^-3) and
        # along the look directions
        v_kms = [100.0, -50.0, 200.0]
        vdf, sc_pot = PsdMomentsTestCase._drifting_maxwellian(1.0, 500.0, [0.0] * 3)
        moments = self._moments(vdf, 2.0, v_kms, 500.0)
        model = mms.make_model_kappa(vdf, *moments, kappa)
        self.assertEqual(model.data.attrs["UNITS"], "s^3/m^6")

        n, v, _, _, t, _ = mms.psd_moments(model, sc_pot)
        np.testing.assert_allclose(n.data, 2.0, rtol=0.01)
        np.testing.assert_allclose(v.data, moments[1].data, atol=2.0)
        np.testing.assert_allclose(
            np.trace(t.data, axis1=1, axis2=2) / 3, 500.0, rtol=0.01
        )

    def test_make_model_kappa_maxwellian_limit(self):
        # As kappa -> infinity, the model tends to the Maxwellian of
        # make_model_vdf, as 1 / kappa (2e-4 at kappa = 1e4)
        v_kms = [100.0, -50.0, 200.0]
        vdf, sc_pot = PsdMomentsTestCase._drifting_maxwellian(1.0, 500.0, [0.0] * 3)
        n_s, v_xyz, t_s = self._moments(vdf, 2.0, v_kms, 500.0)
        time = vdf.time.data
        b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 10.0], (len(time), 1)))
        t_xyz = pyrf.ts_tensor_xyz(time, np.tile(np.eye(3) * 500.0, (len(time), 1, 1)))
        maxwellian = mms.make_model_vdf(
            vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz, isotropic=True
        )
        f_max = maxwellian.data.data * 1e-18  # s^3/km^6 -> s^3/m^6

        model = mms.make_model_kappa(vdf, n_s, v_xyz, t_s, kappa=1e4)
        error = np.nanmax(np.abs(model.data.data - f_max)) / np.nanmax(f_max)
        self.assertLess(error, 1e-3)

    def test_make_model_kappa_phi_per_time(self):
        # Each time uses its own azimuths (FPI burst); the first time's were
        # used for all times
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 500.0, [0.0] * 3)
        vdf.phi.data[1] += 5.0
        moments = self._moments(vdf, 2.0, [300.0, 0.0, 0.0], 500.0)
        model = mms.make_model_kappa(vdf, *moments)

        swapped = vdf.isel(time=[1, 0]).assign_coords(time=vdf.time.data)
        model_swapped = mms.make_model_kappa(swapped, *moments)
        np.testing.assert_allclose(model.data.data[1], model_swapped.data.data[0])
        change = np.abs(model.data.data[0] - model.data.data[1])
        self.assertGreater(np.max(change) / np.max(model.data.data[0]), 0.01)

    @data(("electrons", 1.0), ("ions", -1.0))
    @unpack
    def test_make_model_kappa_sc_pot(self, species, sign):
        # The energies are corrected as in make_model_vdf (electrons lose,
        # ions gain e V); the electron channels below the potential are NaN
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 50.0, [0.0] * 3)
        vdf.attrs["species"] = species
        time = vdf.time.data
        sc_pot = pyrf.ts_scalar(time, np.full(len(time), 20.0))
        moments = self._moments(vdf, 2.0, [0.0] * 3, 50.0)
        model = mms.make_model_kappa(vdf, *moments, sc_pot=sc_pot)

        shifted = vdf.copy(deep=True)
        shifted.energy.data[:] = vdf.energy.data - sign * 20.0
        valid = shifted.energy.data[0] >= 0
        shifted.energy.data[:] = np.clip(shifted.energy.data, 0.0, None)
        expected = mms.make_model_kappa(shifted, *moments)
        np.testing.assert_allclose(
            model.data.data[:, valid], expected.data.data[:, valid]
        )
        self.assertTrue(np.all(np.isnan(model.data.data[:, ~valid])))
        self.assertEqual(np.any(~valid), sign > 0)


class Mms4DisplacementTestCase(unittest.TestCase):
    # Tetrahedron (km) and B along z: R_par is the z separation
    offsets = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 2.0], [3.0, 8.0, -4.0], [-5.0, 6.0, 7.0]]
    )

    def _constant(self, time, value):
        return pyrf.ts_vec_xyz(time, np.tile(value, (len(time), 1)))

    def test_mms4_displacement_values(self):
        time = generate_timeline(16.0, 32)
        b_mms = [self._constant(time, [0.0, 0.0, 20.0]) for _ in range(4)]
        r_mms = [self._constant(time, o) for o in self.offsets]

        r_par, r_perp, theta_br = mms.mms4_displacement(b_mms, r_mms)

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        r_ij = np.array([self.offsets[i] - self.offsets[j] for i, j in pairs])
        np.testing.assert_allclose(r_par.data[0], r_ij[:, 2])
        np.testing.assert_allclose(r_perp.data[0], np.hypot(r_ij[:, 0], r_ij[:, 1]))
        np.testing.assert_allclose(
            theta_br.data[0],
            np.rad2deg(np.arccos(r_ij[:, 2] / np.linalg.norm(r_ij, axis=1))),
        )
        self.assertListEqual(
            list(r_par.pair.data), ["12", "13", "14", "23", "24", "34"]
        )
        np.testing.assert_array_equal(r_par.time.data, time)
        self.assertEqual(r_par.attrs["UNITS"], "km")
        self.assertEqual(theta_br.attrs["UNITS"], "deg")

    def test_mms4_displacement_pair_average_field(self):
        # MMS1 B along z and MMS2 B along x: pair 12 uses B along (1, 0, 1)
        time = generate_timeline(16.0, 32)
        b_mms = [self._constant(time, [0.0, 0.0, 20.0]) for _ in range(4)]
        b_mms[1] = self._constant(time, [20.0, 0.0, 0.0])
        r_mms = [self._constant(time, o) for o in self.offsets]

        r_par, r_perp, _ = mms.mms4_displacement(b_mms, r_mms)

        r_12 = self.offsets[0] - self.offsets[1]
        b_hat = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(r_par.data[:, 0], r_12 @ b_hat)
        np.testing.assert_allclose(
            r_perp.data[:, 0], np.linalg.norm(r_12 - (r_12 @ b_hat) * b_hat)
        )

    def test_mms4_displacement_resamples_to_mms1_field(self):
        # Positions at 1 Hz drifting along z at 2 km/s, B at 16 Hz: R is
        # interpolated to the B times of MMS1
        time_r = generate_timeline(1.0, 10)
        time_b = generate_timeline(16.0, 64)
        t_r = (time_r - time_r[0]).astype(np.int64) * 1e-9
        r_mms = [
            pyrf.ts_vec_xyz(time_r, o + np.outer(t_r * (k + 1), [0.0, 0.0, 2.0]))
            for k, o in enumerate(self.offsets)
        ]
        b_mms = [self._constant(time_b, [0.0, 0.0, 20.0]) for _ in range(4)]

        r_par, _, _ = mms.mms4_displacement(b_mms, r_mms)

        t_b = (time_b - time_r[0]).astype(np.int64) * 1e-9
        expected_12 = self.offsets[0, 2] - self.offsets[1, 2] + 2.0 * t_b * (1 - 2)
        np.testing.assert_allclose(r_par.data[:, 0], expected_12, atol=1e-9)
        np.testing.assert_array_equal(r_par.time.data, time_b)

    def test_mms4_displacement_input(self):
        time = generate_timeline(16.0, 32)
        b_mms = [self._constant(time, [0.0, 0.0, 20.0]) for _ in range(3)]
        r_mms = [self._constant(time, o) for o in self.offsets[:3]]
        with self.assertRaises(ValueError):
            mms.mms4_displacement(b_mms, r_mms)


@ddt
class PadPalimTestCase(unittest.TestCase):
    def test_pad_palim(self):
        data = np.random.random((3, 2, 12))
        data[0, 0, 5] = np.nan
        pad = _pad_case(data)

        # Range: strictly inside
        result = mms.pad_palim(pad, [30.0, 90.0])
        np.testing.assert_array_equal(result.theta.data[0], [37.5, 52.5, 67.5, 82.5])
        np.testing.assert_array_equal(result.data.data, data[..., 2:6])
        np.testing.assert_array_equal(result.attrs["delta_pitchangle_minus"], 7.5)

        # 90: 82.5 and 97.5 are equally close, averaged (NaN-mean)
        result = mms.pad_palim(pad, 90.0)
        self.assertEqual(result.data.shape, (3, 2, 1))
        np.testing.assert_array_equal(result.theta.data, 90.0)
        expected = np.nanmean(data[..., 5:7], axis=2)
        np.testing.assert_allclose(result.data.data[..., 0], expected)
        self.assertEqual(result.attrs["delta_pitchangle_minus"][0], 15.0)
        self.assertEqual(result.attrs["delta_pitchangle_plus"][0], 15.0)

        # Without averaging (irfu-matlab "noav"), and the closest bin
        result = mms.pad_palim(pad, 90.0, average=False)
        np.testing.assert_array_equal(result.theta.data[0], [82.5, 97.5])
        result = mms.pad_palim(pad, 50.0)
        np.testing.assert_array_equal(result.theta.data, 52.5)
        np.testing.assert_array_equal(result.data.data[..., 0], data[..., 3])

        with self.assertRaises(ValueError):
            mms.pad_palim(pad, [10.0, 20.0, 30.0])

        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")

        with self.assertRaises(ValueError):
            mms.pad_palim(vdf, 90.0)


@ddt
class PadDiffusionTestCase(unittest.TestCase):
    def test_pad_diffusion_legendre(self):
        # f = cos(a), constant D: df/dt = -2 D cos(a), and with the central
        # differences on a grid of step h: -2 D cos(a) sin(h)/h sin(2h)/(2h)
        alpha = np.deg2rad(7.5 + 15.0 * np.arange(12))
        pad = _pad_case(np.tile(np.cos(alpha), (2, 3, 1)))
        result = mms.pad_diffusion(pad, 0.1)
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6/s")

        step = np.deg2rad(15.0)
        factor = np.sin(step) / step * np.sin(2 * step) / (2 * step)
        expected = -0.2 * np.cos(alpha[2:-2]) * factor
        np.testing.assert_allclose(
            result.data.data[..., 2:-2], np.broadcast_to(expected, (2, 3, 8))
        )

        # f = P2(cos(a)) on a fine grid: -6 D f
        pitch = 0.25 + 0.5 * np.arange(360)
        p_2 = (3 * np.cos(np.deg2rad(pitch)) ** 2 - 1) / 2
        pad = _pad_case(np.tile(p_2, (2, 3, 1)), pitch=pitch)
        result = mms.pad_diffusion(pad, 0.1).data.data[..., 10:-10]
        np.testing.assert_allclose(
            result, np.broadcast_to(-0.6 * p_2[10:-10], result.shape), atol=2e-4
        )

    def test_pad_diffusion_coefficient(self):
        # D (energy, pitch angle), linear in D, irfu-matlab edges
        alpha = np.deg2rad(7.5 + 15.0 * np.arange(12))
        pad = _pad_case(np.tile(np.cos(alpha), (2, 3, 1)))
        d_aa = np.array([0.1, 0.2, 0.3])[:, None] * np.ones(12)
        result = mms.pad_diffusion(pad, d_aa).data.data
        np.testing.assert_allclose(
            result[:, 2], 3 * mms.pad_diffusion(pad, 0.1).data.data[:, 0]
        )

        f_0, f_1 = np.cos(alpha[:2])
        flux_0 = 0.1 * np.sin(alpha[0]) * (f_1 - f_0) / (alpha[1] - alpha[0])
        flux_1 = (
            0.1
            * np.sin(alpha[1])
            * (np.cos(alpha[2]) - f_0)
            / (2 * (alpha[1] - alpha[0]))
        )
        expected = (flux_1 - flux_0) / (alpha[1] - alpha[0]) / np.sin(alpha[0])
        np.testing.assert_allclose(result[:, 0, 0], expected)

        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")

        with self.assertRaises(ValueError):
            mms.pad_diffusion(vdf, 0.1)


class ProbeFieldsTestCase(unittest.TestCase):
    # Probes 1-4 60 m from the centre along +x, -x, +y, -y, probes 5 and 6
    # 14.6 m along +z and -z (the frame of the probes)
    R_PROBES = np.array(
        [[60, 0, 0], [-60, 0, 0], [0, 60, 0], [0, -60, 0], [0, 0, 14.6], [0, 0, -14.6]]
    )

    @staticmethod
    def _v6(time, pot):
        return xr.DataArray(pot, coords=[time, np.arange(1, 7)], dims=["time", "probe"])

    def test_probe_fields_uniform_field(self):
        # In a uniform field E (V/m) with the spacecraft at -10 V, each field
        # Eij is the component of E along the direction from probe i to j
        rng = np.random.default_rng(0)
        e_xyz = rng.normal(size=(50, 3)) * 1e-3
        pot = -10.0 - e_xyz @ self.R_PROBES.T
        out = mms.probe_fields(
            self._v6(generate_timeline(8192.0, 50), pot), timing=None
        )

        np.testing.assert_allclose(out.v.data, pot)
        np.testing.assert_allclose(out.v_sc.data, -10.0)

        def along(i, j):
            unit = self.R_PROBES[j - 1] - self.R_PROBES[i - 1]
            return e_xyz @ (unit / np.linalg.norm(unit)) * 1e3

        expected = {
            "e_probe": [along(1, 2)] * 2 + [along(3, 4)] * 2 + [along(5, 6)] * 2,
            "e_diag": [along(1, 3), along(1, 4), along(2, 3), along(2, 4)],
            "e_pair": [along(1, 2), along(3, 4), along(5, 6)],
        }

        for name, columns in expected.items():
            np.testing.assert_allclose(out[name].data, np.transpose(columns))

        self.assertEqual(out.e_probe.attrs["UNITS"], "mV/m")
        self.assertEqual((out.l_spin, out.l_axial), (120.0, 29.2))
        self.assertAlmostEqual(out.d_diag, 60.0 * np.sqrt(2))

    def test_probe_fields_boom_shortening(self):
        # Effective lengths 96 m and 14.5 m (Pedersen et al. 1998)
        pot = np.random.default_rng(1).normal(size=(20, 6))
        v6 = self._v6(generate_timeline(8192.0, 20), pot)
        out = mms.probe_fields(v6, timing=None, boom_shortening=True)

        self.assertEqual((out.l_spin, out.l_axial), (96.0, 14.5))
        np.testing.assert_allclose(
            out.e_pair.sel(boom="12").data, (pot[:, 0] - pot[:, 1]) / 96.0 * 1e3
        )

    def test_probe_fields_timing(self):
        # Potentials sampled with the channel delays of the burst mode: the
        # DFT shift gives them at the times of V1 (exact for periodic
        # band-limited signals), the interpolation is that of
        # correct_edp_probe_timing
        n_t, f_s = 256, 8192.0
        t_s = np.arange(n_t) / f_s

        def true(t_k):
            # Potential of probe k at the times t_k: periodic on the window
            freqs = np.array([3, 17, 40, 90]) * f_s / n_t
            phases = np.arange(6)[:, None] * 0.7
            return np.sum(
                np.cos(2 * np.pi * freqs[None, :, None] * t_k + phases[:, :, None]),
                axis=1,
            )

        def at(delay_us):
            return true(t_s + delay_us * 1e-6)

        p_0, p_3, p_5 = at(0.0), at(7.629), at(15.259)
        p_12, p_34, p_56 = at(26.703), at(30.518), at(34.332)
        sampled = np.column_stack(
            [
                p_0[0],
                p_0[0] - (p_12[0] - p_12[1]),
                p_3[2],
                p_3[2] - (p_34[2] - p_34[3]),
                p_5[4],
                p_5[4] - (p_56[4] - p_56[5]),
            ]
        )
        v6 = self._v6(generate_timeline(f_s, n_t), sampled)

        # To the rounding of the time tags to the ns (at 8192 Hz)
        out = mms.probe_fields(v6)
        np.testing.assert_allclose(out.v.data, p_0.T, atol=1e-7)
        self.assertEqual(out.timing, "dft")

        out = mms.probe_fields(v6, timing="resample")
        np.testing.assert_array_equal(out.v.data, mms.correct_edp_probe_timing(v6).data)

    def test_probe_fields_errors(self):
        v6 = self._v6(generate_timeline(8192.0, 20), np.zeros((20, 6)))

        with self.assertRaises(TypeError):
            mms.probe_fields(v6.data)

        with self.assertRaisesRegex(ValueError, "six probes"):
            mms.probe_fields(v6[:, :4])

        with self.assertRaisesRegex(ValueError, "timing must be"):
            mms.probe_fields(v6, timing="linear")

        v6.data[3, 2] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN"):
            mms.probe_fields(v6)


class ProbeAlignTimesTestCase(unittest.TestCase):
    def test_probe_align_times_values(self):
        # 20 s spin (18 deg/s), B along x in the spin plane except for
        # 30 s <= t < 40 s where B is along z. Probe 1 (phase + 30 deg) is
        # within 25 deg of +-B for phase mod 180 in [125, 175] deg, i.e.
        # t mod 10 s in [6.94, 9.72] s, probe 3 (phase + 120 deg) for phase
        # mod 180 in [35, 85] deg, i.e. t mod 10 s in [1.94, 4.72] s.
        t_0 = np.datetime64("2019-01-01T00:00:00", "ns")

        def timeline(f_s, duration=60.0):
            t_sec = np.arange(0.0, duration, 1.0 / f_s)
            return t_sec, t_0 + (t_sec * 1e9).astype("timedelta64[ns]")

        t_pot, time_pot = timeline(32.0)
        sc_pot = xr.DataArray(
            np.tile([5.0, 4.0, 5.5, 4.5, 6.0, 6.5], (len(t_pot), 1)),
            coords=[time_pot, np.arange(1, 7)],
            dims=["time", "probe"],
        )
        t_b, time_b = timeline(16.0, 80.0)
        time_b = time_b - np.timedelta64(10, "s")
        b_data = np.tile([30.0, 0.0, 2.0], (len(t_b), 1))
        b_data[(t_b - 10 >= 30) & (t_b - 10 < 40)] = [2.0, 0.0, 30.0]
        b_xyz = pyrf.ts_vec_xyz(time_b, b_data)
        t_ph, time_ph = timeline(4.0, 80.0)
        z_phase = pyrf.ts_scalar(
            time_ph - np.timedelta64(10, "s"), np.mod(18.0 * (t_ph - 10), 360.0)
        )
        inputs = [inp.data.copy() for inp in [sc_pot, b_xyz, z_phase]]

        start1, end1, start3, end3 = mms.probe_align_times(None, b_xyz, sc_pot, z_phase)

        def seconds(times):
            return (times - t_0) / np.timedelta64(1, "s")

        spins = np.array([0, 1, 2, 4, 5]) * 10.0
        dt_pot = 1.0 / 32.0
        for start, end, (t_l, t_r) in [
            (start1, end1, (125 / 18, 175 / 18)),
            (start3, end3, (35 / 18, 85 / 18)),
        ]:
            np.testing.assert_allclose(seconds(start), spins + t_l, atol=dt_pot)
            np.testing.assert_allclose(seconds(end), spins + t_r, atol=dt_pot)

        for inp, data_ in zip([sc_pot, b_xyz, z_phase], inputs):
            np.testing.assert_array_equal(inp.data, data_)


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

    def test_psd2dpf_relativistic(self):
        # Inverse of dpf2psd, also for the relativistic phase-space density
        vdf = generate_vdf(64.0, 10, (32, 32, 16), False, "electrons")
        vdf.data.attrs["UNITS"] = "1/(cm^2 s sr keV)"
        vdf = vdf.assign(energy=(vdf.energy + 1.0) * 1e3)

        for relativistic in [False, True]:
            psd = mms.dpf2psd(vdf, relativistic=relativistic)
            result = mms.psd2dpf(psd, relativistic=relativistic)
            np.testing.assert_allclose(result.data.data, vdf.data.data, rtol=1e-12)

        psd = mms.dpf2psd(vdf, relativistic=True)
        self.assertTrue(np.all(mms.psd2dpf(psd).data.data > vdf.data.data))


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
        mms.psd_moments(vdf, generate_ts(64.0, 100, tensor_order=0))

    @data({"energy_range": [1, 1000]}, {"no_sc_pot": True})
    def test_psd_moments_options(self, options):
        vdf = generate_vdf(64.0, 100, (32, 32, 16))
        vdf.data.attrs["FIELDNAM"] = "MMS1 FPI/DIS brstSkyMap dist"
        mms.psd_moments(vdf, generate_ts(64.0, 100, tensor_order=0), **options)

    @data(
        (
            np.random.random((100, 32)),  # energy
            np.random.random((100, 32)),  # delta_v
            random.random(),  # q_e
            np.random.random(100),  # sc_pot
            random.random(),  # p_mass
            random.choice([True, False]),  # flag_inner_electron
            random.random(),  # w_inner_electron
            np.random.random((100, 32, 16)),  # phi
            np.random.random((100, 32, 16)),  # theta
            np.ones((100, 32), dtype=bool),  # use_energy
            np.random.random((100, 32, 32, 16)),  # vdf
            np.random.random((100, 32, 16)),  # delta_ang
            np.zeros((1, 10, 1, 1)),  # exact_kernels
            False,  # use_exact
        )
    )
    def test_moms(self, value):
        result = _moms.__wrapped__(*value)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], np.ndarray)

    def test_moms_exact_kernels(self):
        # The midpoint kernels passed as exact kernels (delta_ang = 1) give the
        # midpoint moments
        rng = np.random.default_rng(0)
        n_t, n_e, n_ph, n_th = 3, 4, 8, 6
        phi = rng.uniform(0, 360, (n_t, n_ph, n_th))
        theta = rng.uniform(0, 180, (n_t, n_ph, n_th))
        delta_ang = rng.uniform(0.1, 0.2, (n_t, n_ph, n_th))
        ph, th = np.deg2rad(phi), np.deg2rad(theta)
        s_t, c_t, s_p, c_p = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        kernels = np.stack(
            [
                s_t,
                -c_p * s_t**2,
                -s_p * s_t**2,
                -s_t * c_t,
                c_p**2 * s_t**3,
                s_p**2 * s_t**3,
                s_t * c_t**2,
                c_p * s_p * s_t**3,
                c_p * s_t**2 * c_t,
                s_p * s_t**2 * c_t,
            ],
            axis=1,
        )
        args = [
            np.tile(10.0 * 2.0 ** np.arange(n_e), (n_t, 1)),
            rng.uniform(1.0, 2.0, (n_t, n_e)),
            constants.e,
            np.zeros(n_t),
            constants.m_p,
            False,
            0.0,
            phi,
            theta,
            np.ones((n_t, n_e), dtype=bool),
            rng.uniform(0.0, 1.0, (n_t, n_e, n_ph, n_th)),
        ]
        expected = _moms.__wrapped__(*args, delta_ang, np.zeros((1, 10, 1, 1)), False)
        exact = _moms.__wrapped__(
            *args,
            np.ones(delta_ang.shape),
            kernels * delta_ang[:, None],
            True,
        )

        for res, exp in zip(exact, expected):
            np.testing.assert_allclose(res, exp, rtol=1e-12)

    @staticmethod
    def _drifting_maxwellian(n, t_ev, v_kms, n_t=2, energy1=None):
        # Isotropic proton Maxwellian on an FPI-like burst skymap (32
        # log-spaced energies, 32 phi, 16 theta). One energy table unless
        # energy1 is given, in which case samples alternate energy0/energy1.
        m_p, q_e = constants.proton_mass, constants.elementary_charge
        ratio = (3e4 / 10.0) ** (1 / 31)
        energy0 = 10.0 * ratio ** np.arange(32)
        energy1 = energy0 if energy1 is None else energy1
        phi = 5.625 + 11.25 * np.arange(32)
        theta = 5.625 + 11.25 * np.arange(16)

        step_table = np.zeros(n_t, dtype=np.uint8)
        if energy1 is not energy0:
            step_table = np.arange(n_t, dtype=np.uint8) % 2

        energy = np.where(step_table[:, None] == 1, energy1, energy0)

        # Particle velocity is minus the FPI look direction
        speed = np.sqrt(2 * q_e * energy / m_p)[..., None, None]
        ph, th = np.deg2rad(phi)[:, None], np.deg2rad(theta)[None, :]
        v_x = -speed * np.sin(th) * np.cos(ph)
        v_y = -speed * np.sin(th) * np.sin(ph)
        v_z = -speed * np.cos(th) * np.ones_like(ph)

        v_d = np.array(v_kms) * 1e3
        v_th2 = 2 * q_e * t_ev / m_p
        dv2 = (v_x - v_d[0]) ** 2 + (v_y - v_d[1]) ** 2 + (v_z - v_d[2]) ** 2
        vdf = n * 1e-6 / (np.pi * v_th2) ** 1.5 * np.exp(-dv2 / v_th2)  # s^3/cm^6

        time = generate_timeline(1 / 0.15, n_t)
        vdf = pyrf.ts_skymap(
            time,
            vdf,
            energy,
            np.tile(phi, (n_t, 1)),
            theta,
            energy0=energy0,
            energy1=energy1,
            esteptable=step_table,
            attrs={"FIELDNAM": "MMS1 FPI/DIS brstSkyMap dist"},
            glob_attrs={
                "species": "ions",
                "delta_energy_plus": energy * (np.sqrt(ratio) - 1),
                "delta_energy_minus": energy * (1 - 1 / np.sqrt(ratio)),
            },
        )
        return vdf, pyrf.ts_scalar(time, np.zeros(n_t))

    def test_psd_moments_drifting_maxwellian(self):
        v_kms = [200.0, 150.0, -250.0]
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, v_kms)
        n, v, _, p2, t, _ = mms.psd_moments(vdf, sc_pot)

        self.assertAlmostEqual(float(n.data[0]), 1.0, delta=0.01)
        np.testing.assert_allclose(v.data[0], v_kms, atol=2.0)
        np.testing.assert_allclose(np.diag(t.data[0]), 1000.0, rtol=0.01)

        # Off-diagonal elements: T is isotropic (0) and the full second moment
        # p2 reduces to m n Vi Vj (nPa). Pxz/Pyz used the wrong angular kernel.
        i_row, i_col = [0, 0, 1], [1, 2, 2]
        np.testing.assert_allclose(t.data[0][i_row, i_col], 0.0, atol=5.0)
        v_d = np.array(v_kms) * 1e3
        p2_expected = constants.proton_mass * 1e6 * np.outer(v_d, v_d) * 1e9
        np.testing.assert_allclose(
            p2.data[0][i_row, i_col], p2_expected[i_row, i_col], rtol=0.02
        )

    @data(False, True)
    def test_psd_moments_no_warnings(self, relativistic):
        # Electrons with V_sc above the lowest channel (unused speeds warned
        # "invalid value in sqrt"), and a time without particles: n = 0 and
        # NaN bulk velocity, pressure, temperature and heat flux (0 / 0
        # warned)
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        vdf.attrs["species"] = "electrons"
        vdf.data.data[1] = 0.0
        sc_pot.data[:] = 15.0

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            n, v, p, p2, t, h = mms.psd_moments(vdf, sc_pot, relativistic=relativistic)

        self.assertGreater(n.data[0], 0.0)
        self.assertEqual(n.data[1], 0.0)
        np.testing.assert_array_equal(p2.data[1], 0.0)

        for moment in [v, p, t, h]:
            self.assertTrue(np.all(np.isfinite(moment.data[0])))
            self.assertTrue(np.all(np.isnan(moment.data[1])))

    @data(("s^3/cm^6", 1.0), ("s^3/m^6", 1e12), ("S^3/KM^6", 1e30), (None, 1.0))
    @unpack
    def test_psd_moments_units(self, units, scale):
        # The same Maxwellian in other units, or without UNITS (s^3/cm^6, as
        # FPI), gives the same moments. s^3/m^6 and s^3/km^6 were taken as
        # s^3/cm^6 (moments 1e12 and 1e30 too large)
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        expected = mms.psd_moments(vdf, sc_pot)

        vdf.data.data *= scale
        vdf.data.attrs.pop("UNITS", None)

        if units is not None:
            vdf.data.attrs["UNITS"] = units

        for result, reference in zip(mms.psd_moments(vdf, sc_pot), expected):
            atol = 1e-10 * np.max(np.abs(reference.data))
            np.testing.assert_allclose(
                result.data, reference.data, rtol=1e-10, atol=atol
            )

    @data("keV/(cm^2 s sr keV)", "1/(cm^2 s sr eV)")
    def test_psd_moments_units_flux(self, units):
        # Fluxes are converted to phase-space densities (vdf_convert), as
        # irfu-matlab; they were taken as s^3/cm^6
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        expected = mms.psd_moments(vdf, sc_pot)

        flux = mms.vdf_convert(vdf, units)
        for result, reference in zip(mms.psd_moments(flux, sc_pot), expected):
            atol = 1e-10 * np.max(np.abs(reference.data))
            np.testing.assert_allclose(
                result.data, reference.data, rtol=1e-10, atol=atol
            )

        flux.data.attrs["UNITS"] = "counts"

        with self.assertRaises(ValueError):
            mms.psd_moments(flux, sc_pot)

    def test_psd_moments_model_vdf(self):
        # Moments of a model distribution (make_model_vdf, in s^3/km^6)
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        time = vdf.time.data
        n_s = pyrf.ts_scalar(time, np.full(len(time), 2.0))
        v_xyz = pyrf.ts_vec_xyz(time, np.tile([100.0, -50.0, 200.0], (len(time), 1)))
        t_xyz = pyrf.ts_tensor_xyz(time, np.tile(np.eye(3) * 800.0, (len(time), 1, 1)))
        b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 10.0], (len(time), 1)))
        model = mms.make_model_vdf(vdf, b_xyz, sc_pot, n_s, v_xyz, t_xyz)

        n, v, _, _, t, _ = mms.psd_moments(model, sc_pot)
        np.testing.assert_allclose(n.data, 2.0, rtol=0.01)
        np.testing.assert_allclose(v.data, v_xyz.data, atol=2.0)
        np.testing.assert_allclose(
            np.trace(t.data, axis1=1, axis2=2) / 3, 800.0, rtol=0.01
        )

    def test_psd_moments_tables_differ_in_some_channels(self):
        # Tables that differ in all but one channel must use the alternating-table
        # speed widths (which don't use delta_energy_*). `all(e_tmp) == 0` flagged
        # them as a single table, so zero delta_energy widths gave n = 0.
        ratio = (3e4 / 10.0) ** (1 / 31)
        energy1 = 10.0 * ratio ** (np.arange(32) + 0.5)
        energy1[0] = 10.0
        vdf, sc_pot = self._drifting_maxwellian(
            1.0, 1000.0, [200.0, 150.0, -250.0], n_t=4, energy1=energy1
        )
        vdf.attrs["delta_energy_plus"] = np.zeros_like(vdf.attrs["delta_energy_plus"])
        vdf.attrs["delta_energy_minus"] = np.zeros_like(vdf.attrs["delta_energy_minus"])
        n, _, _, _, _, _ = mms.psd_moments(vdf, sc_pot)
        np.testing.assert_allclose(n.data, 1.0, rtol=0.01)

    def test_psd_moments_non_contiguous(self):
        # get_dist returns the data as a transposed float32 view (not
        # C-contiguous), which made the numba kernels fail on every call
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0])
        data = vdf.data.data.astype(np.float32)
        data_t = np.ascontiguousarray(data.transpose(0, 3, 2, 1)).transpose(0, 3, 2, 1)
        self.assertFalse(data_t.flags["C_CONTIGUOUS"])

        vdf_c = vdf.copy()
        vdf_c["data"] = vdf.data.copy(data=data)
        vdf_t = vdf.copy()
        vdf_t["data"] = vdf.data.copy(data=data_t)
        self.assertFalse(vdf_t.data.data.flags["C_CONTIGUOUS"])

        result = self._moments(vdf_t, sc_pot)
        self._assert_moments_equal(result, self._moments(vdf_c, sc_pot))
        self.assertAlmostEqual(float(result[0][0]), 1.0, delta=0.01)

        # The caller's data is left unchanged
        np.testing.assert_array_equal(vdf_t.data.data, data)

    def test_psd_moments_clipped_tables(self):
        # Fewer than 32 channels (vdf_elim output) raised IndexError in the
        # alternating-table and fast-mode speed widths. The clipped VDF gives
        # the same moments as the same channels of the full one: exactly with
        # one table (delta_energy widths), to 1 % with alternating tables
        # (the widths of the end channels are extrapolated linearly, as in
        # irfu-matlab).
        ratio = (3e4 / 10.0) ** (1 / 31)
        energy1 = 10.0 * ratio ** (np.arange(32) + 0.5)
        vdf_alt, sc_pot_alt = self._drifting_maxwellian(
            1.0, 1000.0, [200.0, 150.0, 0.0], n_t=4, energy1=energy1
        )
        vdf_fast, sc_pot_fast = self._drifting_maxwellian(
            1.0, 1000.0, [200.0, 150.0, 0.0]
        )
        vdf_fast.data.attrs["FIELDNAM"] = "MMS1 FPI/DIS fastSkyMap dist"

        for vdf, sc_pot, e_int, rtol in [
            (vdf_alt, sc_pot_alt, [120.0, 7000.0], 1e-2),
            (vdf_fast, sc_pot_fast, [120.0, 1500.0], 1e-10),
        ]:
            tables = np.vstack([vdf.attrs["energy0"], vdf.attrs["energy1"]])
            kept = np.where(np.any((tables > e_int[0]) & (tables < e_int[1]), 0))[0]
            channels = [kept[0], kept[-1] + 1]
            self.assertLess(len(kept), 32)

            vdf_clip = mms.vdf_elim(vdf, e_int)
            self.assertEqual(vdf_clip.energy.shape[1], len(kept))
            result = self._moments(vdf_clip, sc_pot)
            expected = self._moments(vdf, sc_pot, en_channels=channels)
            if rtol > 1e-10:
                # n, V and the diagonal of P (the other elements are ~0)
                result = [result[0], result[1], np.diagonal(result[2], 0, 1, 2)]
                expected = [expected[0], expected[1], np.diagonal(expected[2], 0, 1, 2)]
            for res, exp in zip(result, expected):
                np.testing.assert_allclose(res, exp, rtol=rtol)

    def test_psd_moments_en_channels_fpi_index(self):
        # The recommended energy index of the FPI partial-moments files
        # (partidxi_fpi_*_l2, uint16) as the first channel, to compare the ion
        # moments with FPI's
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0])
        result = self._moments(vdf, sc_pot, en_channels=[np.uint16(11), 32])
        self._assert_moments_equal(
            result, self._moments(vdf, sc_pot, en_channels=[11, 32])
        )
        self.assertTrue(np.all(result[0] < self._moments(vdf, sc_pot)[0]))

    def test_psd_moments_en_channels_time_series(self):
        # A time series of first channels (as partidxi_fpi_*_l2): at each time
        # the channels from it to the last, as energy_range with each time's
        # own range
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0], n_t=4)
        first = np.array([0, 3, 11, 3], dtype=np.uint16)
        energy = vdf.energy.data[0]
        e_min = [0.0 if k == 0 else (energy[k - 1] + energy[k]) / 2 for k in first]
        e_range = np.stack([e_min, np.full(4, 1e6)], axis=1)

        result = self._moments(
            vdf, sc_pot, en_channels=pyrf.ts_scalar(vdf.time.data, first)
        )
        self._assert_moments_equal(
            result, self._moments(vdf, sc_pot, energy_range=e_range)
        )

        # A constant series is en_channels=[k, 32]
        for k in [0, 11]:
            constant = pyrf.ts_scalar(vdf.time.data, np.full(4, k, dtype=np.uint16))
            self._assert_moments_equal(
                self._moments(vdf, sc_pot, en_channels=constant),
                self._moments(vdf, sc_pot, en_channels=[k, 32]),
            )

        # On other times: the nearest sample, not interpolated
        shifted = pyrf.ts_scalar(vdf.time.data + np.timedelta64(1, "ms"), first)
        self._assert_moments_equal(
            self._moments(vdf, sc_pot, en_channels=shifted), result
        )

    def test_psd_moments_en_channels_time_series_invalid(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        time = vdf.time.data

        for first in [
            pyrf.ts_scalar(time, np.array([0, 32])),
            pyrf.ts_scalar(time, np.array([-1, 0])),
            xr.DataArray(
                np.zeros((2, 2)), coords=[time, [0, 1]], dims=["time", "comp"]
            ),
        ]:
            with self.assertRaises(ValueError):
                mms.psd_moments(vdf, sc_pot, en_channels=first)

    @staticmethod
    def _electrons_through_potential(t_ev, sc_pot, energy01=False, rate="brst"):
        # Isotropic electron Maxwellian (1 cm^-3, t_ev) seen through a positive
        # spacecraft potential: a channel at energy E samples the ambient
        # distribution at E - sc_pot (Liouville), nothing below sc_pot
        m_e, q_e = constants.electron_mass, constants.elementary_charge
        ratio = (3e4 / 6.5) ** (1 / 31)
        energy0 = 6.5 * ratio ** np.arange(32)
        energy1 = 6.5 * ratio ** (np.arange(32) + 0.5) if energy01 else energy0
        step_table = np.arange(4, dtype=np.uint8) % 2 if energy01 else np.zeros(4)
        energy = np.where(step_table[:, None] == 1, energy1, energy0)

        v_th2 = 2 * q_e * t_ev / m_e
        e_amb = np.clip(energy - sc_pot, 0.0, None)
        vdf = 1e-6 / (np.pi * v_th2) ** 1.5 * np.exp(-2 * q_e * e_amb / m_e / v_th2)
        vdf[energy <= sc_pot] = 0.0
        vdf = np.tile(vdf[:, :, None, None], (1, 1, 32, 16))

        time = generate_timeline(1 / 0.03, 4)
        vdf = pyrf.ts_skymap(
            time,
            vdf,
            energy,
            np.tile(5.625 + 11.25 * np.arange(32), (4, 1)),
            5.625 + 11.25 * np.arange(16),
            energy0=energy0,
            energy1=energy1,
            esteptable=step_table.astype(np.uint8),
            attrs={"FIELDNAM": f"MMS1 FPI/DES {rate}SkyMap dist"},
            glob_attrs={
                "species": "electrons",
                "delta_energy_plus": energy * (np.sqrt(ratio) - 1),
                "delta_energy_minus": energy * (1 - 1 / np.sqrt(ratio)),
            },
        )
        return vdf, pyrf.ts_scalar(time, sc_pot * np.ones(4))

    @data(
        (20.0, 5.0, False, "brst"),
        (20.0, 10.0, False, "brst"),
        (50.0, 15.0, False, "brst"),
        (20.0, 10.0, True, "brst"),
        (20.0, 10.0, False, "fast"),
    )
    @unpack
    def test_psd_moments_sc_pot_widths(self, t_ev, sc_pot, energy01, rate):
        # The speed widths come from the channel edges corrected for the
        # spacecraft potential (irfu-matlab); the uncorrected edges gave
        # n = 0.87, 0.80, 0.86 for these cold electrons
        vdf, scpot = self._electrons_through_potential(t_ev, sc_pot, energy01, rate)
        n_psd = mms.psd_moments(vdf, scpot)[0]
        np.testing.assert_allclose(n_psd.data, 1.0, rtol=0.02)

    def test_psd_moments_angular_widths_attrs(self):
        # delta_phi_* and delta_theta_* attrs are in degrees (FPI files); they
        # were used as radians (n x 3283). Half-widths of 5.625 deg give the
        # same moments as the 11.25 deg grid spacing used without them.
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0])
        expected = self._moments(vdf, sc_pot)

        n_t, _, n_phi, n_theta = vdf.data.shape
        vdf.attrs["delta_phi_minus"] = np.full((n_t, n_phi), 5.625)
        vdf.attrs["delta_phi_plus"] = np.full((n_t, n_phi), 5.625)
        vdf.attrs["delta_theta_minus"] = np.full(n_theta, 5.625)
        vdf.attrs["delta_theta_plus"] = np.full(n_theta, 5.625)
        result = self._moments(vdf, sc_pot)
        self._assert_moments_equal(result, expected)
        self.assertAlmostEqual(float(result[0][0]), 1.0, delta=0.01)

    def test_psd_moments_pitch_angle_range(self):
        # Isotropic Maxwellian, oblique B: the density in the cone [0, a] is
        # n (1 - cos(a)) / 2, closely with the solid-angle fractions
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        b_xyz = pyrf.ts_vec_xyz(vdf.time.data, np.tile([3.0, -4.0, 5.0], (2, 1)))
        options = {"b_xyz": b_xyz, "angle_integration": "exact"}
        n_full = mms.psd_moments(vdf, sc_pot, angle_integration="exact")[0].data

        for weight, rtol in [("fraction", 2e-3), ("centre", 0.05)]:
            n_cone = mms.psd_moments(
                vdf,
                sc_pot,
                pitch_angle_range=[0.0, 30.0],
                pitch_angle_weight=weight,
                **options,
            )[0].data
            expected = n_full * (1 - np.cos(np.deg2rad(30.0))) / 2
            np.testing.assert_allclose(n_cone, expected, rtol=rtol)

            # Complementary ranges, and all the pitch angles
            n_lower, n_upper = (
                mms.psd_moments(
                    vdf,
                    sc_pot,
                    pitch_angle_range=r,
                    pitch_angle_weight=weight,
                    **options,
                )[0].data
                for r in [[0.0, 90.0], [90.0, 180.0]]
            )
            np.testing.assert_allclose(n_lower + n_upper, n_full, rtol=1e-12)

            for integration in ["midpoint", "exact"]:
                result = mms.psd_moments(
                    vdf,
                    sc_pot,
                    b_xyz=b_xyz,
                    pitch_angle_range=[0.0, 180.0],
                    pitch_angle_weight=weight,
                    angle_integration=integration,
                )
                expected = mms.psd_moments(vdf, sc_pot, angle_integration=integration)
                for res, exp in zip(result, expected):
                    np.testing.assert_array_equal(res.data, exp.data)

        # No B: NaN moments
        b_xyz.data[0] = np.nan
        n_i = mms.psd_moments(vdf, sc_pot, b_xyz=b_xyz, pitch_angle_range=[0, 90])
        self.assertTrue(np.isnan(n_i[0].data[0]))
        self.assertTrue(np.isfinite(n_i[0].data[1]))

    def test_psd_moments_pitch_angle_input(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        b_xyz = pyrf.ts_vec_xyz(vdf.time.data, np.tile([0.0, 0.0, 1.0], (2, 1)))

        for kwargs in [
            {"pitch_angle_range": [0.0, 90.0]},
            {"pitch_angle_range": [90.0], "b_xyz": b_xyz},
            {"pitch_angle_range": [90.0, 30.0], "b_xyz": b_xyz},
            {"pitch_angle_range": [0.0, 200.0], "b_xyz": b_xyz},
            {
                "pitch_angle_range": [0.0, 90.0],
                "b_xyz": b_xyz,
                "pitch_angle_weight": "x",
            },
        ]:
            with self.assertRaises(ValueError):
                mms.psd_moments(vdf, sc_pot, **kwargs)

    def test_psd_moments_float32(self):
        # FPI angles and data are float32: the exact kernels (float64) failed
        # in numba; the moments are those of float64 inputs
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        vdf_32 = vdf.assign(
            phi=vdf.phi.astype(np.float32), theta=vdf.theta.astype(np.float32)
        )
        vdf_32["data"] = vdf_32.data.astype(np.float32)

        for angle_integration in ["midpoint", "exact"]:
            options = {"angle_integration": angle_integration}
            result = mms.psd_moments(vdf_32, sc_pot, **options)
            expected = mms.psd_moments(vdf, sc_pot, **options)

            for res, exp in zip(result[:2], expected[:2]):
                np.testing.assert_allclose(res.data, exp.data, rtol=1e-5, atol=1e-3)

    def test_psd_moments_exact_angles(self):
        # FPI grid: the exact angular integrals change the moments by < 1 %
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        n_m, v_m, *_ = mms.psd_moments(vdf, sc_pot)
        n_e, v_e, *_ = mms.psd_moments(vdf, sc_pot, angle_integration="exact")
        np.testing.assert_allclose(n_e.data, n_m.data, rtol=0.01)
        np.testing.assert_allclose(v_e.data, v_m.data, rtol=0.01)
        self.assertAlmostEqual(float(n_e.data[0]), 1.0, delta=0.01)

        # 4 rings of 8 phi bins (FEEPS electrons): the midpoint rule
        # underestimates the density by 8 %, the exact integrals don't
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        n_t = vdf.data.shape[0]
        theta, edges = np.array([5.0, 65.0, 115.0, 175.0]), np.array(
            [0, 35, 90, 145, 180]
        )
        coarse = pyrf.ts_skymap(
            vdf.time.data,
            vdf.data.data[:, :, ::4, [0, 5, 10, 15]].copy(),
            vdf.energy.data,
            np.tile(22.5 + 45.0 * np.arange(8), (n_t, 1)),
            theta,
            energy0=vdf.attrs["energy0"],
            energy1=vdf.attrs["energy1"],
            esteptable=vdf.attrs["esteptable"],
            attrs=dict(vdf.data.attrs),
            glob_attrs={
                "species": "ions",
                "delta_phi_minus": np.full((n_t, 8), 22.5),
                "delta_phi_plus": np.full((n_t, 8), 22.5),
                "delta_theta_minus": theta - edges[:-1],
                "delta_theta_plus": edges[1:] - theta,
            },
        )
        n_m = mms.psd_moments(coarse, sc_pot)[0].data
        n_e = mms.psd_moments(coarse, sc_pot, angle_integration="exact")[0].data
        np.testing.assert_allclose(n_m / n_e, 11.6017 / (4 * np.pi), rtol=1e-4)
        self.assertAlmostEqual(float(n_e[0]), 1.0, delta=0.01)

        # The attribute sets the default, the argument has precedence
        coarse.attrs["angle_integration"] = "exact"
        np.testing.assert_array_equal(mms.psd_moments(coarse, sc_pot)[0].data, n_e)
        n_kw = mms.psd_moments(coarse, sc_pot, angle_integration="midpoint")[0].data
        np.testing.assert_array_equal(n_kw, n_m)

        with self.assertRaises(ValueError):
            mms.psd_moments(coarse, sc_pot, angle_integration="trapezoid")

    def test_psd_moments_any_data_rate(self):
        # The data rate in FIELDNAM is for information only: survey skymaps
        # (FEEPS) and skymaps without FIELDNAM used to raise
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [0.0, 0.0, 0.0])
        expected = self._moments(vdf, sc_pot)

        vdf.data.attrs["FIELDNAM"] = "mms1_feeps_srvy_ion_skymap"
        self._assert_moments_equal(self._moments(vdf, sc_pot), expected)

        del vdf.data.attrs["FIELDNAM"]
        self._assert_moments_equal(self._moments(vdf, sc_pot), expected)

    def _moments(self, vdf, sc_pot, **kwargs):
        # n, V, P and q at all times
        n, v, p, _, _, h = mms.psd_moments(vdf, sc_pot, **kwargs)
        return [n.data, v.data, p.data, h.data]

    def _assert_moments_equal(self, result, expected):
        for res, exp in zip(result, expected):
            np.testing.assert_allclose(res, exp, rtol=1e-12)

    def test_psd_moments_energy_range(self):
        # The channels strictly within the range, as with en_channels
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0])
        energy = vdf.energy.data[0]
        channels = np.where((energy > 100.0) & (energy < 3000.0))[0]

        result = self._moments(vdf, sc_pot, energy_range=[100.0, 3000.0])
        expected = self._moments(
            vdf, sc_pot, en_channels=[channels[0], channels[-1] + 1]
        )
        self._assert_moments_equal(result, expected)

        # It was ignored: full moments
        n_full = self._moments(vdf, sc_pot)[0]
        self.assertTrue(np.all(result[0] < 0.9 * n_full))

    def test_psd_moments_energy_range_time_dependent(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0], n_t=4)
        fixed_1 = self._moments(vdf, sc_pot, energy_range=[100.0, 3000.0])
        fixed_2 = self._moments(vdf, sc_pot, energy_range=[50.0, 1e4])

        # Same range at all times, as an array and as a DataArray on another
        # time line, gives the fixed range
        e_range = np.tile([100.0, 3000.0], (4, 1))
        result = self._moments(vdf, sc_pot, energy_range=e_range)
        self._assert_moments_equal(result, fixed_1)
        n_full = self._moments(vdf, sc_pot)[0]
        self.assertTrue(np.all(result[0] < 0.9 * n_full))

        time = vdf.time.data + np.timedelta64(1, "ms")
        e_range_ts = xr.DataArray(e_range, coords=[time, [0, 1]], dims=["time", "comp"])
        result = self._moments(vdf, sc_pot, energy_range=e_range_ts)
        self._assert_moments_equal(result, fixed_1)

        # Range changing halfway: each half has the moments of its range
        e_range[2:] = [50.0, 1e4]
        result = self._moments(vdf, sc_pot, energy_range=e_range)
        self._assert_moments_equal(
            [res[:2] for res in result], [exp[:2] for exp in fixed_1]
        )
        self._assert_moments_equal(
            [res[2:] for res in result], [exp[2:] for exp in fixed_2]
        )

    def test_psd_moments_energy_range_no_channel(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0], n_t=4)

        # Time steps without channel get NaN moments, with a warning
        e_range = np.tile([100.0, 3000.0], (4, 1))
        e_range[1] = [1e5, 2e5]
        with self.assertLogs("pyrfu", level="WARNING"):
            n = self._moments(vdf, sc_pot, energy_range=e_range)[0]

        self.assertTrue(np.isnan(n[1]))
        self.assertTrue(np.all(np.isfinite(n[[0, 2, 3]])))

        # Fixed range without channel, wrong shape, or both options
        for kwargs in [
            {"energy_range": [1e5, 2e5]},
            {"energy_range": [1.0, 2.0, 3.0]},
            {"energy_range": [100.0, 3000.0], "en_channels": [0, 32]},
        ]:
            with self.assertRaises(ValueError):
                mms.psd_moments(vdf, sc_pot, **kwargs)

    def test_psd_moments_burst_speed_widths(self):
        # One energy table in burst mode: (time, energy) widths, not tiled to
        # (time^2, energy)
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, 0.0], n_t=6)
        module = sys.modules["pyrfu.mms.psd_moments"]
        moms = module._moms

        with mock.patch.object(module, "_moms", side_effect=moms) as mocked:
            mms.psd_moments(vdf, sc_pot)

        self.assertEqual(mocked.call_args.args[1].shape, (6, 32))

    def test_psd_moments_partial_moments_mask(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        n_full = mms.psd_moments(vdf, sc_pot)[0].data

        # Not a 0/1 mask -> ignored, full moments (it used to be applied as weights)
        weights = 0.5 * np.ones(vdf.data.shape)
        n_weights = mms.psd_moments(vdf, sc_pot, partial_moments=weights)[0].data
        np.testing.assert_allclose(n_weights, n_full)

        # A 0/1 mask is applied: keeping all bins gives n, keeping none gives 0
        ones = np.ones(vdf.data.shape, dtype=int)
        n_ones = mms.psd_moments(vdf, sc_pot, partial_moments=ones)[0].data
        np.testing.assert_allclose(n_ones, n_full)
        mask = ones.copy()
        mask[:, :, :16, :] = 0
        n_half = mms.psd_moments(vdf, sc_pot, partial_moments=mask)[0].data
        self.assertTrue(np.all(n_half < 0.9 * n_full))

    def test_psd_moments_relativistic(self):
        # Electrons 30 keV to 500 keV moving downward (2 upper rings of 4, 8
        # phi bins, FEEPS-like): the relativistic moments are the velocity
        # integrals of f with the relativistic speeds, the momentum gamma m v
        # (pressure) and the kinetic energy (gamma - 1) m c^2 (energy flux)
        m_e, q_e, c_l = (
            constants.electron_mass,
            constants.elementary_charge,
            constants.c,
        )
        n_t, ratio = 2, (5e5 / 3e4) ** (1 / 11)
        energy = np.tile(3e4 * ratio ** np.arange(12), (n_t, 1))
        delta_minus, delta_plus = energy * (1 - ratio**-0.5), energy * (ratio**0.5 - 1)
        theta, edges = np.array([5.0, 65.0, 115.0, 175.0]), np.array(
            [0.0, 35.0, 90.0, 145.0, 180.0]
        )
        flux = 1e5 * (energy / 3e4)[:, :, None, None] ** -3 * np.ones((1, 1, 8, 4))
        flux[..., 2:] = 0.0  # 1/(cm^2 s sr keV)

        time = generate_timeline(0.05, n_t)
        flux = pyrf.ts_skymap(
            time,
            flux,
            energy,
            np.tile(22.5 + 45.0 * np.arange(8), (n_t, 1)),
            theta,
            energy0=energy[0],
            energy1=energy[0],
            esteptable=np.zeros(n_t, dtype=np.uint8),
            attrs={"UNITS": "1/(cm^2 s sr keV)"},
            glob_attrs={
                "species": "electrons",
                "angle_integration": "exact",
                "delta_energy_minus": delta_minus,
                "delta_energy_plus": delta_plus,
                "delta_phi_minus": np.full((n_t, 8), 22.5),
                "delta_phi_plus": np.full((n_t, 8), 22.5),
                "delta_theta_minus": theta - edges[:-1],
                "delta_theta_plus": edges[1:] - theta,
            },
        )
        vdf = mms.dpf2psd(flux, relativistic=True)
        vdf.data.data *= 1e-12
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        sc_pot = pyrf.ts_scalar(time, np.zeros(n_t))

        # Direct sums: speeds, Lorentz factors and widths in speed (SI)
        def speed(e_ev):
            return c_l * np.sqrt(1 - (1 + q_e * e_ev / (m_e * c_l**2)) ** -2)

        gamma = 1 + q_e * energy[0] / (m_e * c_l**2)
        v_e = speed(energy[0])
        f_v = (
            vdf.data.data[0, :, 0, 0]
            * 1e12
            * v_e**2
            * (speed(energy[0] + delta_plus[0]) - speed(energy[0] - delta_minus[0]))
        )

        # Exact angular integrals over the 2 upper rings (velocity -look)
        c_a, c_b = np.cos(np.deg2rad(edges[:2])), np.cos(np.deg2rad(edges[1:3]))
        omega = 2 * np.pi * np.sum(c_a - c_b)
        k_z = -np.pi * np.sum(c_a**2 - c_b**2)
        k_zz = 2 * np.pi * np.sum(c_a**3 - c_b**3) / 3

        n_exp = np.sum(f_v) * omega
        f_z = np.sum(f_v * v_e) * k_z
        p2_zz = m_e * np.sum(gamma * f_v * v_e**2) * k_zz
        q_z = m_e * c_l**2 * np.sum((gamma - 1) * f_v * v_e) * k_z

        n, v, p, p2, _, h = mms.psd_moments(vdf, sc_pot, relativistic=True)
        np.testing.assert_allclose(n.data * 1e6, n_exp, rtol=1e-10)
        np.testing.assert_allclose(v.data[:, 2] * 1e3, f_z / n_exp, rtol=1e-10)
        np.testing.assert_allclose(v.data[:, :2], 0.0, atol=1e-6)
        np.testing.assert_allclose(p2.data[:, 2, 2] * 1e-9, p2_zz, rtol=1e-10)

        # Energy flux from the heat flux and the bulk-flow terms of psd_moments
        v_z, p_zz = v.data[:, 2] * 1e3, p.data[:, 2, 2] * 1e-9
        p_tr = np.trace(p.data, axis1=1, axis2=2) * 1e-9
        q_tot = h.data[:, 2] * 1e-3 + v_z * p_zz
        q_tot += 0.5 * v_z * (p_tr + m_e * n.data * 1e6 * v_z**2)
        np.testing.assert_allclose(q_tot, q_z, rtol=1e-10)

        # The density is the integral of the intensity, J / v dE dOmega
        j_si = flux.data.data[0, :, 0, 0] * 1e4 / (1e3 * q_e)
        n_flux = np.sum(j_si * q_e * (delta_minus[0] + delta_plus[0]) / v_e) * omega
        np.testing.assert_allclose(n.data * 1e6, n_flux, rtol=0.02)

        # The attribute sets the default, the argument has precedence
        vdf.attrs["relativistic"] = True
        n_attr = mms.psd_moments(vdf, sc_pot)[0].data
        np.testing.assert_array_equal(n_attr, n.data)
        n_nr = mms.psd_moments(vdf, sc_pot, relativistic=False)[0].data
        del vdf.attrs["relativistic"]
        np.testing.assert_array_equal(mms.psd_moments(vdf, sc_pot)[0].data, n_nr)
        self.assertTrue(np.all(n_nr > 1.3 * n.data))

    def test_psd_moments_relativistic_low_energy(self):
        # 1 keV protons: the relativistic moments are the non-relativistic ones
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        result = mms.psd_moments(vdf, sc_pot, relativistic=True)
        expected = mms.psd_moments(vdf, sc_pot)

        # (the heat flux is a small difference of large terms)
        for res, exp, rtol in zip(result, expected, [1e-4] * 5 + [1e-2]):
            atol = 1e-5 * np.max(np.abs(exp.data))
            np.testing.assert_allclose(res.data, exp.data, rtol=rtol, atol=atol)

        # Below the spacecraft potential, as the non-relativistic moments
        sc_pot.data[:] = 50.0
        result = mms.psd_moments(vdf, sc_pot, relativistic=True)
        expected = mms.psd_moments(vdf, sc_pot)
        np.testing.assert_allclose(result[0].data, expected[0].data, rtol=1e-4)


@ddt
class PsdRebinTestCase(unittest.TestCase):
    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_vdf_type(self, vdf):
        with self.assertRaises(TypeError):
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )

    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_phi_type(self, vdf):
        with self.assertRaises(TypeError):
            mms.psd_rebin(
                vdf.data,
                vdf.phi,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.energy[0, :],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.energy[1, :],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                pyrf.ts_scalar(vdf.time.data, vdf.attrs["esteptable"]),
            )

    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_output(self, vdf):
        result = mms.psd_rebin(
            vdf,
            vdf.phi.data,
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

    def test_psd_rebin_values(self):
        # 30 ms DES-like burst with alternating energy tables and f(t) = t + 1,
        # phi increasing within each pair (no wrap).
        n_t = 20
        time = np.datetime64("2020-01-01", "ns") + np.arange(n_t) * np.timedelta64(
            30, "ms"
        )
        vdf = generate_vdf(1.0, n_t, (32, 32, 16), energy01=True)
        vdf = vdf.assign_coords(time=time)
        vdf.data.data[...] = (np.arange(n_t) + 1.0)[:, None, None, None]
        phi = vdf.phi.data.astype(np.float64)
        phi[1::2] += 5.625

        time_r, vdf_r, _, _ = mms.psd_rebin(
            vdf,
            phi,
            vdf.attrs["energy0"],
            vdf.attrs["energy1"],
            vdf.attrs["esteptable"],
        )

        # Time stamps at the middle of each pair (the time step used to overflow int16)
        np.testing.assert_array_equal(time_r, time[::2] + np.timedelta64(15, "ms"))

        # Every pair is filled, including the last one (used to be all zeros).
        # esteptable[2k] = 0 -> energy0 (sample 2k) on even rows, energy1 on odd.
        expected_even = np.arange(1.0, n_t, 2)[:, None, None, None]
        expected_odd = np.arange(2.0, n_t + 1, 2)[:, None, None, None]
        np.testing.assert_array_equal(
            vdf_r[:, 0:63:2, ...],
            np.broadcast_to(expected_even, vdf_r[:, 0:63:2].shape),
        )
        np.testing.assert_array_equal(
            vdf_r[:, 1:64:2, ...], np.broadcast_to(expected_odd, vdf_r[:, 1:64:2].shape)
        )

    def test_psd_rebin_phi_relabelling(self):
        # Azimuths as in FPI burst data: the spin advances them by 2.8 deg per
        # sample, and they are labelled from one bin lower when the first one
        # would pass 11.25 deg (2015-10-30 DIS: 9.13, 20.38, ... then 0.63,
        # 11.88, ...). The rebinned samples must pair the same directions.
        n_t, d_phi, advance = 40, 11.25, 2.8
        vdf = generate_vdf(64.0, n_t, [32, 32, 16], energy01=True, species="ions")
        phi0 = (3.5 + advance * np.arange(n_t)) % d_phi
        phi = phi0[:, None] + d_phi * np.arange(32)[None, :]
        vdf.data.data[:] = (2.0 + np.cos(np.deg2rad(phi)))[:, None, :, None]

        wrap = phi[0:-1:2, 0] > phi[1::2, 0]
        self.assertTrue(np.any(wrap) and not np.all(wrap))

        _, vdf_r, _, phi_r = mms.psd_rebin(
            vdf,
            phi,
            vdf.attrs["energy0"],
            vdf.attrs["energy1"],
            vdf.attrs["esteptable"],
        )

        # Midpoint of the matching directions of the two samples
        np.testing.assert_allclose(phi_r, phi[0:-1:2] + advance / 2, atol=1e-4)

        # Both energy tables at a column see the same direction (to the spin
        # advance): it was 31 deg apart (shift by 2 bins) for relabelled pairs
        np.testing.assert_array_less(
            np.abs(vdf_r[:, 0::2] - vdf_r[:, 1::2]), np.deg2rad(advance) * 1.01
        )

    @idata(itertools.product([False, True], [0, 1]))
    @unpack
    def test_psd_rebin_energy_table_order(self, phi_wrap, first_step):
        # The lower table (energy0) must always go to the even channels, whether
        # or not phi wraps between the two samples of a pair (the phi-wrap
        # branch ignored the step table). Each sample holds a constant value,
        # so the order does not depend on the phi shift.
        n_t = 8
        vdf = generate_vdf(64.0, n_t, (32, 32, 16), energy01=True)
        step_table = (np.arange(n_t) + first_step) % 2
        vdf.data.data[...] = (np.arange(n_t) + 1.0)[:, None, None, None]
        phi = vdf.phi.data.astype(np.float64) + 5.625
        phi[1::2] += -5.625 if phi_wrap else 5.625

        _, vdf_r, energy_r, _ = mms.psd_rebin(
            vdf, phi, vdf.attrs["energy0"], vdf.attrs["energy1"], step_table
        )

        # Samples using energy0 (step 0) in the even channels, energy1 (step 1)
        # in the odd channels
        samples = np.arange(n_t).reshape(-1, 2)
        pair_steps = step_table.reshape(-1, 2)
        from_energy0 = samples[pair_steps == 0] + 1.0
        from_energy1 = samples[pair_steps == 1] + 1.0
        np.testing.assert_array_equal(vdf_r[:, 0:63:2, 0, 0].max(axis=1), from_energy0)
        np.testing.assert_array_equal(vdf_r[:, 0:63:2, 0, 0].min(axis=1), from_energy0)
        np.testing.assert_array_equal(vdf_r[:, 1:64:2, 0, 0].max(axis=1), from_energy1)
        np.testing.assert_array_equal(vdf_r[:, 1:64:2, 0, 0].min(axis=1), from_energy1)
        np.testing.assert_array_equal(energy_r[0:63:2], vdf.attrs["energy0"])


class EstimatePhaseSpeedTestCase(unittest.TestCase):
    def test_estimate_phase_speed(self):
        # Power ridge along f = v k / (2 pi), v = 20 km/s: the speed is recovered
        # and the caller's power spectrum is unchanged (parts were set to 0)
        k = np.linspace(-0.3, 0.3, 121)
        freq = np.linspace(0.0, 1000.0, 201)
        k_mat, f_mat = np.meshgrid(k, freq)
        power = np.exp(-(((f_mat - 2e4 * k_mat / (2 * np.pi)) / 20.0) ** 2))
        power *= k_mat > 0
        power[50, 60] = np.nan
        power_in = power.copy()

        vph = mms.estimate_phase_speed(power, freq, k, 100.0)

        self.assertAlmostEqual(vph / 2e4, 1.0, delta=0.01)
        np.testing.assert_array_equal(power, power_in)

    def test_estimate_phase_speed_weights(self):
        # The fit is irfu-matlab's least squares weighted by
        # w = 1 + log10(P / P_max): the slope 2 pi sum(w k f) / sum(w k^2)
        # over the selected points. scipy's sigma=w weighted them by 1 / w^2,
        # the weakest the most (two modes: 13 % from the speed, against 10 %)
        k = np.linspace(-0.1, 0.1, 201)
        freq = np.linspace(0.0, 4000.0, 401)
        k_mat, f_mat = np.meshgrid(k, freq)
        v_ph = 1e5
        power = np.exp(-(((f_mat - v_ph * k_mat / (2 * np.pi)) / 80.0) ** 2))
        power += 0.5 * np.exp(
            -(((f_mat - 1.4 * v_ph * k_mat / (2 * np.pi)) / 80.0) ** 2)
        )
        power *= np.exp(-(((k_mat - 0.03) / 0.03) ** 2))

        vph = mms.estimate_phase_speed(power, freq, k, 100.0)

        # The selection of the function: masked k and f, the maximum, the
        # positive k side, power > 0.3 max and v within a factor 3 of the
        # maximum's
        sel_power = power.copy()
        sel_power[:, np.abs(k) > 2 * k.max() / 3] = 0.0
        sel_power[freq < 100.0, :] = 0.0
        i_f, i_k = np.unravel_index(np.argmax(sel_power), sel_power.shape)
        v_guess = freq[i_f] / k[i_k]
        sel_power[:, k < 0.0] = 0.0
        selected = sel_power > 0.3 * sel_power.max()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(f_mat / k_mat)

        selected &= (ratio > v_guess / 3) & (ratio < 3 * v_guess)
        weights = 1 + np.log10(sel_power[selected] / sel_power[selected].max())
        k_s, f_s = k_mat[selected], f_mat[selected]
        expected = 2 * np.pi * np.sum(weights * k_s * f_s) / np.sum(weights * k_s**2)

        self.assertAlmostEqual(vph / expected, 1.0, delta=1e-6)

    def test_estimate_phase_speed_negative(self):
        # Ridge along f = v k / (2 pi) with v = -20 km/s (k < 0), and a weaker
        # one of opposite sign (k > 0): the k > 0 side is left out
        k = np.linspace(-0.3, 0.3, 121)
        freq = np.linspace(0.0, 1000.0, 201)
        k_mat, f_mat = np.meshgrid(k, freq)

        def ridge(v_ph):
            return np.exp(-(((f_mat - v_ph * k_mat / (2 * np.pi)) / 20.0) ** 2))

        power = ridge(-2e4) * (k_mat < 0) + 0.5 * ridge(1e4) * (k_mat > 0)

        vph = mms.estimate_phase_speed(power, freq, k, 100.0)

        self.assertAlmostEqual(vph / -2e4, 1.0, delta=0.01)


@ddt
class FeepsActiveEyesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"], ["sitl", "l2"]))
    @unpack
    def test_feeps_active_eyes_output(self, data_rate, dtype, lev):
        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            TEST_TINT,
            random.randint(1, 4),
        )
        self.assertIsInstance(result, dict)

        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            pyrf.iso86012datetime64(np.array(TEST_TINT)),
            str(random.randint(1, 4)),
        )
        self.assertIsInstance(result, dict)

    def test_feeps_active_eyes_time_types(self):
        # str, datetime64 and datetime.datetime give the same eyes on each side
        # of the 2017-08-16 change (datetime.datetime used to raise)
        var = {"tmmode": "srvy", "dtype": "electron", "lev": "l2"}
        new = {"top": [3, 5, 9, 10, 12], "bottom": [2, 4, 5, 9, 10]}
        old = {"top": [3, 4, 5, 11, 12], "bottom": [3, 4, 5, 11, 12]}

        for start, expected in [
            ("2017-08-16T00:00:00", new),
            ("2017-08-15T23:59:00", old),
        ]:
            for t_start in [
                start,
                np.datetime64(start, "ns"),
                datetime.datetime.fromisoformat(start),
            ]:
                result = mms.feeps_active_eyes(var, [t_start, t_start], 1)
                self.assertDictEqual(result, expected)

    def test_feeps_active_eyes_sitl(self):
        # SITL: top eyes 5, 11 and 12 only, as IDL SPEDAS and pySPEDAS (also for
        # the ions); after 2017-08-16 in survey, those of them that are active,
        # sorted
        for dtype in ["electron", "ion"]:
            var = {"tmmode": "srvy", "dtype": dtype, "lev": "sitl"}
            result = mms.feeps_active_eyes(var, ["2017-08-01", "2017-08-02"], 1)
            self.assertDictEqual(result, {"top": [5, 11, 12], "bottom": []})

        tint = ["2019-01-01", "2019-01-02"]
        for mms_id, dtype, top in [
            (1, "electron", [5, 12]),
            (2, "electron", [5, 11]),
            (4, "electron", [5, 11]),
            (1, "ion", []),
        ]:
            var = {"tmmode": "srvy", "dtype": dtype, "lev": "sitl"}
            result = mms.feeps_active_eyes(var, tint, mms_id)
            self.assertListEqual(result["top"], top)
            self.assertListEqual(result["bottom"], [])


def _feeps_omni_4sc(values, energies=None):
    # Omni spectra of the 4 spacecraft (3 channels, 4 times), one constant per
    # spacecraft (NaN for no data), with their mmsId
    time = generate_timeline(1.0, 4)
    out = []
    for i, value in enumerate(values):
        energy = np.array([50.0, 70.0, 100.0]) if energies is None else energies[i]
        out.append(
            xr.DataArray(
                np.full((4, 3), value),
                coords=[time, energy],
                dims=["time", "energy"],
                attrs={"mmsId": i + 1, "species": "e"},
            )
        )
    return out


class FeepsAvg4scTestCase(unittest.TestCase):
    def test_feeps_avg_4sc_nan(self):
        # Spacecraft at 1, 2 and 4 and one without data: 7 / 3; a time without
        # data on any spacecraft gives NaN without warning
        omni = _feeps_omni_4sc([1.0, 2.0, 4.0, np.nan])
        for b in omni:
            b.data[2] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = mms.feeps_avg_4sc(tuple(omni))

        expected = np.full((4, 3), 7.0 / 3.0)
        expected[2] = np.nan
        np.testing.assert_allclose(result.data, expected)
        self.assertEqual(result.attrs["mmsId"], "4sc_avg")
        self.assertEqual(omni[0].attrs["mmsId"], 1)

    def test_feeps_avg_4sc_combined_energies(self):
        # Output channel 0 = channel 1 of MMS1 and channel 2 of MMS2 (MMS3 and
        # MMS4 left out), channels 1 and 2 = the same channel on all
        energies = [np.array([50.0, 70.0, 100.0]), np.array([40.0, 54.0, 80.0])] * 2
        omni = _feeps_omni_4sc([1.0, 2.0, 4.0, 8.0], energies)
        combined = {0: [1, 2, None, None], 1: [2, 3, 2, 3], 2: [3, 3, 3, 3]}

        result = mms.feeps_avg_4sc(omni, combined_energies=combined)

        np.testing.assert_allclose(result.data[0], [1.5, 15.0 / 4.0, 15.0 / 4.0])
        np.testing.assert_allclose(result.energy.data, [52.0, 75.0, 90.0])

    def test_feeps_avg_4sc_pad(self):
        # PADs (time, theta) of the 4 spacecraft, one without data
        time = generate_timeline(1.0, 4)
        theta = 8.18 + 16.36 * np.arange(11)
        pads = [
            xr.DataArray(
                np.full((4, 11), value),
                coords=[time, theta],
                dims=["time", "theta"],
                attrs={"mmsId": i + 1, "probe": i + 1, "mms": i + 1, "MMS": i + 1},
            )
            for i, value in enumerate([1.0, 2.0, 4.0, np.nan])
        ]

        result = mms.feeps_avg_4sc(pads, flag="pad")

        np.testing.assert_allclose(result.data, np.full((4, 11), 7.0 / 3.0))
        np.testing.assert_array_equal(result.theta.data, theta)
        for key in ["mmsId", "probe", "mms", "MMS"]:
            self.assertEqual(result.attrs[key], "4sc_avg")

    def test_feeps_avg_4sc_input(self):
        with self.assertRaises(TypeError):
            mms.feeps_avg_4sc(_feeps_omni_4sc([1.0, 2.0])[0])

        with self.assertRaises(TypeError):
            mms.feeps_avg_4sc(_feeps_omni_4sc([1.0, 2.0]), combined_energies=[[1, 1]])

        # Datasets were documented but failed with AttributeError
        omni = [b.to_dataset(name="omni") for b in _feeps_omni_4sc([1.0, 2.0])]
        with self.assertRaisesRegex(TypeError, "DataArray"):
            mms.feeps_avg_4sc(omni)

        with self.assertRaises(ValueError):
            mms.feeps_avg_4sc(_feeps_omni_4sc([1.0, 2.0]), flag="gpd")


@ddt
class FeepsCorrectEnergiesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_correct_energies_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_correct_energies(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)

    def test_feeps_correct_energies_values(self):
        # Energies from the table, the loader's dimension names kept (they were
        # renamed energy-top-1), and the caller's attrs left alone
        feeps_alle = generate_feeps(64.0, 10, "brst", "electron", "l2", "flux", 2)
        attrs = dict(feeps_alle.attrs)

        result = mms.feeps_correct_energies(feeps_alle)
        result.attrs["new"] = 1

        self.assertTupleEqual(result["top-1"].dims, ("time", "energy_top-1"))
        np.testing.assert_array_equal(
            result["energy_top-1"].data, mms.feeps_energy_table(2, "top", 1)
        )
        self.assertDictEqual(feeps_alle.attrs, attrs)

        # Already split (15 channels): clear error
        split, _ = mms.feeps_split_integral_ch(feeps_alle)
        with self.assertRaisesRegex(ValueError, "feeps_split_integral_ch"):
            mms.feeps_correct_energies(split)


class FeepsCorrectionsTestCase(unittest.TestCase):
    def test_feeps_corrections(self):
        # MMS2 ions in 2017: zero-gain and bad eye bottom-7, bad channels
        feeps_alle = generate_feeps(64.0, 100, "brst", "ion", "l2", "flux", 2)
        time = feeps_alle.time.data
        time = time - time[0] + np.datetime64("2017-07-11T22:00:00", "ns")
        feeps_alle = feeps_alle.assign_coords(time=time)
        feeps_alle["spinsectnum"].data = np.arange(100) % 64
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_corrections(feeps_alle)

        # The chain of IDL SPEDAS mms_load_feeps
        expected = mms.feeps_correct_energies(feeps_ref)
        expected = mms.feeps_flat_field_corrections(expected)
        expected = mms.feeps_remove_bad_data(expected)
        expected, _ = mms.feeps_split_integral_ch(expected)
        expected = mms.feeps_remove_sun(expected)
        xr.testing.assert_identical(result, expected)

        self.assertEqual(result["top-6"].shape, (100, 15))
        self.assertIn("energy_top-6", result["top-6"].dims)
        self.assertTrue(np.isnan(result["bottom-7"].data).all())
        self.assertTrue(np.isfinite(result["top-6"].data[:, 1:]).any())
        xr.testing.assert_identical(feeps_alle, feeps_ref)


@ddt
class FeepsFlatFieldCorrectionsTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_flat_field_corrections_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_flat_field_corrections(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)

    @idata(range(1, 5))
    def test_feeps_flat_field_corrections_values(self, mms_id):
        feeps_alle = generate_feeps(64.0, 100, "brst", "ion", "l2", "flux", mms_id)
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_flat_field_corrections(feeps_alle)

        # Each eye is scaled by its gain (1 if not in the table), and the eyes
        # with a gain of 0 are NaN (they used to be zeros)
        for k in filter(lambda x: x[:3] in ["top", "bot"], feeps_ref):
            sensor, eye = k.split("-")
            gain = g_corr.get(f"mms{mms_id}-{sensor[:3]}{int(eye)}", 1.0)
            gain = np.nan if gain == 0 else gain
            np.testing.assert_array_equal(result[k].data, feeps_ref[k].data * gain)
            self.assertDictEqual(result[k].attrs, feeps_ref[k].attrs)

        self.assertDictEqual(result.attrs, feeps_ref.attrs)

        # The caller's data must not be changed (they used to be scaled in place)
        xr.testing.assert_identical(feeps_alle, feeps_ref)


@ddt
class FeepsOmniTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_omni_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_omni(feeps_alle)
        self.assertIsInstance(result, xr.DataArray)

    @data("l2", "sitl")
    def test_feeps_omni_energy_check(self, lev):
        # mms1 electron omni energies
        energies = np.array(
            [33.2, 51.9, 70.6, 89.4, 107.1, 125.2, 146.5, 171.3]
            + [200.2, 234.0, 273.4, 319.4, 373.2, 436.0, 509.2]
        )
        energies += 14.0
        time = generate_timeline(64.0, 10)

        # 4 eyes at 1, top-2 at 3 with channel 3 20 % off and channel 4 5 % off,
        # bottom-1 with NaN energies
        eye_energies = {k: energies.copy() for k in ["top-1", "top-2", "bottom-1"]}
        eye_energies["top-2"][3] *= 1.2
        eye_energies["top-2"][4] *= 1.05
        eye_energies["bottom-1"][:] = np.nan
        eye_energies["bottom-2"] = energies.copy()

        feeps_dict = {}
        for k, e_eye in eye_energies.items():
            value = 3.0 if k == "top-2" else 1.0
            feeps_dict[k] = xr.DataArray(
                np.full((len(time), len(energies)), value),
                coords=[time, e_eye],
                dims=["time", f"energy-{k}"],
                attrs={"UNITS": "1/(cm^2 s sr keV)"},
            )

        feeps_alle = xr.Dataset(feeps_dict)
        feeps_alle.attrs = {
            "tmmode": "brst",
            "dtype": "electron",
            "lev": lev,
            "units_name": "flux",
            "species": "e",
            "mmsId": 1,
        }

        result = mms.feeps_omni(feeps_alle)
        np.testing.assert_allclose(result.energy.data, energies)

        # Channel 3 of top-2 is averaged only without the check (SITL), the
        # other channels of top-2 and the NaN-energy eye are always averaged
        expected = np.full(len(energies), 1.5)
        if lev != "sitl":
            expected[3] = 1.0

        np.testing.assert_allclose(result.data, np.tile(expected, (len(time), 1)))

    def test_feeps_omni_unsplit(self):
        # Unsplit eyes (16 channels) used to fail with a broadcasting error
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)

        with self.assertRaisesRegex(ValueError, "feeps_split_integral_ch"):
            mms.feeps_omni(feeps_alle)


@ddt
class FeepsPadTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_ouput(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))
        self.assertIsInstance(result, xr.DataArray)

    def test_feeps_pad_input_unchanged(self):
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        energy = np.linspace(40.0, 600.0, feeps_alle.sizes["energy"])
        feeps_alle = feeps_alle.assign_coords(energy=energy)

        # Integer counts with a whole eye at zero
        for k in feeps_alle:
            if k.startswith(("top", "bottom")):
                counts = np.round(100 * feeps_alle[k].data).astype(np.int64)
                feeps_alle[k] = feeps_alle[k].copy(data=counts)

        feeps_alle["top-2"].data[:] = 0
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.isfinite(result.data).all())
        self.assertEqual(result.attrs["energy_range"], [70.0, 600.0])

        # The zeros used to be set to NaN in the caller's data, and
        # energy_range added to the caller's attrs
        xr.testing.assert_identical(feeps_alle, feeps_ref)

    def test_feeps_pad_cdf_angles(self):
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        energy = np.linspace(40.0, 600.0, feeps_alle.sizes["energy"])
        feeps_alle = feeps_alle.assign_coords(energy=energy)

        # CDF pitch angles at 10 degrees, one eye with the fill value
        pitch_angle = np.full((feeps_alle.sizes["time"], 18), 10.0)
        pitch_angle[:, 0] = -2147483648
        feeps_alle["pitch_angle"] = (["time", "eye"], pitch_angle)

        b_bcs = generate_ts(64.0, 100, tensor_order=1)
        result = mms.feeps_pad(feeps_alle, b_bcs)

        # Burst: the CDF angles are used (they used to be computed from B),
        # 10 +/- 21.4 degrees covers the first two bins only
        self.assertTrue(np.isfinite(result.data[:, :2]).all())
        self.assertTrue(np.isnan(result.data[:, 2:]).all())

        # Angles from B on request, and always in survey
        result_b = mms.feeps_pad(feeps_alle, b_bcs, angles_from_bfield=True)
        feeps_alle.attrs["tmmode"] = "srvy"
        result_srvy = mms.feeps_pad(feeps_alle, b_bcs)
        expected = mms.feeps_pad(feeps_alle.drop_vars("pitch_angle"), b_bcs)
        xr.testing.assert_allclose(result_srvy, expected)
        self.assertFalse(np.isnan(result_b.data[:, 2:]).all())

    def test_feeps_pad_options(self):
        feeps_alle = generate_feeps(64.0, 20, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        energy = np.linspace(40.0, 600.0, feeps_alle.sizes["energy"])
        feeps_alle = feeps_alle.assign_coords(energy=energy)
        feeps_alle["pitch_angle"] = (
            ["time", "eye"],
            np.full((feeps_alle.sizes["time"], 18), 10.0),
        )
        for eye in filter(lambda x: x.startswith(("top", "bottom")), feeps_alle):
            feeps_alle[eye].attrs["UNITS"] = "counts/s"
        b_bcs = generate_ts(64.0, 20, tensor_order=1)

        # bin_size not dividing 180: bins of 25 degrees from 0 up to 175, as
        # IDL SPEDAS and pyspedas (they were 25.7 degrees wide, labelled as 25)
        result = mms.feeps_pad(feeps_alle, b_bcs, bin_size=25.0, energy=[32.0, 600.0])
        np.testing.assert_allclose(result.theta.data, 12.5 + 25.0 * np.arange(7))
        self.assertTrue(np.isfinite(result.data[:, :2]).all())
        self.assertTrue(np.isnan(result.data[:, 2:]).all())

        # 32 keV accepted (it raised), units of the input (count rate)
        self.assertEqual(result.attrs["UNITS"], "counts/s")

        with self.assertRaises(ValueError):
            mms.feeps_pad(feeps_alle, b_bcs, energy=[20.0, 600.0])

        with self.assertWarns(UserWarning):
            result = mms.feeps_pad(feeps_alle, b_bcs, energy=[700.0, 800.0])
        self.assertTrue(np.isnan(result.data).all())

        feeps_alle.attrs["tmmode"] = "fast"
        with self.assertRaises(ValueError):
            mms.feeps_pad(feeps_alle, b_bcs)

    def test_feeps_pad_unsplit(self):
        # Unsplit eyes (16 channels) used to average the integral channel
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)

        with self.assertRaisesRegex(ValueError, "feeps_split_integral_ch"):
            mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))

    def test_feeps_pad_values(self):
        feeps_alle = generate_feeps(64.0, 20, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        energy = np.linspace(40.0, 600.0, feeps_alle.sizes["energy"])
        feeps_alle = feeps_alle.assign_coords(energy=energy)

        # top-1 at 10 and top-2 at 30 degrees, the other eyes are filled
        pitch_angle = np.full((feeps_alle.sizes["time"], 18), -2147483648.0)
        pitch_angle[:, :2] = [10.0, 30.0]
        feeps_alle["pitch_angle"] = (["time", "eye"], pitch_angle)
        feeps_alle["top-1"].data[:] = 2.0
        feeps_alle["top-2"].data[:] = 4.0
        feeps_alle["top-2"].data[0, :] = np.nan

        result = mms.feeps_pad(feeps_alle, generate_ts(64.0, 20, tensor_order=1))

        # +/- 21.4 degrees: top-1 covers the bins 0-1, top-2 the bins 0-3
        expected = np.full((20, 11), np.nan)
        expected[:, :2], expected[:, 2:4] = 3.0, 4.0
        expected[0, :2], expected[0, 2:4] = 2.0, np.nan  # top-2 NaN
        np.testing.assert_allclose(result.data, expected)

    def test_feeps_pad_nan_energies(self):
        feeps_alle = generate_feeps(64.0, 20, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        energy = np.linspace(40.0, 600.0, feeps_alle.sizes["energy"])
        feeps_alle = feeps_alle.assign_coords(energy=energy)

        pitch_angle = np.full((feeps_alle.sizes["time"], 18), -2147483648.0)
        pitch_angle[:, :2] = [10.0, 30.0]
        feeps_alle["pitch_angle"] = (["time", "eye"], pitch_angle)
        feeps_alle["top-1"].data[:] = 2.0
        feeps_alle["top-2"].data[:] = 4.0

        # top-1 without energy calibration (NaN energies) is left out
        top_1 = feeps_alle["top-1"].rename(energy="energy_top-1")
        feeps_alle["top-1"] = top_1.assign_coords(
            {"energy_top-1": np.full(len(energy), np.nan)}
        )

        b_bcs = generate_ts(64.0, 20, tensor_order=1)
        result = mms.feeps_pad(feeps_alle, b_bcs)

        expected = np.full((20, 11), np.nan)
        expected[:, :4] = 4.0
        np.testing.assert_array_equal(result.data, expected)

        feeps_alle.attrs["dtype"] = "proton"
        with self.assertRaisesRegex(ValueError, "Invalid species"):
            mms.feeps_pad(feeps_alle, b_bcs)


def _spin_avg_case(dim, values):
    # 3 samples of a partial spin, 5 spins of 8 sectors, 4 samples of a partial
    # spin, with the data equal to the spin number (0 to 6)
    spin_sectors = np.hstack([[5, 6, 7], np.tile(np.arange(8), 5), [0, 1, 2, 3]])
    spin = np.hstack([[0] * 3, np.repeat(np.arange(1, 6), 8), [6] * 4]).astype(float)
    time = generate_timeline(0.4, len(spin))
    data = np.tile(spin[:, None], (1, len(values)))
    out = xr.DataArray(
        data, coords=[time, values], dims=["time", dim], attrs={"mmsId": 1}
    )
    return spin_sectors, out


@ddt
class FeepsPadSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        feeps_pad = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))
        result = mms.feeps_pad_spinavg(feeps_pad, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)

    def test_feeps_pad_spin_avg_input_types(self):
        feeps_alle = generate_feeps(64.0, 100, "srvy", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_pad = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))

        # A NumPy array used to raise TypeError; the 8 complete spins (the
        # partial last one was kept)
        result = mms.feeps_pad_spinavg(feeps_pad, feeps_alle.spinsectnum)
        result_np = mms.feeps_pad_spinavg(feeps_pad, feeps_alle.spinsectnum.data)

        self.assertEqual(len(result.time), 8)
        xr.testing.assert_identical(result, result_np)

    def test_feeps_pad_spin_avg_windows(self):
        # Partial spin (3 samples), 5 spins of 8 sectors, partial spin (4)
        theta = 8.1818 + 16.3636 * np.arange(11)
        spin_sectors, pad = _spin_avg_case("theta", theta)
        pad.data[3:11, 4] = np.nan  # spin 1 without data in one bin

        result = mms.feeps_pad_spinavg(pad, spin_sectors)

        # One row per complete spin, at its first sample, on the PAD bins: the
        # windows used to be shifted by one sample and interpolated onto 12 bin
        # edges, and the partial spins were kept
        expected = np.tile(np.arange(1.0, 6.0)[:, None], (1, 11))
        expected[0, 4] = np.nan
        np.testing.assert_array_equal(result.data, expected)
        np.testing.assert_array_equal(result.theta.data, theta)
        np.testing.assert_array_equal(
            result.time.data, pad.time.data[[3, 11, 19, 27, 35]]
        )

        with self.assertWarns(FutureWarning):
            mms.feeps_pad_spinavg(pad, spin_sectors, 16.3636)

        with self.assertRaises(ValueError):
            mms.feeps_pad_spinavg(pad, spin_sectors[:-1])


@ddt
class FeepsPitchAnglesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pitch_angles_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pitch_angles(
            feeps_alle, generate_ts(64.0, 100, tensor_order=1)
        )
        self.assertIsInstance(result[0], xr.DataArray)

    def test_feeps_pitch_angles_interpolation(self):
        # B turning by 45 degrees per FEEPS sample, at 10 times the FEEPS rate
        feeps_alle = generate_feeps(64.0, 100, "srvy", "electron", "l2", "flux", 1)
        time = generate_timeline(640.0, 1000)
        phi = np.deg2rad(4.5) * np.arange(1000)
        b_xyz = np.stack([np.cos(phi), np.sin(phi), 0.5 * np.ones(1000)], axis=1)
        b_fast = xr.DataArray(
            b_xyz, coords=[time, ["x", "y", "z"]], dims=["time", "comp"]
        )

        # The angles at the FEEPS times are those of B at these times (they
        # used to be averaged over the sample, while B turns)
        result, _ = mms.feeps_pitch_angles(feeps_alle, b_fast)
        expected, _ = mms.feeps_pitch_angles(feeps_alle, b_fast[::10])

        np.testing.assert_array_equal(result.time.data, feeps_alle.time.data)
        np.testing.assert_allclose(result.data, expected.data, atol=1e-10)

    def test_feeps_pitch_angles_along_eye(self):
        # B along the electron top-5 eye (and against bottom-5): 0 and 180
        # degrees, not NaN from cos slightly beyond 1 (it used to be NaN)
        module = importlib.import_module("pyrfu.mms.feeps_pitch_angles")
        v_eye = -np.matmul(module.t_top, module.v_fcs[5])
        b_bcs = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 1), 50.0 * v_eye[None, :] / np.linalg.norm(v_eye)
        )

        pas = module._calc_pas("electron", b_bcs)[0]

        self.assertFalse(np.isnan(pas).any())
        self.assertEqual(pas[4], 0.0)
        self.assertEqual(pas[13], 180.0)


@ddt
class FeepsRemoveBadDataTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_bad_data_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_remove_bad_data(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)

    def test_feeps_remove_bad_data_input_unchanged(self):
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_remove_bad_data(feeps_alle)

        # Bad eye and bad lowest channel are NaN in the output only
        self.assertTrue(np.isnan(result["top-1"].data).all())
        self.assertTrue(np.isnan(result["top-2"].data[:, 0]).all())
        xr.testing.assert_identical(feeps_alle, feeps_ref)

    @data(
        (1, "2019-01-01T00:00:00", {"top-6": 2, "bottom-6": 2, "bottom-8": 1}),
        (
            2,
            "2017-07-11T22:00:00",
            {"top-6": 1, "top-7": 15, "top-8": 2, "bottom-7": 15},
        ),
        (
            2,
            "2019-01-01T00:00:00",
            {"top-6": 1, "top-7": 15, "top-8": 2, "bottom-7": 2},
        ),
        # MMS4 bottom-6 and bottom-7: the 3 lowest channels are bad
        (
            4,
            "2019-01-01T00:00:00",
            {
                "top-6": 2,
                "top-7": 15,
                "top-8": 1,
                "bottom-6": 3,
                "bottom-7": 3,
                "bottom-8": 2,
            },
        ),
    )
    @unpack
    def test_feeps_remove_bad_data_ions(self, mms_id, t_start, n_bad):
        # Ion eyes (6-8) used to be skipped: bad eyes and channels kept
        feeps_alle = generate_feeps(64.0, 100, "brst", "ion", "l2", "flux", mms_id)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        time = feeps_alle.time.data
        time = time - time[0] + np.datetime64(t_start, "ns")
        feeps_alle = feeps_alle.assign_coords(time=time)

        result = mms.feeps_remove_bad_data(feeps_alle)

        # Number of lowest channels set to NaN in each eye (15: bad eye)
        for k in filter(lambda x: x[:3] in ["top", "bot"], result):
            n_nan = int(np.isnan(result[k].data).all(axis=0).sum())
            self.assertEqual(n_nan, n_bad.get(k, 0), k)

    def test_feeps_remove_bad_data_electrons(self):
        # MMS4 electrons in 2019 (2018-10-01 table): bad eyes top-1, bottom-4
        # and bottom-11, bad channel 0 of the other eyes but bottom-9 (0-1);
        # the bad ion channels 0-2 are skipped
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 4)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        time = feeps_alle.time.data
        time = time - time[0] + np.datetime64("2019-01-01T00:00:00", "ns")
        feeps_alle = feeps_alle.assign_coords(time=time)

        result = mms.feeps_remove_bad_data(feeps_alle)

        n_bad = {"top-1": 15, "bottom-4": 15, "bottom-11": 15, "bottom-9": 2}
        for k in filter(lambda x: x[:3] in ["top", "bot"], result):
            n_nan = int(np.isnan(result[k].data).all(axis=0).sum())
            self.assertEqual(n_nan, n_bad.get(k, 1), k)

    def test_feeps_remove_bad_data_nan_energies(self):
        # mms2 top-5 is a good eye in the 2018-10-01 table, without energy
        # calibration: its bad lowest channel used to be kept
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 2)
        eye = feeps_alle["top-5"].rename(energy="energy-top-5")
        eye = eye.assign_coords({"energy-top-5": np.full(eye.shape[1], np.nan)})
        feeps_alle = feeps_alle.drop_vars("top-5").assign({"top-5": eye})

        result = mms.feeps_remove_bad_data(feeps_alle)

        self.assertTrue(np.isnan(result["top-5"].data[:, 0]).all())
        self.assertTrue(np.isfinite(result["top-5"].data[:, 1:]).all())

    @data(
        ("2017-07-11T22:00:00", ["top-12", "top-5"]),
        ("2018-04-01T11:00:00", ["top-12", "top-5"]),
        ("2018-04-01T13:00:00", ["bottom-12", "bottom-2", "top-12"]),
        ("2018-10-01T00:00:00", ["bottom-12", "bottom-2", "top-12"]),
        ("2024-03-15T10:00:00", ["bottom-12", "bottom-2", "top-12"]),
    )
    @unpack
    def test_feeps_remove_bad_data_table(self, t_start, bad_eyes):
        # mms2 bad eyes: 2017-10-01 table before 2018-04-01 12:00 UTC (half way
        # to the next table), 2018-10-01 table after, as in pyspedas
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 2)
        time = feeps_alle.time.data
        time = time - time[0] + np.datetime64(t_start, "ns")
        feeps_alle = feeps_alle.assign_coords(time=time)

        result = mms.feeps_remove_bad_data(feeps_alle)

        eyes = [k for k in result if k.startswith(("top", "bottom"))]
        all_nan = sorted(k for k in eyes if np.isnan(result[k].data).all())
        self.assertListEqual(all_nan, bad_eyes)


@ddt
class ReadFeepsSectorMasksCsvTestCase(unittest.TestCase):
    sun_path = os.path.join(os.path.dirname(mms.__file__), "sun")

    def _table(self, mms_id, file_date):
        file_name = f"MMS{mms_id:d}_FEEPS_ContaminatedSectors_{file_date}.csv"
        file_path = os.path.join(self.sun_path, file_name)
        return np.loadtxt(file_path, delimiter=",", encoding="utf-8-sig")

    @data(
        ("2017-07-11T22:00:00", "20170531"),
        ("2020-01-01T00:00:00", "20181005"),
        ("2024-03-15T10:00:00", "20240202"),
        ("2025-06-01T00:00:00", "20250814"),
        ("2026-06-01T00:00:00", "20251226"),
    )
    @unpack
    def test_read_feeps_sector_masks_csv_date(self, t_start, file_date):
        # Masks from the file nearest to the start time; the list used to
        # stop at 20181005
        masks = mms.read_feeps_sector_masks_csv([t_start, t_start])

        for mms_id in range(1, 5):
            table = self._table(mms_id, file_date)
            for i in range(12):
                top = list(np.where(table[:, i] == 1)[0])
                bottom = list(np.where(table[:, i + 12] == 1)[0])
                self.assertListEqual(masks[f"mms{mms_id:d}_imask_top-{i + 1:d}"], top)
                self.assertListEqual(
                    masks[f"mms{mms_id:d}_imask_bottom-{i + 1:d}"], bottom
                )

    def test_read_feeps_sector_masks_csv_files(self):
        files = os.listdir(self.sun_path)
        file_dates = sorted({f[-12:-4] for f in files if f.endswith(".csv")})
        self.assertEqual(file_dates[-1], "20251226")

        # 64 sectors x 24 eyes of 0/1 for each spacecraft and date
        for file_date in file_dates:
            for mms_id in range(1, 5):
                table = self._table(mms_id, file_date)
                self.assertTupleEqual(table.shape, (64, 24))
                self.assertTrue(np.isin(table, [0, 1]).all())

        # MMS4 20250216 table as updated upstream in April 2025
        self.assertEqual(int(self._table(4, "20250216").sum()), 564)


@ddt
class FeepsRemoveSunTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_sun(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_remove_sun(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)

    def test_feeps_remove_sun_input_unchanged(self):
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_remove_sun(feeps_alle)

        # Masked sectors are NaN in the output only
        self.assertTrue(np.isnan(result["top-5"].data).any())
        xr.testing.assert_identical(feeps_alle, feeps_ref)


class FeepsRemoveSunlitSectorsTestCase(unittest.TestCase):
    def test_feeps_remove_sunlit_sectors(self):
        feeps_alle = generate_feeps(64.0, 640, "brst", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_alle["spinsectnum"].data = np.arange(640) % 64

        eyes = [k for k in feeps_alle if k.startswith(("top", "bottom"))]
        rng = np.random.default_rng(1)
        for k in eyes:
            feeps_alle[k].data = rng.uniform(1.0, 2.0, feeps_alle[k].shape)

        # Two sunlit eye-sector pairs, much brighter than the others
        feeps_alle["top-1"].data[feeps_alle["spinsectnum"].data == 10] = 1e3
        feeps_alle["bottom-3"].data[feeps_alle["spinsectnum"].data == 20] = 1e3
        feeps_ref = feeps_alle.copy(deep=True)

        # Quantile between the two sunlit sums and the others
        n_cells = 64 * len(eyes)
        result = mms.feeps_remove_sunlit_sectors(
            feeps_alle, outliers_q=(n_cells - 2.5) / (n_cells - 1)
        )

        sunlit = {"top-1": 10, "bottom-3": 20}
        for k in eyes:
            mask = feeps_alle["spinsectnum"].data == sunlit.get(k, -1)
            self.assertTrue(np.isnan(result[k].data[mask]).all())
            np.testing.assert_array_equal(
                result[k].data[~mask], feeps_alle[k].data[~mask]
            )

        xr.testing.assert_identical(feeps_alle, feeps_ref)

    def test_feeps_remove_sunlit_sectors_survey(self):
        # Survey samples are labelled with 8 of the 64 sectors
        feeps_alle = generate_feeps(64.0, 800, "srvy", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_alle["spinsectnum"].data = np.tile(np.arange(7, 64, 8), 100)

        eyes = [k for k in feeps_alle if k.startswith(("top", "bottom"))]
        rng = np.random.default_rng(2)
        for k in eyes:
            feeps_alle[k].data = rng.uniform(1.0, 2.0, feeps_alle[k].shape)

        sunlit = feeps_alle["spinsectnum"].data == 15
        feeps_alle[eyes[0]].data[sunlit] = 1e3

        # Quantile between the sunlit average and the others, over the 8
        # sectors with data (the 56 empty sectors used to count as zeros, and 4
        # more eye-sector pairs were masked)
        n_cells = 8 * len(eyes)
        result = mms.feeps_remove_sunlit_sectors(
            feeps_alle, outliers_q=(n_cells - 1.5) / (n_cells - 1)
        )

        self.assertTrue(np.isnan(result[eyes[0]].data[sunlit]).all())
        n_nans = sum(int(np.isnan(result[k].data).sum()) for k in eyes)
        self.assertEqual(n_nans, np.isnan(result[eyes[0]].data[sunlit]).size)

        # Sunlit in the first channel only: found with i_en=0
        feeps_alle[eyes[0]].data[sunlit, 1:] = 1.5
        result = mms.feeps_remove_sunlit_sectors(
            feeps_alle, outliers_q=(n_cells - 1.5) / (n_cells - 1), i_en=0
        )
        self.assertTrue(np.isnan(result[eyes[0]].data[sunlit]).all())


def _pad_case(data, pitch=None):
    # Pitch-angle distribution (time, energy, pitch angle) of ions in
    # s^3/cm^6, as get_pitch_angle_dist, with energies 100 * 2^i eV of widths
    # +/- 25 % and pitch-angle bins of 15 degrees (default)
    n_t, n_e, n_pa = data.shape
    pitch = 7.5 + 15.0 * np.arange(n_pa) if pitch is None else np.asarray(pitch)
    energy = np.tile(100.0 * 2.0 ** np.arange(n_e), (n_t, 1))
    pad = xr.Dataset(
        {
            "data": (["time", "idx0", "idx1"], data),
            "energy": (["time", "idx0"], energy),
            "theta": (["time", "idx1"], np.tile(pitch, (n_t, 1))),
            "time": generate_timeline(1.0, n_t),
            "idx0": np.arange(n_e),
            "idx1": np.arange(n_pa),
        }
    )
    pad.attrs = {
        "species": "ions",
        "delta_energy_minus": 0.25 * energy,
        "delta_energy_plus": 0.25 * energy,
        "delta_pitchangle_minus": np.full(n_pa, 7.5),
        "delta_pitchangle_plus": np.full(n_pa, 7.5),
    }
    pad.data.attrs["UNITS"] = "s^3/cm^6"
    return pad


def _feeps_skymap_case(data_rate, n_pts, value=2.0, mms_id=1):
    # Corrected and split eyes at the same intensity, spin sectors of 64
    # samples per spin (burst) or 8 labelled 7, 15, ..., 63 (survey)
    feeps_alle = generate_feeps(
        64.0, n_pts, data_rate, "electron", "l2", "flux", mms_id
    )
    feeps_alle = mms.feeps_correct_energies(feeps_alle)
    feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

    if data_rate == "brst":
        sectors = np.arange(n_pts) % 64
    else:
        sectors = np.tile(np.arange(7, 64, 8), n_pts // 8 + 1)[:n_pts]

    feeps_alle["spinsectnum"].data = sectors

    for k in filter(lambda x: x.startswith(("top", "bottom")), feeps_alle):
        feeps_alle[k].data = np.full(feeps_alle[k].shape, value)

    return feeps_alle


@ddt
class FeepsSkymapTestCase(unittest.TestCase):
    @data(("brst", 128, 32), ("srvy", 16, 8))
    @unpack
    def test_feeps_skymap_uniform(self, data_rate, n_pts, n_phi):
        feeps_alle = _feeps_skymap_case(data_rate, n_pts)
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_skymap(feeps_alle, to_psd=False)

        # one skymap per spin, every bin filled with the intensity
        self.assertTupleEqual(result.data.shape, (2, 15, n_phi, 4))
        np.testing.assert_array_equal(result.data.data, 2.0)
        np.testing.assert_array_equal(result.theta.data, [5.0, 65.0, 115.0, 175.0])
        np.testing.assert_array_equal(
            result.time.data, feeps_alle.time.data[[0, n_pts // 2]]
        )
        np.testing.assert_allclose(result.attrs["coverage"], 1.0)
        self.assertEqual(result.attrs["angle_integration"], "exact")
        self.assertEqual(result.phi.attrs["COORDINATE_SYSTEM"], "DMPA")
        xr.testing.assert_identical(feeps_alle, feeps_ref)

    def test_feeps_skymap_complete_spins(self):
        # Partial spin (sectors 31-63), a complete spin, partial spin (7-47):
        # one skymap, at the first sample of the complete spin (the partial
        # spins gave skymaps too)
        feeps_alle = _feeps_skymap_case("srvy", 24).isel(time=slice(3, 22))
        result = mms.feeps_skymap(feeps_alle, to_psd=False)

        np.testing.assert_array_equal(result.time.data, feeps_alle.time.data[[5]])
        np.testing.assert_array_equal(result.data.data, 2.0)

    @data(("brst", 10, 32, 10.0), ("srvy", 1, 8, 15.0 - 3.5))
    @unpack
    def test_feeps_skymap_direction(self, data_rate, i_sample, n_phi, sector):
        # A single sample of top-3 lands at its look direction: polar angle 65
        # degrees, azimuth in BCS plus the spin phase at the centre of the
        # accumulation (survey: 3.5 sectors before the label 15)
        feeps_alle = _feeps_skymap_case(data_rate, 64)
        for k in filter(lambda x: x.startswith(("top", "bottom")), feeps_alle):
            feeps_alle[k].data[:] = np.nan
        feeps_alle["top-3"].data[i_sample] = 1.0

        result = mms.feeps_skymap(feeps_alle, to_psd=False)

        phi = (22.5 + 5.625 * sector - 75.94) % 360.0
        i_phi = int(phi // (360.0 / n_phi))
        finite = np.argwhere(np.isfinite(result.data.data).all(axis=1))
        self.assertListEqual(finite.tolist(), [[0, i_phi, 1]])
        np.testing.assert_array_equal(result.data.data[0, :, i_phi, 1], 1.0)
        # one bin of the 35-90 degree band
        self.assertAlmostEqual(
            result.attrs["coverage"][0, 0], np.cos(np.deg2rad(35)) / 2 / n_phi
        )

    def test_feeps_skymap_fgm_phase(self):
        # B rotating at the spin phase of the sectors (1 spin per second)
        feeps_alle = _feeps_skymap_case("brst", 128)
        feeps_alle["top-3"].data[::3] = 5.0
        expected = mms.feeps_skymap(feeps_alle, to_psd=False)

        time = generate_timeline(512.0, 1024)
        t_s = (time - time[0]).astype(float) / 1e9

        def b_fields(shift, b_dmpa=(20.0, 0.0, 5.0)):
            psi = np.deg2rad(-75.94 + 360.0 * t_s + shift)
            b_x, b_y, b_z = b_dmpa
            b_bcs = np.stack(
                [
                    np.cos(psi) * b_x + np.sin(psi) * b_y,
                    -np.sin(psi) * b_x + np.cos(psi) * b_y,
                    np.full(len(t_s), b_z),
                ],
                axis=1,
            )
            b_dmpa = np.tile(b_dmpa, (len(t_s), 1))
            return pyrf.ts_vec_xyz(time, b_bcs), pyrf.ts_vec_xyz(time, b_dmpa)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = mms.feeps_skymap(feeps_alle, *b_fields(0.0), to_psd=False)
            # B along the spin axis: no FGM phase, the sectors are used
            along_z = mms.feeps_skymap(
                feeps_alle, *b_fields(0.0, (0.0, 0.0, 20.0)), to_psd=False
            )

        xr.testing.assert_identical(result.data, expected.data)
        xr.testing.assert_identical(along_z.data, expected.data)
        self.assertEqual(result.attrs["spin_phase"], "FGM (b_bcs, b_dmpa)")
        self.assertIn("B along the spin axis", along_z.attrs["spin_phase"])

        with self.assertWarnsRegex(UserWarning, "differs from the spin sectors"):
            mms.feeps_skymap(feeps_alle, *b_fields(3.0), to_psd=False)

    def test_feeps_skymap_survey_fgm_widths(self):
        # Survey: 8 samples per spin, labelled 7, 15, ..., 63, the centre of the
        # accumulation 3.5 sectors before the label
        feeps_alle = _feeps_skymap_case("srvy", 64)
        feeps_alle["top-3"].data[::3] = 5.0
        expected = mms.feeps_skymap(feeps_alle, to_psd=False)

        time = generate_timeline(512.0, 512)
        t_s = (time - time[0]).astype(float) / 1e9
        psi = np.deg2rad(-75.94 + 39.375 + 360.0 * 8.0 * t_s)
        b_bcs = np.stack(
            [20.0 * np.cos(psi), -20.0 * np.sin(psi), np.full(len(t_s), 5.0)], axis=1
        )
        b_dmpa = np.tile([20.0, 0.0, 5.0], (len(t_s), 1))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = mms.feeps_skymap(
                feeps_alle,
                pyrf.ts_vec_xyz(time, b_bcs),
                pyrf.ts_vec_xyz(time, b_dmpa),
                to_psd=False,
            )

        self.assertEqual(result.attrs["spin_phase"], "FGM (b_bcs, b_dmpa)")
        np.testing.assert_allclose(result.data.data, expected.data.data)

        # Energy widths from the channel bounds of the eyes
        energy = expected.energy.data[0] / 1e3
        for k in filter(lambda x: x.startswith(("top", "bottom")), feeps_alle):
            dim = feeps_alle[k].dims[1]
            feeps_alle = feeps_alle.assign_coords(
                {
                    f"energy_lower_bound_{k}": (dim, 0.9 * energy),
                    f"energy_upper_bound_{k}": (dim, 1.2 * energy),
                }
            )

        result = mms.feeps_skymap(feeps_alle, to_psd=False)
        np.testing.assert_allclose(
            result.attrs["delta_energy_minus"], 0.1 * result.energy.data
        )
        np.testing.assert_allclose(
            result.attrs["delta_energy_plus"], 0.2 * result.energy.data
        )

    def test_feeps_skymap_coverage_fill(self):
        # No data on the 5 degree ring (top-5)
        feeps_alle = _feeps_skymap_case("brst", 128)
        feeps_alle["top-5"].data[:] = np.nan
        feeps_alle["top-1"].data[:64] = 6.0

        result = mms.feeps_skymap(feeps_alle, to_psd=False)
        ring = mms.feeps_skymap(feeps_alle, to_psd=False, fill_empty="ring")
        omni = mms.feeps_skymap(feeps_alle, to_psd=False, fill_empty="omni")

        cap = (1.0 - np.cos(np.deg2rad(35.0))) / 2.0
        np.testing.assert_allclose(result.attrs["coverage"], 1.0 - cap)
        np.testing.assert_allclose(omni.attrs["coverage"], 1.0 - cap)
        self.assertTrue(np.isnan(result.data.data[..., 0]).all())
        self.assertTrue(np.isnan(ring.data.data[..., 0]).all())

        # "omni": the mean of all the samples of the spin, i.e. the spin
        # average of feeps_omni (top-1 at 6 in the first spin only)
        spin_avg = mms.feeps_spin_avg(
            mms.feeps_omni(feeps_alle), feeps_alle.spinsectnum.data
        )
        np.testing.assert_allclose(
            omni.data.data[..., 0], np.tile(spin_avg.data[..., None], (1, 1, 32))
        )
        self.assertTrue((omni.data.data[0, :, :, 0] > 2.0).all())
        np.testing.assert_array_equal(
            omni.data.data[..., 1:], result.data.data[..., 1:]
        )

        with self.assertRaises(ValueError):
            mms.feeps_skymap(feeps_alle, fill_empty="nearest")

    def test_feeps_skymap_psd(self):
        feeps_alle = _feeps_skymap_case("brst", 128)
        flux = mms.feeps_skymap(feeps_alle, to_psd=False)
        result = mms.feeps_skymap(feeps_alle)

        # Relativistic, in s^3/cm^6 as FPI
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6")
        np.testing.assert_allclose(
            result.data.data,
            1e-12 * mms.dpf2psd(flux, relativistic=True).data.data,
            rtol=1e-12,
        )
        self.assertEqual(result.attrs["species"], "electrons")
        self.assertTrue(result.attrs["relativistic"])
        self.assertFalse(flux.attrs["relativistic"])

        # Isotropic: psd_moments integrates the 4 rings exactly
        sc_pot = pyrf.ts_scalar(result.time.data, np.zeros(2))
        n_exact = mms.psd_moments(result, sc_pot)[0].data
        n_mid = mms.psd_moments(result, sc_pot, angle_integration="midpoint")[0].data
        np.testing.assert_allclose(n_mid / n_exact, 11.6017 / (4 * np.pi), rtol=1e-4)

        feeps_alle = feeps_alle.copy(deep=True)
        for k in filter(lambda x: x.startswith(("top", "bottom")), feeps_alle):
            feeps_alle[k].attrs["UNITS"] = "counts/s"

        with self.assertRaises(ValueError):
            mms.feeps_skymap(feeps_alle)


@ddt
class FeepsSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_omni = mms.feeps_omni(feeps_alle)

        result = mms.feeps_spin_avg(feeps_omni, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)

    def test_feeps_spin_avg_input_types(self):
        feeps_alle = generate_feeps(64.0, 100, "srvy", "electron", "l2", "flux", 1)
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_omni = mms.feeps_omni(feeps_alle)

        # Spin sectors 0..11 repeated: complete spins start at 0, 12, ..., 84
        # (the partial last one, 96-99, was kept)
        spin_starts = np.arange(0, 96, 12)

        # A DataArray used to give one row per sample (aligned on time)
        result = mms.feeps_spin_avg(feeps_omni, feeps_alle.spinsectnum)
        result_np = mms.feeps_spin_avg(feeps_omni, feeps_alle.spinsectnum.data)

        self.assertEqual(len(result.time), len(spin_starts))
        np.testing.assert_array_equal(
            result.time.data, feeps_omni.time.data[spin_starts]
        )
        xr.testing.assert_identical(result, result_np)

    def test_feeps_spin_avg_windows(self):
        # Partial spin (3 samples), 5 spins of 8 sectors, partial spin (4)
        spin_sectors, flux = _spin_avg_case("energy", [100.0, 200.0])
        flux[3:11] = np.nan  # spin 1 without data

        result = mms.feeps_spin_avg(flux, spin_sectors)

        # One row per complete spin, at its first sample: the windows used to
        # be shifted by one sample and the last two spins set to 0, and the
        # partial spins were kept
        expected = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result.data[:, 0], expected)
        np.testing.assert_array_equal(result.data[:, 1], expected)
        np.testing.assert_array_equal(
            result.time.data, flux.time.data[[3, 11, 19, 27, 35]]
        )
        self.assertDictEqual(result.attrs, flux.attrs)

        with self.assertRaises(ValueError):
            mms.feeps_spin_avg(flux, spin_sectors[:-1])

    def test_feeps_spin_avg_gap_and_errors(self):
        # A spin cut by a data gap is left out; ValueError without a complete
        # spin
        spin_sectors, flux = _spin_avg_case("energy", [100.0, 200.0])
        keep = np.r_[0:13, 16 : len(flux)]  # samples 13-15 of spin 2 missing
        result = mms.feeps_spin_avg(flux[keep], spin_sectors[keep])

        np.testing.assert_array_equal(result.data[:, 0], [1.0, 3.0, 4.0, 5.0])

        # A single complete spin (sectors 0-7) is kept
        result = mms.feeps_spin_avg(flux[3:11], spin_sectors[3:11])
        np.testing.assert_array_equal(result.time.data, flux.time.data[[3]])

        with self.assertRaisesRegex(ValueError, "no complete spin"):
            mms.feeps_spin_avg(flux[:10], spin_sectors[:10])


class FeepsSectorSpecTestCase(unittest.TestCase):
    def test_feeps_sector_spec(self):
        # Partial spin (sectors 62-63), 2 spins of 64 sectors, partial spin (0-2),
        # with the data equal to 100 * spin + sector
        spin_sectors = np.hstack([[62, 63], np.tile(np.arange(64), 2), [0, 1, 2]])
        spin = np.hstack([[0, 0], np.repeat([1, 2], 64), [3, 3, 3]])
        value = 100.0 * spin + spin_sectors
        time = generate_timeline(64.0 / 20.0, len(spin))

        # 3 energy channels around the value, and one sample without data
        top = np.stack([value - 1.0, value, value + 1.0], axis=1)
        bottom = top.copy()
        bottom[10] = np.nan
        feeps_alle = xr.Dataset(
            {
                "top-1": (["time", "energy-top-1"], top),
                "bottom-1": (["time", "energy-bottom-1"], bottom),
                "spinsectnum": (["time"], spin_sectors),
            },
            coords={"time": time},
            attrs={"mmsId": 1},
        )

        result = mms.feeps_sector_spec(feeps_alle)

        # One row per complete spin at its first sample (the spectra used to be
        # padded with NaN to every sample, row 0 empty and spins one row late,
        # and the partial spins were kept)
        self.assertListEqual(sorted(result.data_vars), ["bottom-1", "top-1"])
        np.testing.assert_array_equal(result.time.data, time[[2, 66]])
        self.assertEqual(result.sizes["sectornum"], 64)

        expected = np.full((2, 64), np.nan)
        expected[0] = 100.0 + np.arange(64)
        expected[1] = 200.0 + np.arange(64)
        np.testing.assert_array_equal(result["top-1"].data, expected)

        expected[0, 8] = np.nan
        np.testing.assert_array_equal(result["bottom-1"].data, expected)
        self.assertDictEqual(result.attrs, feeps_alle.attrs)

        # Float spin sectors (used to raise IndexError)
        feeps_alle["spinsectnum"] = feeps_alle["spinsectnum"].astype(float)
        xr.testing.assert_identical(mms.feeps_sector_spec(feeps_alle), result)

    def test_feeps_sector_spec_unsplit(self):
        # Unsplit eyes (16 channels) used to average the integral channel
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)

        with self.assertRaisesRegex(ValueError, "feeps_split_integral_ch"):
            mms.feeps_sector_spec(feeps_alle)


@ddt
class FeepsSplitIntegralChTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_split_integral_ch(self, data_rate, dtype):
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        mms.feeps_split_integral_ch(feeps_alle)

    def test_feeps_split_integral_ch_other_variables(self):
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        eyes = [k for k in feeps_alle if k.startswith(("top", "bottom"))]

        # Pitch angles (one column per eye) and a 1-D variable
        time = feeps_alle.time.data
        feeps_alle["pitch_angle"] = (["time", "eye"], np.ones((len(time), 18)))
        feeps_alle["other"] = (["time"], np.arange(len(time), dtype=float))

        out, out_500kev = mms.feeps_split_integral_ch(feeps_alle)

        # Only the eyes are split: pitch_angle used to lose its last column
        # and the 1-D variables other than spinsectnum used to be dropped
        for k in eyes:
            np.testing.assert_array_equal(out[k].data, feeps_alle[k].data[:, :-1])
            np.testing.assert_array_equal(out_500kev[k].data, feeps_alle[k].data[:, -1])

        for k in ["spinsectnum", "pitch_angle", "other"]:
            xr.testing.assert_identical(out[k], feeps_alle[k])

        self.assertListEqual(
            sorted(out_500kev.data_vars), sorted([*eyes, "spinsectnum"])
        )


class FkPowerSpectrumTestCase(unittest.TestCase):
    # A plane wave of potential cos(k.r - 2 pi f t) seen by the six probes of a
    # spacecraft spinning in 20 s, sampled with the channel delays of the
    # burst mode (as v6_edp_brst_l2)
    F_S = 8192.0
    T_0 = np.datetime64("2019-01-01T00:00:00", "ns")

    def _probes(self, t_s, phase_0):
        # Positions (m) of the six probes at the times t_s (s), probe k in the
        # spin plane at the spin phase + 30, 210, 120 and 300 deg
        phase = np.deg2rad(phase_0 + 18.0 * t_s)
        spin_plane = [
            60.0 * np.column_stack([np.cos(phase + p), np.sin(phase + p), 0 * t_s])
            for p in np.deg2rad([30.0, 210.0, 120.0, 300.0])
        ]
        axial = [np.tile([0.0, 0.0, z], (len(t_s), 1)) for z in [14.6, -14.6]]
        return spin_plane + axial

    def _inputs(self, k_vec, f_0, b_xyz, phase_0=0.0):
        t_s = np.arange(int(2.5 * self.F_S)) / self.F_S

        def pot(k, delay):
            r_k = self._probes(t_s + delay, phase_0)[k]
            return -5.0 + np.cos(r_k @ k_vec - 2 * np.pi * f_0 * (t_s + delay))

        # V1, V3 and V5, and V1 - V2, V3 - V4, V5 - V6, sampled after V1
        d_12, d_34, d_56 = [
            pot(k, d) - pot(k + 1, d)
            for k, d in [(0, 26.703e-6), (2, 30.518e-6), (4, 34.332e-6)]
        ]
        v_3, v_5 = pot(2, 7.629e-6), pot(4, 15.259e-6)
        v_1 = pot(0, 0.0)
        v6 = xr.DataArray(
            np.column_stack([v_1, v_1 - d_12, v_3, v_3 - d_34, v_5, v_5 - d_56]),
            coords=[generate_timeline(self.F_S, len(t_s)), np.arange(1, 7)],
            dims=["time", "probe"],
        )
        t_b = np.arange(0.0, 2.5, 1 / 128)
        b_xyz = pyrf.ts_vec_xyz(
            self.T_0 + (t_b * 1e9).astype("timedelta64[ns]"),
            np.tile(b_xyz, (len(t_b), 1)),
        )
        t_z = np.arange(0.0, 2.5, 0.25)
        z_phase = pyrf.ts_scalar(
            self.T_0 + (t_z * 1e9).astype("timedelta64[ns]"),
            (phase_0 + 18.0 * t_z) % 360.0,
        )
        return v6, b_xyz, z_phase

    def _tint(self, t_1, t_2):
        times = self.T_0 + (np.array([t_1, t_2]) * 1e9).astype("timedelta64[ns]")
        return list(pyrf.datetime642iso8601(times))

    def _peak(self, power):
        # Frequency and wave-number bin [k, k + dk) of the maximum
        i_f, i_k = np.unravel_index(np.nanargmax(power.data), power.shape)
        return power.f.data[i_f], power.k.data[i_k], np.diff(power.k.data[:2])[0]

    def test_fk_power_spectrum_axial(self):
        # Probes 5 and 6: k along z, B along +z or -z; k > 0 along B
        k_vec = np.array([0.0, 0.0, 2 * np.pi / 200.0])

        for sign in [1.0, -1.0]:
            v6, b_xyz, _ = self._inputs(k_vec, 500.0, [0.0, 0.0, 20.0 * sign])

            for field in [True, False]:
                power = mms.fk_power_spectrum(
                    v6, b_xyz, self._tint(1.0, 1.5), 5, field=field, n_f=50
                )
                f_peak, k_peak, d_k = self._peak(power)

                self.assertLessEqual(k_peak, sign * k_vec[2])
                self.assertLess(sign * k_vec[2], k_peak + d_k)
                self.assertLess(abs(f_peak / 500.0 - 1.0), 0.1)

        # Wave numbers +-pi / 14.6 m (potentials: 29.2 m), normalised power
        self.assertEqual(power.dims, ("f", "k"))
        self.assertEqual(power.k.size, 101)
        self.assertAlmostEqual(power.k.data[0], -np.pi / 29.2)
        self.assertEqual(np.nanmax(power.data), 1.0)
        self.assertGreaterEqual(np.nanmin(power.data), 1e-3)
        self.assertEqual(power.k.attrs["UNITS"], "rad/m")

    def test_fk_power_spectrum_spin_plane(self):
        # Probes 1 and 2 (3 and 4) along x at 1.25 s, k and B along x: the
        # other probes are perpendicular to k
        k_vec = np.array([2 * np.pi / 1000.0, 0.0, 0.0])

        for probe, phase_0 in [(1, -30.0 - 22.5), (3, -120.0 - 22.5)]:
            for sign in [1.0, -1.0]:
                v6, b_xyz, z_phase = self._inputs(
                    k_vec, 200.0, [20.0 * sign, 0.0, 0.0], phase_0
                )

                for field in [True, False]:
                    power = mms.fk_power_spectrum(
                        v6,
                        b_xyz,
                        self._tint(1.2, 1.3),
                        probe,
                        z_phase=z_phase,
                        field=field,
                        n_f=50,
                    )
                    _, k_peak, d_k = self._peak(power)

                    self.assertLessEqual(k_peak, sign * k_vec[0])
                    self.assertLess(sign * k_vec[0], k_peak + d_k)

    def test_fk_power_spectrum_errors(self):
        v6, b_xyz, z_phase = self._inputs(np.zeros(3), 200.0, [0.0, 0.0, 20.0])
        tint = self._tint(1.0, 1.5)

        with self.assertRaisesRegex(ValueError, "probe must be"):
            mms.fk_power_spectrum(v6, b_xyz, tint, 2, z_phase=z_phase)

        with self.assertRaisesRegex(ValueError, "z_phase"):
            mms.fk_power_spectrum(v6, b_xyz, tint, 1)

        with self.assertRaisesRegex(ValueError, "at least"):
            mms.fk_power_spectrum(v6, b_xyz, self._tint(1.0, 1.01), 5)


@ddt
class FkPowerSpectrum4scTestCase(unittest.TestCase):
    @data((None, None), (random.random(), None), (None, [0.1, 1]))
    @unpack
    def test_fk_power_spectrum_4sc(self, df, f_range):
        e_mms = [generate_ts(64.0, 100, tensor_order=0) for _ in range(4)]
        r_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]

        result = mms.fk_power_spectrum_4sc(
            e_mms, r_mms, b_mms, TEST_TINT, df=df, f_range=f_range
        )
        self.assertIsInstance(result, xr.Dataset)

    @staticmethod
    def _plane_wave(k_vec, f_0=4.0, f_s=128.0, n_pts=1024):
        # cos(2 pi f_0 t - k.r) on a regular tetrahedron of 10 km side pairs
        t_s = np.arange(n_pts) / f_s
        time = np.datetime64("2019-01-01T00:00:00", "ns")
        time = time + (t_s * 1e9).astype("timedelta64[ns]")
        dr_mms = 5.0 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        r_mms = [
            pyrf.ts_vec_xyz(time, np.tile(dr + [6.4e4, 0, 0], (n_pts, 1)))
            for dr in dr_mms
        ]
        e_mms = [
            pyrf.ts_scalar(time, np.cos(2 * np.pi * f_0 * t_s - k_vec @ dr))
            for dr in dr_mms
        ]
        b_mms = [
            pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 20.0], (n_pts, 1))) for _ in dr_mms
        ]
        tint = list(np.datetime_as_string(time[[n_pts // 4, 3 * n_pts // 4]], "ns"))
        return e_mms, r_mms, b_mms, tint

    def test_fk_power_spectrum_4sc_plane_wave(self):
        # k in rad/km in, rad/m out; recovered within one k bin
        k_vec = np.array([0.04, -0.02, 0.03])
        e_mms, r_mms, b_mms, tint = self._plane_wave(k_vec)
        result = mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint, f_range=[3, 5])
        d_k = float(result.k_x[1] - result.k_x[0])
        k_ref = {
            "k_x_avg": k_vec[0],
            "k_y_avg": k_vec[1],
            "k_z_avg": k_vec[2],
            "k_para_avg": k_vec[2],
            "k_perp_avg": np.hypot(k_vec[0], k_vec[1]),
        }
        for key, value in k_ref.items():
            self.assertLess(abs(result.attrs[key] - value / 1e3), d_k, key)
        self.assertAlmostEqual(result.attrs["f_avg"], 4.0, delta=0.1)

    def test_fk_power_spectrum_4sc_k_bins(self):
        # The wave numbers are the centres of the bins of the power maps
        # (linspace(-k_max, k_max, num_k) was up to half a bin off): the peak
        # of each map within half a bin of the wave vector
        k_vec = np.array([0.04, -0.02, 0.03])
        e_mms, r_mms, b_mms, tint = self._plane_wave(k_vec)
        result = mms.fk_power_spectrum_4sc(
            e_mms, r_mms, b_mms, tint, num_k=20, f_range=[3, 5]
        )
        d_k = float(result.k_x[1] - result.k_x[0])
        d_k_mag = float(result.k_mag[1] - result.k_mag[0])

        self.assertAlmostEqual(float(result.k_mag[0]), d_k_mag / 2)
        self.assertAlmostEqual(
            float(result.k_x[-1]) + d_k / 2, float(result.k_mag[-1]) + d_k_mag / 2
        )

        for name, value in zip(
            ["k_x", "k_y", "k_z", "k_mag"], [*k_vec, np.linalg.norm(k_vec)]
        ):
            power = np.nan_to_num(result[f"{name}_f"].data)
            i_k = np.unravel_index(np.argmax(power), power.shape)[0]
            width = d_k_mag if name == "k_mag" else d_k
            self.assertLessEqual(abs(result[name].data[i_k] - value / 1e3), width / 2)

    def test_fk_power_spectrum_4sc_vector(self):
        # Vector input raised AttributeError; the first component is used
        e_mms, r_mms, b_mms, tint = self._plane_wave(np.array([0.04, -0.02, 0.03]))
        e_vec = [
            pyrf.ts_vec_xyz(e_.time.data, np.stack([e_.data, -e_.data, 0 * e_.data], 1))
            for e_ in e_mms
        ]
        result = mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint)
        result_vec = mms.fk_power_spectrum_4sc(e_vec, r_mms, b_mms, tint)
        xr.testing.assert_identical(result_vec, result)

    def test_fk_power_spectrum_4sc_position_gap(self):
        # A gap raised ValueError (array must not contain infs or NaNs); the
        # windows without position are left out
        e_mms, r_mms, b_mms, tint = self._plane_wave(np.array([0.04, -0.02, 0.03]))
        r_gap = [r_.copy() for r_ in r_mms]
        r_gap[2].data[400:500] = np.nan
        result = mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint)
        result_gap = mms.fk_power_spectrum_4sc(e_mms, r_gap, b_mms, tint)
        for key in ["k_x_avg", "k_y_avg", "k_z_avg", "f_avg"]:
            self.assertAlmostEqual(result_gap.attrs[key], result.attrs[key])

        r_nan = [r_.copy() for r_ in r_mms]
        for r_ in r_nan:
            r_.data[:] = np.nan
        with self.assertRaises(ValueError):
            mms.fk_power_spectrum_4sc(e_mms, r_nan, b_mms, tint)

    def test_fk_power_spectrum_4sc_dims(self):
        e_mms, r_mms, b_mms, tint = self._plane_wave(np.array([0.04, -0.02, 0.03]))
        result = mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint)
        self.assertTupleEqual(result.k_y_f.dims, ("k_y", "f"))
        self.assertTupleEqual(result.k_z_f.dims, ("k_z", "f"))

    def test_fk_power_spectrum_4sc_no_complex_warning(self):
        # The power was cast from complex to float (ComplexWarning)
        e_mms, r_mms, b_mms, tint = self._plane_wave(np.array([0.04, -0.02, 0.03]))

        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint)

    def test_fk_power_spectrum_4sc_field_gap(self):
        # A NaN gap in the field made every wavelet coefficient NaN, so no
        # wave vector was left; the windows around the gap are left out
        k_vec = np.array([0.04, -0.02, 0.03])
        e_mms, r_mms, b_mms, tint = self._plane_wave(k_vec)
        e_mms[1].data[400:440] = np.nan
        result = mms.fk_power_spectrum_4sc(e_mms, r_mms, b_mms, tint, f_range=[3, 5])
        d_k = float(result.k_x[1] - result.k_x[0])
        for key, value in zip(["k_x_avg", "k_y_avg", "k_z_avg"], k_vec):
            self.assertLess(abs(result.attrs[key] - value / 1e3), d_k, key)


@ddt
class GetDataDownloadTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_data")

    def test_get_file_content_aws_error(self):
        # S3 errors are raised (they were logged, then gave UnboundLocalError)
        s3_object = mock.Mock(key="mms1/file.cdf")
        s3_object.get.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "We encountered an error"}},
            "GetObject",
        )

        with self.assertLogs("pyrfu", level="ERROR"), self.assertRaises(ClientError):
            self.module._get_file_content_sources("aws", s3_object)

    def test_get_file_content_sdc_error(self):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("connection reset")

        with self.assertLogs("pyrfu", level="ERROR"):
            with self.assertRaises(requests.ConnectionError):
                self.module._get_file_content_sources("sdc", "url", session, {})

    def test_get_file_content_sdc_timeout(self):
        # SDC downloads time out instead of hanging on a stalled connection
        session = mock.Mock()
        session.get.return_value = mock.Mock(content=b"cdf")

        content = self.module._get_file_content_sources("sdc", "url", session, {})

        self.assertEqual(content, b"cdf")
        timeout = session.get.call_args.kwargs["timeout"]
        self.assertIsNotNone(timeout)
        self.assertTrue(all(0 < t_ < np.inf for t_ in np.atleast_1d(timeout)))

    @data(
        [],  # no file
        ["url"],  # download error
    )
    def test_get_data_keeps_sdc_session(self, file_names):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("connection reset")
        sources = (file_names, session, {})
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with self.assertRaises((FileNotFoundError, requests.ConnectionError)):
                with (
                    self.assertLogs("pyrfu", level="ERROR")
                    if file_names
                    else nullcontext()
                ):
                    mms.get_data("b_gse_fgm_srvy_l2", tint, 1, source="sdc")

        session.close.assert_not_called()  # shared, cached session


def _sdc_session(chunks=(b"cdf", b"data"), status_error=None, stream_error=None):
    # Mocked SDC session whose get() returns a streamed response
    response = mock.MagicMock()
    if status_error:
        response.raise_for_status.side_effect = status_error

    def iter_content(chunk_size=1):
        for chunk in chunks:
            yield chunk
        if stream_error:
            raise stream_error

    response.iter_content.side_effect = iter_content
    response.json.return_value = {"files": []}
    session = mock.MagicMock()
    session.get.return_value = response
    response.__enter__.return_value = response
    return session


def _write_config(test_case, local):
    # A temporary MMS configuration, never the user's
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    path = os.path.join(directory.name, "mms_config.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"local": local}, file)
    return path


class DownloadAncillaryTestCase(unittest.TestCase):
    _FILE = "MMS1_DEFATT_2019257_2019258.V00"

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.download_ancillary")
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = root.name
        files = [{"file_name": self._FILE, "url": "url", "file_size": 7}]
        for name, value in [
            ("list_files_ancillary_sdc", files),
            ("_login_lasp", (_sdc_session(), {}, "")),
        ]:
            patch = mock.patch.object(self.module, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def _check(self):
        out_file = os.path.join(self.root, "ancillary", "mms1", "defatt", self._FILE)
        with open(out_file, "rb") as file:
            self.assertEqual(file.read(), b"cdfdata")

    def test_download_ancillary(self):
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        self.module.download_ancillary("defatt", tint, 1, self.root)
        self._check()

    def test_download_ancillary_default_path(self):
        # The local data path of the configuration
        config = _write_config(self, self.root)
        with mock.patch.object(self.module, "MMS_CFG_PATH", config):
            tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
            self.module.download_ancillary("defatt", tint, 1)
        self._check()

    def test_download_ancillary_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            self.module.download_ancillary(
                "defatt", ["2019-09-14T07:54:00", "2019-09-14T08:11:00"], 1, "/bazinga"
            )

    def test_download_ancillary_defq(self):
        # One file for the four spacecraft, in ancillary/mms/defq
        name = "MMS_DEFQ_2019257_2019258.V00"
        self.module.list_files_ancillary_sdc.return_value = [
            {"file_name": name, "url": "url", "file_size": 7}
        ]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        self.module.download_ancillary("defq", tint, 3, self.root)

        with open(
            os.path.join(self.root, "ancillary", "mms", "defq", name), "rb"
        ) as file:
            self.assertEqual(file.read(), b"cdfdata")


class CopyFilesTestCase(unittest.TestCase):
    _SCI = "mms1/fgm/srvy/l2/2019/09/mms1_fgm_srvy_l2_20190914_v5.211.0.cdf"
    _ANC = "ancillary/mms1/defatt/MMS1_DEFATT_2019257_2019258.V00"
    _TINT = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

    def setUp(self):
        source, target = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.addCleanup(source.cleanup)
        self.addCleanup(target.cleanup)
        self.source, self.target = source.name, target.name

        for rel_path in [self._SCI, self._ANC]:
            path = os.path.join(self.source, *rel_path.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as file:
                file.write(b"cdfdata")

    def _check(self, rel_path):
        # Same relative path in the target, same content
        with open(os.path.join(self.target, *rel_path.split("/")), "rb") as file:
            self.assertEqual(file.read(), b"cdfdata")

    def test_copy_files(self):
        var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        mms.copy_files(var, self._TINT, 1, self.target, self.source)
        self._check(self._SCI)

    def test_copy_files_ancillary(self):
        mms.copy_files_ancillary("defatt", self._TINT, 1, self.target, self.source)
        self._check(self._ANC)

    def test_copy_files_ancillary_defq(self):
        # One file for the four spacecraft, in ancillary/mms/defq
        rel_path = "ancillary/mms/defq/MMS_DEFQ_2019257_2019258.V00"
        path = os.path.join(self.source, *rel_path.split("/"))
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as file:
            file.write(b"cdfdata")

        mms.copy_files_ancillary("defq", self._TINT, 2, self.target, self.source)
        self._check(rel_path)

    def test_copy_files_default_path(self):
        # The local data path of the configuration
        config = _write_config(self, self.source)
        var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}

        for name, call, rel_path in [
            (
                "copy_files",
                lambda: mms.copy_files(var, self._TINT, 1, self.target),
                self._SCI,
            ),
            (
                "copy_files_ancillary",
                lambda: mms.copy_files_ancillary("defatt", self._TINT, 1, self.target),
                self._ANC,
            ),
        ]:
            module = sys.modules[f"pyrfu.mms.{name}"]
            with mock.patch.object(module, "MMS_CFG_PATH", config):
                call()
            self._check(rel_path)

    def test_copy_files_missing_paths(self):
        var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        missing = os.path.join(self.target, "missing")

        for call in [
            lambda: mms.copy_files(var, self._TINT, 1, missing, self.source),
            lambda: mms.copy_files(var, self._TINT, 1, self.target, missing),
            lambda: mms.copy_files_ancillary(
                "defatt", self._TINT, 1, missing, self.source
            ),
            lambda: mms.copy_files_ancillary(
                "defatt", self._TINT, 1, self.target, missing
            ),
        ]:
            with self.assertRaises(FileNotFoundError):
                call()


class DownloadDataTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.download_data")

    def test_download_file(self):
        with tempfile.TemporaryDirectory() as root:
            out_file = os.path.join(root, "mms1", "file.cdf")
            session = _sdc_session()
            self.module._download_file(session, "url", {}, out_file, 7)

            with open(out_file, "rb") as file:
                self.assertEqual(file.read(), b"cdfdata")
            self.assertListEqual(os.listdir(os.path.dirname(out_file)), ["file.cdf"])

        timeout = session.get.call_args.kwargs["timeout"]
        self.assertTrue(all(0 < t_ < np.inf for t_ in np.atleast_1d(timeout)))

    def test_download_file_errors(self):
        # Neither an HTTP error page nor an interrupted download leaves a file
        errors = [
            {"status_error": requests.HTTPError("404")},
            {"stream_error": requests.ConnectionError("connection reset")},
        ]
        for error in errors:
            with tempfile.TemporaryDirectory() as root:
                out_file = os.path.join(root, "file.cdf")
                with self.assertRaises(requests.RequestException):
                    self.module._download_file(
                        _sdc_session(**error), "url", {}, out_file
                    )
                self.assertListEqual(os.listdir(root), [])

    def test_download_data(self):
        files = [
            {
                "url": "url",
                "file_name": "mms1_fgm_srvy_l2_20190914_v5.211.0.cdf",
                "timetag": "2019-09-14T00:00:00",
                "size": 7,
            }
        ]
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(self.module, "list_files_sdc", return_value=files):
                with mock.patch.object(
                    self.module, "_login_lasp", return_value=(_sdc_session(), {}, "")
                ):
                    self.module.download_data(
                        "b_gse_fgm_srvy_l2",
                        ["2019-09-14T07:54:00", "2019-09-14T08:11:00"],
                        1,
                        root,
                    )

            out_file = os.path.join(
                root,
                "mms1",
                "fgm",
                "srvy",
                "l2",
                "",
                "2019",
                "09",
                files[0]["file_name"],
            )
            with open(out_file, "rb") as file:
                self.assertEqual(file.read(), b"cdfdata")

    def test_list_files_sdc_request(self):
        # The SDC listing has a timeout and raises on HTTP errors
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        session = _sdc_session(status_error=requests.HTTPError("500"))
        with mock.patch.object(module, "_login_lasp", return_value=(session, {}, "x/")):
            with self.assertRaises(requests.HTTPError):
                module.list_files_sdc(tint, 1, var)

        self.assertIsNotNone(session.get.call_args.kwargs.get("timeout"))


class GetFeepsTestCase(unittest.TestCase):
    TINT = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_feeps_alleyes")

    # quality indicator of every eye at the 5 times
    quality = np.array([0, 1, 2, 3, 4])

    def _fake_read(
        self, dset_name, cdf_names, tint, verbose, data_path, source, **kwargs
    ):
        # (time, energy) time series for every variable, (time,) for the
        # quality indicators, the energy bounds without time
        times = generate_timeline(1.0, 5)
        energy = np.arange(1, 4, dtype=float)
        out = {}
        for name in kwargs.get("support_names") or []:
            out[name] = energy + (0.4 if "upper" in name else -0.4)
        for name in cdf_names:
            if "quality_indicator" in name:
                out[name] = xr.DataArray(self.quality, coords=[times], dims=["time"])
            else:
                out[name] = xr.DataArray(
                    np.ones((5, 3)), coords=[times, energy], dims=["time", "Epoch_E"]
                )
        return out

    def test_get_feeps_alleyes_source(self):
        # All the eyes are read at once (one download per file), from the source
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("fluxe_brst_l2", self.TINT, 2, source="aws")

        read.assert_called_once()
        dset_name, cdf_names = read.call_args.args[:2]
        self.assertEqual(dset_name, "mms2_feeps_brst_l2_electron")
        self.assertEqual(read.call_args.args[5], "aws")

        eyes = [k for k in out.data_vars if k not in ["spinsectnum", "pitch_angle"]]
        self.assertEqual(len(cdf_names), 2 * len(eyes) + 2)
        self.assertIn(
            "mms2_epd_feeps_brst_l2_electron_top_intensity_sensorid_3", cdf_names
        )
        self.assertIn("energy_top-3", out["top-3"].dims)
        self.assertEqual(out["top-3"].attrs["species"], "electrons")

    def test_get_feeps_alleyes_quality(self):
        # Samples with a quality indicator >= quality_flag are NaN (they used
        # to be kept, the indicators weren't read)
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("fluxe_brst_l2", self.TINT, 2)
            out_2 = mms.get_feeps_alleyes("fluxe_brst_l2", self.TINT, 2, quality_flag=2)
            out_none = mms.get_feeps_alleyes(
                "fluxe_brst_l2", self.TINT, 2, quality_flag=None
            )

        self.assertIn(
            "mms2_epd_feeps_brst_l2_electron_top_quality_indicator_sensorid_3",
            read.call_args_list[0].args[1],
        )
        self.assertFalse(
            any("quality" in name for name in read.call_args_list[2].args[1])
        )

        eyes = [k for k in out.data_vars if k not in ["spinsectnum", "pitch_angle"]]
        self.assertEqual(len(eyes), 18)

        for k in eyes:
            finite = np.isfinite(out[k].data).all(axis=1)
            np.testing.assert_array_equal(finite, [True, True, True, False, False])
            finite = np.isfinite(out_2[k].data).all(axis=1)
            np.testing.assert_array_equal(finite, [True, True, False, False, False])
            self.assertTrue(np.isfinite(out_none[k].data).all())

    def test_get_feeps_alleyes_units(self):
        # Sector masks: one value per spin sector, no quality cut (used to fail)
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("maske_brst_l2", self.TINT, 2)
            out_l1b = mms.get_feeps_alleyes("countse_brst_l1b", self.TINT, 2)

        cdf_names = read.call_args_list[0].args[1]
        self.assertFalse(any("quality" in name for name in cdf_names))
        self.assertIn(
            "mms2_epd_feeps_brst_l2_electron_top_sector_mask_sensorid_3", cdf_names
        )
        self.assertEqual(out["top-3"].dims, ("time", "sectornum"))
        self.assertTrue(np.isfinite(out["top-3"].data).all())

        # Counts are in the L1b files, read from the L1b dataset (it used to
        # read the L2 dataset for every level)
        dset_name, cdf_names = read.call_args_list[1].args[:2]
        self.assertEqual(dset_name, "mms2_feeps_brst_l1b_electron")
        self.assertIn(
            "mms2_epd_feeps_brst_l1b_electron_top_counts_sensorid_3", cdf_names
        )
        self.assertIn("energy_top-3", out_l1b["top-3"].dims)

        with self.assertRaisesRegex(ValueError, "l1b"):
            mms.get_feeps_alleyes("countse_brst_l2", self.TINT, 2)

    def test_get_feeps_alleyes_energy_bounds(self):
        # Channel bounds read once (not per time), along the eye energies
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("fluxe_brst_l2", self.TINT, 2)
            mask = mms.get_feeps_alleyes("maske_brst_l2", self.TINT, 2)

        support = read.call_args_list[0].kwargs["support_names"]
        self.assertIn(
            "mms2_epd_feeps_brst_l2_electron_top_energy_lower_bound_sensorid_3",
            support,
        )
        self.assertEqual(len(support), 36)
        self.assertListEqual(read.call_args_list[1].kwargs["support_names"], [])

        lower = out["top-3"]["energy_lower_bound_top-3"]
        self.assertTupleEqual(lower.dims, ("energy_top-3",))
        np.testing.assert_allclose(lower.data, [0.6, 1.6, 2.6])
        np.testing.assert_allclose(
            out["top-3"]["energy_upper_bound_top-3"].data, [1.4, 2.4, 3.4]
        )
        self.assertNotIn("energy_lower_bound_top-3", mask["top-3"].coords)

    def test_get_feeps_alleyes_ions(self):
        # Survey ions before 2017-08-16: eyes 6-8, from the ion dataset
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("cpsi_srvy_l2", self.TINT, 2)

        dset_name, cdf_names = read.call_args.args[:2]
        self.assertEqual(dset_name, "mms2_feeps_srvy_l2_ion")
        self.assertIn(
            "mms2_epd_feeps_srvy_l2_ion_bottom_count_rate_sensorid_8", cdf_names
        )
        eyes = [k for k in out.data_vars if k not in ["spinsectnum", "pitch_angle"]]
        self.assertListEqual(
            eyes, ["top-6", "top-7", "top-8", "bottom-6", "bottom-7", "bottom-8"]
        )
        self.assertEqual(out.attrs["dtype"], "ion")
        self.assertEqual(out["top-6"].attrs["species"], "ions")

    def test_get_feeps_alleyes_input(self):
        # Invalid species and units (the units raised a bare KeyError)
        with mock.patch.object(self.module, "_db_get_ts_dict") as read:
            with self.assertRaisesRegex(ValueError, "specie"):
                mms.get_feeps_alleyes("fluxp_brst_l2", self.TINT, 2)

            with self.assertRaisesRegex(ValueError, "Invalid units"):
                mms.get_feeps_alleyes("fluxese_brst_l2", self.TINT, 2)

        read.assert_not_called()

    def test_get_feeps_omni(self):
        # Quick look: bad data, split, sun and omni on the loaded eyes, then
        # averaged over the spins of 12 samples
        module = importlib.import_module("pyrfu.mms.get_feeps_omni")
        feeps_alle = generate_feeps(64.0, 100, "brst", "electron", "l2", "flux", 1)
        feeps_alle = mms.feeps_correct_energies(feeps_alle)
        feeps_ref = feeps_alle.copy(deep=True)

        with mock.patch.object(module, "get_feeps_alleyes", return_value=feeps_alle):
            result = mms.get_feeps_omni("fluxe_brst_l2", self.TINT, 1)
            result_spin = mms.get_feeps_omni(
                "fluxe_brst_l2", self.TINT, 1, spin_avg=True
            )

        expected, _ = mms.feeps_split_integral_ch(mms.feeps_remove_bad_data(feeps_ref))
        expected = mms.feeps_omni(mms.feeps_remove_sun(expected))
        xr.testing.assert_identical(result, expected)
        self.assertTrue(np.isfinite(result.data).any())

        expected_spin = mms.feeps_spin_avg(expected, feeps_ref.spinsectnum.data)
        xr.testing.assert_identical(result_spin, expected_spin)
        self.assertEqual(len(result_spin.time), 8)

        xr.testing.assert_identical(feeps_alle, feeps_ref)

    def test_get_feeps_omni_source(self):
        module = importlib.import_module("pyrfu.mms.get_feeps_omni")

        with mock.patch.object(
            module, "get_feeps_alleyes", side_effect=RuntimeError("stop")
        ) as alleyes:
            with self.assertRaises(RuntimeError):
                mms.get_feeps_omni("fluxe_brst_l2", self.TINT, 2, source="sdc")

        self.assertEqual(alleyes.call_args.kwargs["source"], "sdc")

    @pytest.mark.network
    @unittest.skipUnless(
        os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
    )
    def test_get_feeps_alleyes_aws(self):
        out = mms.get_feeps_alleyes(
            "fluxe_brst_l2", self.TINT, 2, verbose=False, source="aws"
        )
        self.assertEqual(len(out.data_vars), 20)
        self.assertGreater(out.sizes["time"], 1000)


def _cdf_four_columns(t_start="2019-09-14T07:54:00"):
    # In-memory CDF with (N, 4) variables labelled as in the MMS files: FGM
    # (x, y, z, r representation), MEC quaternions (qx, qy, qz, qw) and MEC
    # B fields (labels only, last one the magnitude), with padded labels
    time = np.datetime64(t_start, "ns") + np.arange(5) * np.timedelta64(1, "s")
    data = np.tile([1.0, 2.0, 3.0, 4.0], (5, 1))

    cdf = pycdfpp.CDF()
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    labels = {
        "rep_vec_tot": ["x", "y", "z", "r"],
        "rep_quat": ["qx  ", "qy  ", "qz  ", "qw  "],
        "label_b_gsm": ["b_gsm_x ", "b_gsm_y ", "b_gsm_z ", "b_gsm_mag"],
        "label_other": ["a", "b", "c", "d"],
    }
    for name, values in labels.items():
        cdf.add_variable(name, values=np.array([values]))

    for name, attrs in {
        "mms1_fgm_b_gse_srvy_l2": {"REPRESENTATION_1": "rep_vec_tot"},
        "mms1_mec_quat_eci_to_gse": {"REPRESENTATION_1": "rep_quat"},
        "mms1_mec_bsc_gsm": {"LABL_PTR_1": "label_b_gsm"},
        "mms1_mec_other": {"LABL_PTR_1": "label_other"},
    }.items():
        cdf.add_variable(name, values=data)
        cdf[name].add_attribute("DEPEND_0", "Epoch")
        for key, value in attrs.items():
            cdf[name].add_attribute(key, value)

    return bytes(pycdfpp.save(cdf))


@ddt
class GetTsTestCase(unittest.TestCase):
    def test_get_ts_sector_mask(self):
        # Non-record-varying FEEPS sector mask (1, 64) with an epoch: the same
        # mask at all times, one value per spin sector (used to fail)
        time = np.datetime64("2019-09-14T07:54:00", "ns") + np.arange(
            5
        ) * np.timedelta64(1, "s")
        mask = np.zeros((1, 64))
        mask[0, [3, 40]] = 3.0

        cdf = pycdfpp.CDF()
        cdf.add_variable(
            "epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000
        )
        cdf_name = "mms1_epd_feeps_srvy_l2_electron_top_sector_mask_sensorid_3"
        cdf.add_variable(cdf_name, values=mask, is_nrv=True)
        cdf[cdf_name].add_attribute("DEPEND_0", "epoch")

        tint = list(np.datetime_as_string(time[[1, 3]], "ns"))
        result = get_ts(bytes(pycdfpp.save(cdf)), cdf_name, tint)

        self.assertTupleEqual(result.dims, ("time", "sectornum"))
        np.testing.assert_array_equal(result.time.data, time[1:4])
        np.testing.assert_array_equal(result.sectornum.data, np.arange(64))
        np.testing.assert_array_equal(result.data, np.tile(mask, (3, 1)))

        # A writeable copy, not a view of the file values
        result.data[0, 0] = 1.0
        self.assertEqual(result.data[1, 0], 0.0)

        # A file without records gives an empty mask
        cdf = pycdfpp.CDF()
        cdf.add_variable(
            "epoch", values=time[:0], data_type=pycdfpp.DataType.CDF_TIME_TT2000
        )
        cdf.add_variable(cdf_name, values=mask, is_nrv=True)
        cdf[cdf_name].add_attribute("DEPEND_0", "epoch")
        result = get_ts(bytes(pycdfpp.save(cdf)), cdf_name)
        self.assertTupleEqual(result.shape, (0, 64))

    @data(
        ("mms1_fgm_b_gse_srvy_l2", ["x", "y", "z"]),
        ("mms1_mec_quat_eci_to_gse", ["qx", "qy", "qz", "qw"]),
        ("mms1_mec_bsc_gsm", [0, 1, 2]),
        ("mms1_mec_other", [0, 1, 2, 3]),
    )
    @unpack
    def test_get_ts_four_columns(self, cdf_name, comps):
        # Only the magnitude of (x, y, z, magnitude) variables is removed
        result = get_ts(_cdf_four_columns(), cdf_name)
        np.testing.assert_array_equal(
            result.data[0], [1.0, 2.0, 3.0, 4.0][: len(comps)]
        )
        self.assertListEqual(list(result[result.dims[1]].data), comps)

    @data(
        (1, 0.0, 30.0, 15),  # start times, single record (used to be NaT)
        (3, 0.0, 30.0, 15),  # start times
        (1, 15.0, 15.0, 0),  # centred times, single record
        (3, 15.0, 15.0, 0),  # centred times
        (3, 0.0, 20.0, 15),  # deltas do not match the 30 ms sampling period
    )
    @unpack
    def test_get_ts_fpi_epoch_shift(self, n_records, d_minus, d_plus, shift):
        # FPI epochs are moved to the centre of [t - delta_minus, t + delta_plus]
        # (in ms), or by half the sampling period if the accumulation does not
        # match it, as mms.variable2ts in irfu-matlab
        time = np.datetime64("2019-09-14T07:54:00", "ns")
        time = time + np.arange(n_records) * np.timedelta64(30, "ms")

        cdf = pycdfpp.CDF()
        cdf.add_variable(
            "Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000
        )
        cdf["Epoch"].add_attribute("DELTA_MINUS_VAR", "Epoch_minus_var")
        cdf["Epoch"].add_attribute("DELTA_PLUS_VAR", "Epoch_plus_var")
        for name, value in [("Epoch_minus_var", d_minus), ("Epoch_plus_var", d_plus)]:
            cdf.add_variable(name, values=np.array([value]), is_nrv=True)
            cdf[name].add_attribute("UNITS", "ms")
        cdf.add_variable("mms1_des_numberdensity_brst", values=np.ones(n_records))
        cdf["mms1_des_numberdensity_brst"].add_attribute("DEPEND_0", "Epoch")
        content = bytes(pycdfpp.save(cdf))

        if d_minus + d_plus == 30.0 or n_records == 1:
            result = get_ts(content, "mms1_des_numberdensity_brst")
        else:
            with self.assertLogs("pyrfu", level="WARNING"):
                result = get_ts(content, "mms1_des_numberdensity_brst")

        np.testing.assert_array_equal(
            result.time.data, time + np.timedelta64(shift, "ms")
        )


def _cdf_fpi_brst_spectr(n_records, t_start="2019-09-14T07:54:00", offset=0.0):
    # In-memory FPI burst electron omni spectrogram as in the L2 moments files:
    # record varying energy table (DEPEND_1), start times with deltas
    time = np.datetime64(t_start, "ns") + np.arange(n_records) * np.timedelta64(
        30, "ms"
    )
    energy = np.tile(10.0 * 1.3 ** np.arange(32) + offset, (n_records, 1))

    cdf = pycdfpp.CDF()
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    cdf["Epoch"].add_attribute("DELTA_MINUS_VAR", "Epoch_minus_var")
    cdf["Epoch"].add_attribute("DELTA_PLUS_VAR", "Epoch_plus_var")
    for name, value in [("Epoch_minus_var", 0.0), ("Epoch_plus_var", 30.0)]:
        cdf.add_variable(name, values=np.array([value]), is_nrv=True)
        cdf[name].add_attribute("UNITS", "ms")
    cdf.add_variable("mms1_des_energy_brst", values=energy.reshape(-1, 32))
    cdf["mms1_des_energy_brst"].add_attribute("LABLAXIS", "energy")
    cdf.add_variable(
        "mms1_des_energyspectr_omni_brst",
        values=np.arange(n_records * 32, dtype=float).reshape(-1, 32),
    )
    cdf["mms1_des_energyspectr_omni_brst"].add_attribute("DEPEND_0", "Epoch")
    cdf["mms1_des_energyspectr_omni_brst"].add_attribute(
        "DEPEND_1", "mms1_des_energy_brst"
    )
    return bytes(pycdfpp.save(cdf))


@ddt
class GetTsRecordVaryingTableTestCase(unittest.TestCase):
    cdf_name = "mms1_des_energyspectr_omni_brst"

    @data(0, 1, 4, 5)
    def test_get_ts_record_varying_table(self, n_records):
        # 0 records raised MissingDimensionsError (the empty 2-D table was used
        # as the coordinate), 4 records ValueError (compared with x, y, z, r)
        result = get_ts(_cdf_fpi_brst_spectr(n_records), self.cdf_name)
        self.assertTupleEqual(result.shape, (n_records, 32))
        self.assertTupleEqual(result.energy.shape, (32,))
        if n_records:
            np.testing.assert_allclose(result.energy.data, 10.0 * 1.3 ** np.arange(32))
        else:
            self.assertTrue(np.all(np.isnan(result.energy.data)))

    def test_get_ts_append_empty_file(self):
        # As get_data: a file without records, before or after a file with
        # data, gives the data and the energies of the latter
        empty = get_ts(_cdf_fpi_brst_spectr(0), self.cdf_name)
        full = get_ts(_cdf_fpi_brst_spectr(5, offset=1.0), self.cdf_name)
        for result in [pyrf.ts_append(empty, full), pyrf.ts_append(full, empty)]:
            xr.testing.assert_identical(result, full)


def _cdf_fpi_brst_dist(n_records, t_start, deltas=False, energy_delta=True):
    # In-memory FPI burst ion distribution laid out as in the L2 files: record
    # varying phi, energy, steptable parity and energy deltas, with alternating
    # energy tables (no energy0 and energy1 variables), and optionally the
    # accumulation of the start times (Epoch_minus_var 0, Epoch_plus_var 150 ms)
    time = np.datetime64(t_start, "ns") + np.arange(n_records) * np.timedelta64(
        150, "ms"
    )
    parity = np.arange(n_records) % 2
    energies = np.array(
        [[10.0, 20.0, 40.0, 80.0, 160.0], [12.0, 24.0, 48.0, 96.0, 192.0]]
    )

    cdf = pycdfpp.CDF()
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    if deltas:
        cdf["Epoch"].add_attribute("DELTA_MINUS_VAR", "Epoch_minus_var")
        cdf["Epoch"].add_attribute("DELTA_PLUS_VAR", "Epoch_plus_var")
        for name, value in [("Epoch_minus_var", 0.0), ("Epoch_plus_var", 150.0)]:
            cdf.add_variable(name, values=np.array([value]), is_nrv=True)
            cdf[name].add_attribute("UNITS", "ms")
    cdf.add_variable(
        "mms1_dis_phi_brst", values=np.tile(np.arange(4.0), (n_records, 1))
    )
    cdf.add_variable("mms1_dis_theta_brst", values=np.arange(3.0)[None], is_nrv=True)
    cdf.add_variable("mms1_dis_energy_brst", values=energies[parity].reshape(-1, 5))
    cdf.add_variable("mms1_dis_steptable_parity_brst", values=parity.astype(np.uint8))
    if energy_delta:
        cdf.add_variable("mms1_dis_energy_delta_brst", values=np.ones((n_records, 5)))
    cdf.add_variable(
        "mms1_dis_dist_brst",
        values=np.arange(n_records * 60, dtype=float).reshape(n_records, 4, 3, 5),
    )
    for i, dep in enumerate(["Epoch", "mms1_dis_phi_brst", "mms1_dis_theta_brst"]):
        cdf["mms1_dis_dist_brst"].add_attribute(f"DEPEND_{i:d}", dep)
    cdf["mms1_dis_dist_brst"].add_attribute("DEPEND_3", "mms1_dis_energy_brst")

    return bytes(pycdfpp.save(cdf))


@ddt
class GetDataEmptyFilesTestCase(unittest.TestCase):
    # Files with no record, or no record in tint, do not change the result
    tint = ["2019-09-14T07:54:00", "2019-09-14T07:55:00"]

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_data")

    def _get_data(self, var_str, contents):
        sources = ([f"file_{i:d}" for i in range(len(contents))], None, {})
        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with mock.patch.object(
                self.module, "_get_file_content_sources", side_effect=contents
            ):
                return mms.get_data(var_str, self.tint, 1, verbose=False, source="sdc")

    def test_get_dist_delta_time(self):
        # Times at the centres of the accumulations, whose half widths (s) are
        # delta_time_minus/plus, also when files are appended; without the
        # Epoch delta variables, no half widths
        content = _cdf_fpi_brst_dist(4, "2019-09-14T07:54:10", deltas=True)
        result = mms.get_dist(content, "mms1_dis_dist_brst")
        start = np.datetime64("2019-09-14T07:54:10", "ns")
        expected = start + np.arange(4) * np.timedelta64(150, "ms")
        np.testing.assert_array_equal(
            result.time.data, expected + np.timedelta64(75, "ms")
        )
        for key in ["delta_time_minus", "delta_time_plus"]:
            np.testing.assert_allclose(result.attrs[key], np.full(4, 0.075))

        later = _cdf_fpi_brst_dist(2, "2019-09-14T07:54:20", deltas=True)
        result = self._get_data("pdi_fpi_brst_l2", [content, later])
        np.testing.assert_allclose(result.attrs["delta_time_plus"], np.full(6, 0.075))

        result = mms.get_dist(
            _cdf_fpi_brst_dist(4, "2019-09-14T07:54:10"), "mms1_dis_dist_brst"
        )
        self.assertNotIn("delta_time_minus", result.attrs)

    def test_get_dist_no_energy_delta(self):
        # A file without energy widths: no delta_energy_* attributes (they
        # were None, on which reduce raised TypeError), also when appended
        content = _cdf_fpi_brst_dist(4, "2019-09-14T07:54:10", energy_delta=False)
        result = mms.get_dist(content, "mms1_dis_dist_brst")

        for key in ["delta_energy_minus", "delta_energy_plus"]:
            self.assertNotIn(key, result.attrs)

        result.data.attrs["UNITS"] = "s^3/cm^6"
        reduced = mms.reduce(result, np.eye(3), n_mc=2, verbose=False)
        self.assertEqual(reduced.shape[0], 4)

        later = _cdf_fpi_brst_dist(2, "2019-09-14T07:54:20", energy_delta=False)
        result = self._get_data("pdi_fpi_brst_l2", [content, later])
        self.assertEqual(result.sizes["time"], 6)
        self.assertNotIn("delta_energy_plus", result.attrs)

    @data((0, 3), (3, 0), (0, 0))
    @unpack
    def test_get_data_dist_empty_files(self, n_first, n_second):
        empty = _cdf_fpi_brst_dist(0, "2019-09-14T07:54:00")
        full = _cdf_fpi_brst_dist(3, "2019-09-14T07:54:10")
        outside = _cdf_fpi_brst_dist(2, "2019-09-14T08:30:00")
        contents = [full if n_first else empty, full if n_second else empty, outside]

        result = self._get_data("pdi_fpi_brst_l2", contents)

        n_records = 3 if n_first or n_second else 0
        self.assertEqual(result.sizes["time"], n_records)
        self.assertEqual(len(result.attrs["esteptable"]), n_records)
        if n_records:
            expected = self._get_data("pdi_fpi_brst_l2", [full])
            np.testing.assert_array_equal(result.data.data, expected.data.data)
            np.testing.assert_array_equal(
                result.energy0, [10.0, 20.0, 40.0, 80.0, 160.0]
            )
            np.testing.assert_array_equal(
                result.energy1, [12.0, 24.0, 48.0, 96.0, 192.0]
            )

    def test_get_data_ts_empty_files(self):
        contents = [
            _cdf_four_columns("2019-09-14T08:30:00"),
            _cdf_four_columns("2019-09-14T07:54:10"),
        ]
        result = self._get_data("b_gse_fgm_srvy_l2", contents)
        self.assertEqual(result.sizes["time"], 5)
        self.assertListEqual(list(result.shape), [5, 3])


def _cdf_eis(pref, species, p_num, t_start, n_records=4, energy=(50.0, 80.0, 120.0)):
    # In-memory EIS L2 file named as the reprocessed (v*.100) files, e.g.
    # mms1_epd_eis_brst_l2_extof_proton_P5_flux_t0, with the non-record-varying
    # energy tables (one record) of the 6 telescopes
    time = np.datetime64(t_start, "ns") + np.arange(n_records) * np.timedelta64(
        2500, "ms"
    )
    energy = np.array(energy)

    cdf = pycdfpp.CDF()
    cdf.add_attribute("Data_type", ["synthetic"])
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    for name, values in [
        ("spin", np.arange(n_records, dtype=np.uint16) // 2),
        ("sector", np.arange(n_records, dtype=np.int8)),
    ]:
        cdf.add_variable(f"{pref}_{name}", values=values)
        cdf[f"{pref}_{name}"].add_attribute("DEPEND_0", "Epoch")

    for i in range(6):
        cdf.add_variable(f"{pref}_look_t{i:d}", values=np.ones((n_records, 3)) * i)
        cdf[f"{pref}_look_t{i:d}"].add_attribute("DEPEND_0", "Epoch")

        for j, specie in enumerate(species):
            energy_name = f"{pref}_{specie}_t{i:d}_energy"
            for suffix, values in [("", energy), ("_dminus", 0.1 * energy)]:
                cdf.add_variable(
                    f"{energy_name}{suffix}", values=values[None], is_nrv=True
                )
            cdf.add_variable(
                f"{energy_name}_dplus", values=0.2 * energy[None], is_nrv=True
            )
            cdf[energy_name].add_attribute("LABLAXIS", "Energy")

            for unit in ["counts", "cps", "flux"]:
                flux = np.arange(n_records * len(energy), dtype=float)
                flux = flux.reshape(n_records, -1) + 100.0 * i + 1000.0 * j
                name = f"{pref}_{specie}_P{p_num:d}_{unit}_t{i:d}"
                cdf.add_variable(name, values=flux)
                cdf[name].add_attribute("DEPEND_0", "Epoch")
                cdf[name].add_attribute("DEPEND_1", energy_name)

    return bytes(pycdfpp.save(cdf))


@ddt
class GetEisAlltTestCase(unittest.TestCase):
    tint = ["2021-07-11T00:00:00", "2021-07-11T01:00:00"]

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_eis_allt")

    def _get_eis_allt(self, tar_var, contents, tint=None):
        sources = ([f"file_{i:d}" for i in range(len(contents))], None, {})
        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ) as list_files:
            with mock.patch.object(
                self.module, "_get_file_content_sources", side_effect=contents
            ) as get_content:
                result = mms.get_eis_allt(
                    tar_var, tint or self.tint, 1, verbose=False, source="sdc"
                )
        # one listing, and each file read (downloaded) once
        self.assertEqual(list_files.call_count, 1)
        self.assertEqual(get_content.call_count, len(contents))
        return result

    @data(
        # minor version 0 (all files since late 2019) and burst
        ("flux_extof_proton_brst_l2", "brst", "extof", ["proton"], 5, "proton"),
        # P# is not the major version of the file (v2.1.100 has P3)
        (
            "cps_phxtof_oxygen_srvy_l2",
            "srvy",
            "phxtof",
            ["proton", "oxygen"],
            3,
            "oxygen",
        ),
        (
            "counts_extof_alpha_srvy_l2",
            "srvy",
            "extof",
            ["proton", "helium"],
            4,
            "helium",
        ),
        (
            "Flux_electronenergy_electron_srvy_l2",
            "srvy",
            "electronenergy",
            ["electron"],
            4,
            "electron",
        ),
    )
    @unpack
    def test_get_eis_allt_names(self, tar_var, rate, dtype, species, p_num, specie):
        pref = f"mms1_epd_eis_{rate}_l2_{dtype}"
        content = _cdf_eis(pref, species, p_num, "2021-07-11T00:10:00")
        result = self._get_eis_allt(tar_var, [content])

        self.assertIsInstance(result, xr.Dataset)
        self.assertListEqual(
            sorted(result),
            sorted(
                [
                    "spin",
                    "sector",
                    *[f"{k}t{i:d}" for k in ["", "look_"] for i in range(6)],
                ]
            ),
        )
        self.assertEqual(result.attrs["species"], specie)
        self.assertEqual(result.attrs["Data_type"], "synthetic")
        np.testing.assert_array_equal(result.energy.data, [50.0, 80.0, 120.0])
        np.testing.assert_allclose(result.attrs["delta_energy_minus"], [5.0, 8.0, 12.0])
        np.testing.assert_allclose(
            result.attrs["delta_energy_plus"], [10.0, 16.0, 24.0]
        )

        j = species.index(specie)
        for i in range(6):
            self.assertTupleEqual(result[f"t{i:d}"].dims, ("time", "energy"))
            np.testing.assert_array_equal(
                result[f"t{i:d}"].data[:, 0], np.arange(0, 12, 3) + 100 * i + 1000 * j
            )
            np.testing.assert_array_equal(result[f"look_t{i:d}"].data, i)

    def test_get_eis_allt_files(self):
        # two files appended in time and clipped to tint, P# changes but not
        # the energy table
        pref = "mms1_epd_eis_srvy_l2_extof"
        contents = [
            _cdf_eis(pref, ["proton"], 4, "2021-07-11T00:59:50"),
            _cdf_eis(pref, ["proton"], 5, "2021-07-11T01:00:00"),
        ]
        tint = ["2021-07-11T00:59:52", "2021-07-11T01:00:04"]
        result = self._get_eis_allt("flux_extof_proton_srvy_l2", contents, tint)

        expected = np.datetime64("2021-07-11T00:59:52.500", "ns") + np.arange(
            5
        ) * np.timedelta64(2500, "ms")
        np.testing.assert_array_equal(result.time.data, expected)
        np.testing.assert_array_equal(result.t0.data[:, 0], [3, 6, 9, 0, 3])
        np.testing.assert_array_equal(result.sector.data, [1, 2, 3, 0, 1])

    def test_get_eis_allt_energy_change(self):
        pref = "mms1_epd_eis_srvy_l2_extof"
        contents = [
            _cdf_eis(pref, ["proton"], 4, "2021-07-11T00:10:00"),
            _cdf_eis(
                pref, ["proton"], 5, "2021-07-11T00:20:00", energy=(55.0, 80.0, 120.0)
            ),
        ]
        with self.assertRaisesRegex(ValueError, "split the time interval"):
            self._get_eis_allt("flux_extof_proton_srvy_l2", contents)

    def test_get_eis_allt_old_naming(self):
        # not reprocessed v3.0.0 files: mms1_epd_eis_extof_alpha_P2_flux_t0
        content = _cdf_eis(
            "mms1_epd_eis_extof", ["proton", "alpha"], 2, "2021-07-11T00:10:00"
        )
        with self.assertRaisesRegex(ValueError, r"reprocessed \(v\*\.100\)"):
            self._get_eis_allt("flux_extof_proton_srvy_l2", [content])

    def test_get_eis_allt_species(self):
        # no helium in PHxTOF
        content = _cdf_eis(
            "mms1_epd_eis_srvy_l2_phxtof",
            ["proton", "oxygen"],
            4,
            "2021-07-11T00:10:00",
        )
        with self.assertRaisesRegex(
            ValueError, "species in the file are oxygen, proton"
        ):
            self._get_eis_allt("flux_phxtof_alpha_srvy_l2", [content])

    def test_get_eis_allt_unit(self):
        with self.assertRaisesRegex(ValueError, "Invalid data unit"):
            mms.get_eis_allt("dpf_extof_proton_srvy_l2", self.tint, 1, source="sdc")


class _FakeS3Bucket:
    # Bucket listing the given keys with bucket.objects.filter(Prefix=...), and
    # recording the prefixes it was asked for.
    def __init__(self, keys, error=None):
        self.keys, self.error, self.prefixes = keys, error, []
        self.objects = self

    def filter(self, Prefix):  # pylint: disable=invalid-name
        self.prefixes.append(Prefix)

        if self.error is not None:
            raise self.error

        return [
            mock.Mock(key=key, size=len(key))
            for key in self.keys
            if key.startswith(Prefix)
        ]


@ddt
class ListFilesAwsTestCase(unittest.TestCase):
    BUCKET = "gov-nasa-hdrl-data1"
    HELIO = "spdf/cdaweb/data/mms"  # key prefix in the bucket
    FGM_BRST = {"inst": "fgm", "tmmode": "brst", "lev": "l2", "dtype": ""}
    FPI_FAST = {"inst": "fpi", "tmmode": "fast", "lev": "l2", "dtype": "des-moms"}

    def _list(self, keys, tint, var, bucket_prefix="", error=None):
        bucket = _FakeS3Bucket(keys, error)
        resource = mock.Mock()
        resource.Bucket.return_value = bucket

        with mock.patch.object(
            list_files_aws_module, "_s3_resource", return_value=resource
        ):
            out = mms.list_files_aws(tint, 1, var, bucket_prefix=bucket_prefix)

        return out, resource, bucket

    def test_list_files_aws_brst_month_directory(self):
        # HelioCloud: burst files directly in the month directory. The file
        # starting before the time interval covers its start; earlier ones and
        # those starting at or after its end are not needed.
        directory = f"{self.HELIO}/mms1/fgm/brst/l2/2019/09"
        times = ["051233", "075043", "075403", "080703", "081100", "090000"]
        keys = [f"{directory}/mms1_fgm_brst_l2_20190914{t}_v5.207.0.cdf" for t in times]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        bucket_prefix = f"{self.BUCKET}/{self.HELIO}"
        out, resource, _ = self._list(keys, tint, self.FGM_BRST, bucket_prefix)

        resource.Bucket.assert_called_once_with(self.BUCKET)
        self.assertListEqual([f["full_name"] for f in out], [keys[1], keys[2], keys[3]])
        self.assertEqual(out[0]["timetag"], "2019-09-14T07:50:43")
        self.assertEqual(out[0]["file_size"], len(keys[1]))

    def test_list_files_aws_brst_day_directory(self):
        # SDC layout: burst files in a day directory, looked at only if there is
        # no file in the month directory. Keys are built with "/" on all systems.
        prefix = "my-bucket/mms"
        directory = "mms/mms1/fgm/brst/l2/2019/09/14"
        keys = [f"{directory}/mms1_fgm_brst_l2_20190914075403_v5.207.0.cdf"]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        out, resource, bucket = self._list(keys, tint, self.FGM_BRST, prefix)

        resource.Bucket.assert_called_once_with("my-bucket")
        self.assertListEqual([f["full_name"] for f in out], keys)
        self.assertIn(
            "mms/mms1/fgm/brst/l2/2019/09/14/mms1_fgm_brst_l2_20190914", bucket.prefixes
        )
        self.assertTrue(all("\\" not in p for p in bucket.prefixes))

    def test_list_files_aws_latest_version_and_previous_day(self):
        # Only the latest version of each file; a file of the previous day covers
        # the start of the time interval.
        directory = f"{self.HELIO}/mms1/fpi/fast/l2/des-moms/2019/09"
        stem = "mms1_fpi_fast_l2_des-moms"
        keys = [
            f"{directory}/{stem}_20190913220000_v3.4.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.3.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.4.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.10.0.cdf",
            f"{directory}/{stem}_20190914020000_v3.4.0.cdf",
            f"{directory}/mms1_fpi_fast_l2_dis-moms_20190914000000_v3.4.0.cdf",
        ]
        tint = ["2019-09-13T23:30:00", "2019-09-14T01:00:00"]

        out, _, _ = self._list(keys, tint, self.FPI_FAST, f"{self.BUCKET}/{self.HELIO}")

        self.assertListEqual([f["full_name"] for f in out], [keys[0], keys[3]])

    def test_list_files_aws_listing_error(self):
        error = ClientError({"Error": {"Code": "NoSuchBucket"}}, "ListObjectsV2")
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        with self.assertRaises(FileNotFoundError):
            self._list([], tint, self.FGM_BRST, "missing/mms", error=error)

    @data(
        (["2019-09-14T07:54:00", "2019-09-14T08:11:00"], FGM_BRST),
    )
    @unpack
    def test_list_files_aws_input(self, tint, var):
        with self.assertRaises(TypeError):
            mms.list_files_aws(tuple(tint), 1, var)

        with self.assertRaises(TypeError):
            mms.list_files_aws(list(pyrf.iso86012datetime64(np.array(tint))), 1, var)

    @data(
        ("s3://my-bucket/data/mms/", ("my-bucket", "data/mms")),
        ("my-bucket", ("my-bucket", "")),
        ("", ("gov-nasa-hdrl-data1", "spdf/cdaweb/data/mms")),  # "aws" not set
    )
    @unpack
    def test_bucket_and_prefix(self, config_aws, expected):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")

            with open(config_path, "w", encoding="utf-8") as fs:
                json.dump({"aws": config_aws}, fs)

            with mock.patch.object(list_files_aws_module, "MMS_CFG_PATH", config_path):
                result = list_files_aws_module._bucket_and_prefix()

        self.assertTupleEqual(result, expected)
        self.assertTupleEqual(
            list_files_aws_module._bucket_and_prefix("s3://other/x"), ("other", "x")
        )

    @data(True, False)
    def test_s3_resource_credentials(self, has_credentials):
        # Anonymous requests if there are no AWS credentials (public buckets).
        # No region lookup on the EC2 instance metadata service (network)
        credentials = mock.Mock() if has_credentials else None
        environ = {
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        with (
            mock.patch(
                "boto3.session.Session.get_credentials", return_value=credentials
            ),
            mock.patch.dict(os.environ, environ),
        ):
            resource = list_files_aws_module._s3_resource()

        signature = resource.meta.client.meta.config.signature_version
        self.assertEqual(signature is UNSIGNED, not has_credentials)

    @pytest.mark.network
    @unittest.skipUnless(
        os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
    )
    def test_list_files_aws_heliocloud(self):
        # Real listing of the public HelioCloud bucket
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        out = mms.list_files_aws(tint, 1, self.FGM_BRST, bucket_prefix="")
        times = [f["full_name"].rsplit("_", 2)[-2] for f in out]
        self.assertListEqual(
            times,
            [
                "20190914075043",
                "20190914075403",
                "20190914075823",
                "20190914080243",
                "20190914080703",
            ],
        )


@ddt
class ReduceTestCase(unittest.TestCase):
    @data("s^3/cm^6", "s^3/m^6", "s^3/km^6")
    def test_reduce_units(self, value):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species="ions")
        vdf.data.attrs["UNITS"] = value
        result = mms.reduce(vdf, np.eye(3), "1d", "pol")
        self.assertIsInstance(result, xr.DataArray)

    @data(
        (False, "ions", np.eye(3), "1d", "pol"),
        (False, "electrons", np.eye(3), "1d", "pol"),
        (True, "ions", np.eye(3), "1d", "pol"),
        (True, "electrons", np.eye(3), "1d", "pol"),
        (False, "electrons", generate_ts(64.0, 42, tensor_order=2), "1d", "pol"),
        (False, "ions", np.eye(3), "1d", "pol"),
        # (False, "ions", np.eye(3), "2d", "pol"), mc_pol_2d NotImplementedError
    )
    @unpack
    def test_reduce_output(self, energy01, species, xyz, dim, base):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01, species)
        result = mms.reduce(vdf, xyz, dim, base)
        self.assertIsInstance(result, xr.DataArray)

    @data(
        ("2d", "cart", {}),
        ("1d", "pol", {"vg": np.linspace(-1, 1, 42)}),
        ("1d", "pol", {"lower_e_lim": generate_ts(64.0, 42)}),
        ("1d", "pol", {"vg_edges": np.linspace(-1.01, 1.01, 102)}),
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

    def test_reduce_input_handling(self):
        # lower_e_lim as an int (TypeError), base and dim in capitals (base
        # was compared case-sensitively), no progress bar with verbose=False
        vdf = generate_vdf(64.0, 3, [32, 32, 16], energy01=False, species="ions")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = mms.reduce(vdf, np.eye(3), lower_e_lim=0, n_mc=5, verbose=False)
            result_2d = mms.reduce(vdf, np.eye(3), "2D", "CART", n_mc=5, verbose=False)

        self.assertEqual(result.shape, (3, 64))
        self.assertEqual(result_2d.shape, (3, 100, 100))
        self.assertEqual(stderr.getvalue(), "")

        with self.assertRaisesRegex(ValueError, "base must be"):
            mms.reduce(vdf, np.eye(3), "1d", "sph")

    def test_reduce_no_channel(self):
        # V_sc above every channel at the first time: NaN there (IndexError),
        # the grid from the next time; at all times: ValueError
        vdf = generate_vdf(64.0, 3, [32, 32, 16], energy01=False, species="ions")
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.array([1e3, 0.0, 0.0]))
        result = mms.reduce(vdf, np.eye(3), sc_pot=sc_pot, n_mc=5, verbose=False)

        self.assertTrue(np.all(np.isnan(result.data[0])))
        self.assertTrue(np.all(np.isfinite(result.data[1:])))
        self.assertGreater(np.max(np.abs(result.vx.data)), 0.0)

        sc_pot.data[:] = 1e3

        with self.assertRaisesRegex(ValueError, "no energy channel"):
            mms.reduce(vdf, np.eye(3), sc_pot=sc_pot, n_mc=5, verbose=False)

    def test_reduce_default_phi_grid(self):
        # Default azimuthal grid has one point per instrument azimuth. phi is
        # (time, phi), so len(phi) gave one point per time step (5 here).
        vdf = generate_vdf(64.0, 5, [32, 32, 16], energy01=True, species="ions")
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        reduce_module = importlib.import_module("pyrfu.mms.reduce")
        with mock.patch.object(
            reduce_module, "int_sph_dist", wraps=reduce_module.int_sph_dist
        ) as isd:
            mms.reduce(vdf, np.eye(3), "2d", "cart", n_mc=1)

        phi_grid = isd.call_args.args[5]
        np.testing.assert_allclose(phi_grid, np.deg2rad(5.625 + 11.25 * np.arange(32)))

    @data([], ["delta_energy_minus"], ["delta_energy_minus", "delta_energy_plus"])
    def test_reduce_drifting_maxwellian(self, removed):
        # 1 keV proton Maxwellian (n = 1 cm^-3) drifting at (-300, 200, 100) km/s,
        # reduced along x. Uses the channel widths from delta_energy_* or, if
        # missing, the default speed bin edges. The reduced distribution used to
        # be shifted to lower speeds (n 6 %, V 7 % and T 12 % too low), and
        # failed without delta_energy_plus.
        v_d = [-300.0, 200.0, 100.0]
        vdf, sc_pot = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_d)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        for key in removed:
            del vdf.attrs[key]

        v_grid = np.linspace(-2500.0, 2500.0, 251) * 1e3
        result = mms.reduce(vdf, np.eye(3), "1d", "pol", vg=v_grid, n_mc=50)

        v_x, f_x = result.vx.data * 1e3, result.data[0]
        d_v = np.median(np.diff(v_x))
        n = np.sum(f_x) * d_v
        v_bulk = np.sum(v_x * f_x) * d_v / n
        t_x = constants.proton_mass * np.sum((v_x - v_bulk) ** 2 * f_x) * d_v / n
        t_x /= constants.elementary_charge

        self.assertAlmostEqual(n / 1e6, 1.0, delta=0.01)
        self.assertAlmostEqual(v_bulk / 1e3, v_d[0], delta=3.0)
        self.assertAlmostEqual(t_x, 1000.0, delta=40.0)

    def test_reduce_angle_widths(self):
        # FEEPS skymap: 4 polar bands of different widths (delta_theta_*). The
        # Monte-Carlo particles used to be drawn over the median spacing around
        # the eye cones, beyond the poles, and the density was 4 % too large
        sky = mms.feeps_skymap(_feeps_skymap_case("brst", 128))
        energy = sky.energy.data
        half = np.diff(energy, axis=1, append=1.1 * energy[:, -1:]) / 2
        sky.attrs["delta_energy_minus"], sky.attrs["delta_energy_plus"] = half, half

        v_grid = np.linspace(-2.9e8, 2.9e8, 291)
        result = mms.reduce(sky, np.eye(3), "1d", "pol", vg=v_grid, n_mc=200)
        n_reduced = np.sum(result.data, axis=1) * (v_grid[1] - v_grid[0])

        # Direct sum with the relativistic speeds and the exact solid angles
        def speed(e_ev):
            gamma = 1 + constants.e * e_ev / (constants.m_e * constants.c**2)
            return constants.c * np.sqrt(1 - 1 / gamma**2)

        d_v = speed(energy + half) - speed(energy - half)
        theta_1 = np.deg2rad(sky.theta.data - sky.attrs["delta_theta_minus"])
        theta_2 = np.deg2rad(sky.theta.data + sky.attrs["delta_theta_plus"])
        d_phi = np.deg2rad(sky.attrs["delta_phi_minus"] + sky.attrs["delta_phi_plus"])
        d_omega = d_phi[:, :, None] * (np.cos(theta_1) - np.cos(theta_2))
        f_si = sky.data.data * 1e12
        n_direct = np.sum(
            f_si * (speed(energy) ** 2 * d_v)[..., None, None] * d_omega[:, None],
            axis=(1, 2, 3),
        )
        np.testing.assert_allclose(n_reduced, n_direct, rtol=0.01)

    def test_reduce_cart_without_energy_widths(self):
        # Without the energy widths, the default Cartesian grid extends to the
        # upper edge of the inferred speed bins: the top channel speed
        # extrapolated by half a channel in log space (relativistic speeds)
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        del vdf.attrs["delta_energy_minus"], vdf.attrs["delta_energy_plus"]

        result = mms.reduce(vdf, np.eye(3), "2d", "cart", n_mc=1)

        gamma = 1 + constants.electron_volt * vdf.energy.data[0, -2:] / (
            constants.proton_mass * constants.speed_of_light**2
        )
        speed = constants.speed_of_light * np.sqrt(1 - 1 / gamma**2)
        v_max = speed[1] * np.sqrt(speed[1] / speed[0])
        np.testing.assert_allclose(result.vx.data[[0, -1]] * 1e3, [-v_max, v_max])
        np.testing.assert_allclose(result.vy.data[[0, -1]] * 1e3, [-v_max, v_max])


@ddt
class RemoveEdistBackgroundTestCase(unittest.TestCase):
    MODEL = "mms_fpi_brst_l2_des-bgdist_v1.1.0_p0-2.cdf"

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.remove_edist_background")

    def test_load_bgdist_model_local(self):
        with tempfile.TemporaryDirectory() as data_path:
            os.makedirs(os.path.join(data_path, "models", "fpi"))
            file_path = os.path.join(data_path, "models", "fpi", self.MODEL)
            open(file_path, "wb").close()

            with mock.patch.object(self.module.pycdfpp, "load") as load:
                self.module._load_bgdist_model(self.MODEL, "local", data_path)

            load.assert_called_once_with(file_path)

            # Missing model file: clear error with the SDC URL
            with self.assertRaises(FileNotFoundError) as context:
                self.module._load_bgdist_model("missing.cdf", "local", data_path)

            url = "https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/missing.cdf"
            self.assertIn(url, str(context.exception))

    def _load_from_sdc(self, source, s3_resource=None):
        # Load the model with the SDC (and S3) mocked; returns the SDC session
        session = mock.MagicMock()
        session.get.return_value.content = b"sdc cdf bytes"
        login = (session, {"User-Agent": "pyrfu"}, self.module.LASP_PUBL)
        bucket = ("gov-nasa-hdrl-data1", "spdf/cdaweb/data/mms")

        with mock.patch.object(self.module, "_login_lasp", return_value=login):
            with mock.patch.object(
                self.module, "_s3_resource", return_value=s3_resource
            ):
                with mock.patch.object(
                    self.module, "_bucket_and_prefix", return_value=bucket
                ):
                    with mock.patch.object(self.module.pycdfpp, "load") as load:
                        self.module._load_bgdist_model(self.MODEL, source, "")

        return session, load

    def test_load_bgdist_model_sdc(self):
        # Read from the SDC into memory
        session, load = self._load_from_sdc("sdc")

        url = f"https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/{self.MODEL}"
        self.assertEqual(session.get.call_args.args[0], url)
        load.assert_called_once_with(b"sdc cdf bytes")
        session.close.assert_not_called()  # shared, cached session

    def test_load_bgdist_model_aws(self):
        # Read from models/fpi/ in the MMS bucket, without the SDC
        s3_resource = mock.MagicMock()
        s3_object = s3_resource.Object.return_value
        s3_object.get.return_value = {"Body": mock.Mock(read=lambda: b"s3 cdf bytes")}

        session, load = self._load_from_sdc("aws", s3_resource)

        s3_resource.Object.assert_called_once_with(
            "gov-nasa-hdrl-data1", f"spdf/cdaweb/data/mms/models/fpi/{self.MODEL}"
        )
        load.assert_called_once_with(b"s3 cdf bytes")
        session.get.assert_not_called()

    def test_load_bgdist_model_aws_fallback(self):
        # Not in the bucket: read from the SDC
        s3_resource = mock.MagicMock()
        s3_resource.Object.return_value.get.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        with self.assertLogs("pyrfu", level="WARNING"):
            session, load = self._load_from_sdc("aws", s3_resource)

        load.assert_called_once_with(b"sdc cdf bytes")
        session.close.assert_not_called()  # shared, cached session

    def test_models_url(self):
        self.assertEqual(
            self.module._models_url(
                "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"
            ),
            "https://lasp.colorado.edu/mms/sdc/sitl/data/models/fpi/",
        )

    def _inputs(self, n_t=4):
        # Burst DES skymap with alternating energy tables (parity 0, 1, 0, ...),
        # at spin phases of model records 0, 1, 2, ... Model record k is k for
        # parity 0 and 1000 + k for parity 1, so the subtracted background
        # shows which record and table were used.
        vdf = generate_vdf(64.0, n_t, [32, 32, 16], energy01=True, species="electrons")
        vdf.data.data[...] = 1e6
        vdf.data.attrs["CATDESC"] = "MMS1 DES burst distribution"
        vdf.data.attrs["FIELDNAM"] = "mms1_des_dist_brst"

        n_e = pyrf.ts_scalar(vdf.time.data, np.ones(n_t))
        n_e.attrs["GLOBAL"] = {
            "Photoelectron_model_scaling_factor": "0.5",
            "Photoelectron_model_filenames": self.MODEL,
        }
        startdelphi = pyrf.ts_scalar(vdf.time.data, 16 * np.arange(n_t) + 3)
        records = np.arange(360.0)[:, None, None, None] * np.ones((1, 32, 16, 32))
        model = {
            "mms_des_bgdist_p0_brst": mock.Mock(values=records),
            "mms_des_bgdist_p1_brst": mock.Mock(values=records + 1000.0),
            "mms_des_startdelphi_counts_brst": mock.Mock(
                values=16 * np.arange(360) + 8
            ),
        }
        return vdf, n_e, startdelphi, model

    def _run(self, vdf, n_e, startdelphi, model, **kwargs):
        with (
            mock.patch.object(self.module, "get_data", return_value=n_e) as get_data,
            mock.patch.object(
                self.module, "db_get_ts", return_value=startdelphi
            ) as db_get_ts,
            mock.patch.object(
                self.module, "_load_bgdist_model", return_value=model
            ) as load_model,
        ):
            result = mms.remove_edist_background(vdf, **kwargs)

        return result, (get_data, db_get_ts, load_model)

    def test_remove_edist_background_source(self):
        # source is passed to get_data, db_get_ts and the model loader
        vdf, n_e, startdelphi, model = self._inputs()
        (_, _, scale), (get_data, db_get_ts, load_model) = self._run(
            vdf, n_e, startdelphi, model, source="SDC"
        )

        self.assertEqual(get_data.call_args.kwargs["source"], "sdc")
        self.assertEqual(db_get_ts.call_args.kwargs["source"], "sdc")
        self.assertEqual(load_model.call_args.args[:2], (self.MODEL, "sdc"))
        self.assertEqual(scale, 0.5)

        with self.assertRaises(ValueError):
            mms.remove_edist_background(vdf, source="bazinga")

    def test_remove_edist_background_model(self):
        # Parity 1 samples used the parity 0 model; each sample uses the model
        # record of its spin phase, scaled by the photoelectron factor
        vdf, n_e, startdelphi, model = self._inputs()
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model)

        expected = 0.5 * np.array([0.0, 1001.0, 2.0, 1003.0])
        np.testing.assert_allclose(vdf_bkg.data.data[:, 0, 0, 0], expected)
        np.testing.assert_allclose(vdf_new.data.data[:, 0, 0, 0], 1e6 - expected)

    def test_remove_edist_background_spin_phase_time(self):
        # startdelphi_count is read separately: an extra sample before the
        # distributions used to shift every sample to the next spin phase
        vdf, n_e, startdelphi, model = self._inputs()
        dt = vdf.time.data[1] - vdf.time.data[0]
        extra = pyrf.ts_scalar(
            np.hstack([vdf.time.data[0] - dt, vdf.time.data]),
            np.hstack([16 * 100 + 3, startdelphi.data]),
        )
        (_, vdf_bkg, _), _ = self._run(vdf, n_e, extra, model)

        expected = 0.5 * np.array([0.0, 1001.0, 2.0, 1003.0])
        np.testing.assert_allclose(vdf_bkg.data.data[:, 0, 0, 0], expected)

    def test_remove_edist_background_n_art(self):
        # n_art = 0 removes no photoelectrons (it used to mean "default")
        vdf, n_e, startdelphi, model = self._inputs()
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model, n_art=0.0)

        np.testing.assert_allclose(vdf_bkg.data.data, 0.0)
        np.testing.assert_allclose(vdf_new.data.data, vdf.data.data)

        # The attributes of the outputs and of the input are independent
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model)
        vdf_new.attrs["test"] = 1
        self.assertNotIn("test", vdf.attrs)
        self.assertNotIn("test", vdf_bkg.attrs)


class RemoveIdistBackgroundTestCase(unittest.TestCase):
    def test_remove_idist_background(self):
        # The background (omni differential energy flux, 1/(cm^2 s sr)) converted
        # to phase-space density is removed (negative values set to 0), and the
        # caller's VDF is unchanged (it was a shallow copy)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], species="ions")
        vdf.data.data[...] = 1e-23
        vdf_in = vdf.data.data.copy()
        def_bg = pyrf.ts_scalar(vdf.time.data, np.full(10, 1e3))

        result = mms.remove_idist_background(vdf, def_bg)

        with np.errstate(divide="ignore"):  # the first energy channel is 0
            coeff = constants.proton_mass / (
                constants.elementary_charge * vdf.energy.data.astype(np.float64)
            )
        psd_bg = 1e3 * 1e4 / 2 * coeff**2 / 1e12  # (time, energy)
        expected = np.clip(1e-23 - psd_bg[:, :, None, None], 0.0, None)
        expected = np.broadcast_to(expected, vdf_in.shape)

        self.assertTrue(np.any(expected > 0) and np.any(expected == 0))
        np.testing.assert_allclose(result.data.data, expected, rtol=1e-5, atol=1e-30)
        np.testing.assert_array_equal(vdf.data.data, vdf_in)


class RemoveImomsBackgroundTestCase(unittest.TestCase):
    @staticmethod
    def _measured(n_t=5):
        # Moments measured with a background at rest (n_bg = 0.5 cm^-3,
        # p_bg = 0.01 nPa) added to a known plasma (n = 0.5 cm^-3, V, P)
        n_true, v_true = 0.5, np.array([-800.0, 50.0, 20.0])
        p_true = np.array([[0.2, 0.01, 0.02], [0.01, 0.15, 0.03], [0.02, 0.03, 0.1]])
        n_bg, p_bg = 0.5, 0.01

        n_meas = n_true + n_bg
        v_meas = n_true * v_true / n_meas
        fact = constants.proton_mass * 1e21  # m n v v in nPa (cm^-3, km/s)
        p_meas = p_true + fact * n_true * np.outer(v_true, v_true)
        p_meas += p_bg * np.eye(3) - fact * n_meas * np.outer(v_meas, v_meas)

        time = generate_timeline(1.0, n_t)
        measured = (
            pyrf.ts_scalar(time, np.full(n_t, n_meas)),
            pyrf.ts_vec_xyz(time, np.tile(v_meas, (n_t, 1))),
            pyrf.ts_tensor_xyz(
                time, np.tile(p_meas, (n_t, 1, 1)), attrs={"UNITS": "nPa"}
            ),
        )
        return time, measured, (n_true, v_true, p_true), (n_bg, p_bg)

    def test_remove_imoms_background_values(self):
        # The dynamic pressure terms lacked a 1e21 unit factor, so only p_bg
        # was removed from the pressure tensor
        time, measured, expected, (n_bg, p_bg) = self._measured()
        n_i, v_i, p_i = mms.remove_imoms_background(
            *measured,
            pyrf.ts_scalar(time, np.full(len(time), n_bg)),
            pyrf.ts_scalar(time, np.full(len(time), p_bg)),
        )

        np.testing.assert_allclose(n_i.data, expected[0])
        np.testing.assert_allclose(v_i.data, np.tile(expected[1], (len(time), 1)))
        np.testing.assert_allclose(
            p_i.data, np.tile(expected[2], (len(time), 1, 1)), atol=1e-12
        )
        self.assertEqual(p_i.attrs["UNITS"], "nPa")

    def test_remove_imoms_background_time_alignment(self):
        # Background moments on another time line are resampled first (they
        # used to be subtracted sample by sample, or raise a shape error)
        time, measured, expected, (n_bg, p_bg) = self._measured(10)
        time_bg = time[0] + (time[-1] - time[0]) * np.linspace(0, 1, 4)
        _, _, p_i = mms.remove_imoms_background(
            *measured,
            pyrf.ts_scalar(time_bg, np.full(4, n_bg)),
            pyrf.ts_scalar(time_bg, np.full(4, p_bg)),
        )
        np.testing.assert_allclose(
            p_i.data, np.tile(expected[2], (10, 1, 1)), atol=1e-12
        )


@ddt
class RotateTensorTestCase(unittest.TestCase):
    @staticmethod
    def _tensor(time):
        p_xyz = np.array([[2.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.5]])
        return pyrf.ts_tensor_xyz(time, np.tile(p_xyz, (len(time), 1, 1)))

    @staticmethod
    def _invariants(p_fac):
        # Rotation-invariant quantities about the parallel direction
        p_fac = p_fac.data[0]
        return [
            p_fac[0, 0],
            p_fac[1, 1] + p_fac[2, 2],
            p_fac[0, 1] ** 2 + p_fac[0, 2] ** 2,
        ]

    def test_rotate_tensor_fac_b_along_x(self):
        # B exactly along x gave NaN (B x x = 0); now y is the reference
        time = generate_timeline(1.0, 3)
        p_xyz = self._tensor(time)
        for perp in ["pp", ""]:
            result = mms.rotate_tensor(
                p_xyz, "fac", pyrf.ts_vec_xyz(time, np.tile([10.0, 0, 0], (3, 1))), perp
            )
            near_x = mms.rotate_tensor(
                p_xyz,
                "fac",
                pyrf.ts_vec_xyz(time, np.tile([10.0, 1e-9, 0], (3, 1))),
                perp,
            )
            self.assertTrue(np.all(np.isfinite(result.data)))
            self.assertAlmostEqual(float(result.data[0, 0, 0]), 2.0)
            np.testing.assert_allclose(
                self._invariants(result), self._invariants(near_x), rtol=1e-7
            )

    def test_rotate_tensor_rot_along_y(self):
        # A single vector exactly along y gave NaN (reference axis y)
        time = generate_timeline(1.0, 3)
        result = mms.rotate_tensor(self._tensor(time), "rot", np.array([0.0, 5.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(result.data)))
        self.assertAlmostEqual(float(result.data[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(np.trace(result.data[0])), 4.5)

    def test_rotate_tensor_pp_no_warning(self):
        # P23 = 0 (diagonal tensor) gave divide-by-zero warnings in "pp"
        time = generate_timeline(1.0, 3)
        p_xyz = pyrf.ts_tensor_xyz(time, np.tile(np.diag([2.0, 1.0, 3.0]), (3, 1, 1)))
        b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, 10.0], (3, 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = mms.rotate_tensor(p_xyz, "fac", b_xyz, "pp")
        np.testing.assert_allclose(np.diag(result.data[0]), [3.0, 1.5, 1.5])

    def test_rotate_tensor_fac_time_varying(self):
        # A different tensor and B at each time (rotated at all times at once;
        # it was a loop): P_par = b.P.b, trace kept, equal perpendicular
        # diagonal terms for "pp", no perpendicular off-diagonal term for "qq"
        time = generate_timeline(1.0, 50)
        rng = np.random.default_rng(3)
        a_mat = rng.standard_normal((50, 3, 3))
        p_xyz = pyrf.ts_tensor_xyz(time, a_mat @ np.transpose(a_mat, [0, 2, 1]))
        b_xyz = pyrf.ts_vec_xyz(time, rng.standard_normal((50, 3)))
        b_hat = b_xyz.data / np.linalg.norm(b_xyz.data, axis=1, keepdims=True)
        p_par = np.einsum("ti,tij,tj->t", b_hat, p_xyz.data, b_hat)

        for perp in ["pp", "qq"]:
            result = mms.rotate_tensor(p_xyz, "fac", b_xyz, perp).data
            np.testing.assert_allclose(result[:, 0, 0], p_par, rtol=1e-12)
            np.testing.assert_allclose(
                np.trace(result, axis1=1, axis2=2),
                np.trace(p_xyz.data, axis1=1, axis2=2),
                rtol=1e-12,
            )

            if perp == "pp":
                np.testing.assert_allclose(result[:, 1, 1], result[:, 2, 2])
            else:
                np.testing.assert_allclose(result[:, 1, 2], 0.0, atol=1e-12)

    @data(
        (
            generate_data(100, tensor_order=1),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            42,
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            "bazinga!!",
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_data(100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "rot",
            generate_ts(64.0, 100, tensor_order=2),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "gse",
            generate_ts(64.0, 100, tensor_order=2),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            42,
        ),
    )
    @unpack
    def test_rotate_tensor_input_type(self, inp, rot_flag, vec, perp):
        with self.assertRaises(TypeError):
            mms.rotate_tensor(inp, rot_flag, vec, perp)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=2),
            "bazinga!!",
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "rot",
            generate_data(100, tensor_order=2),
            "",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            "bazinga!!",
        ),
    )
    @unpack
    def test_rotate_tensor_input_method(self, inp, flag, vec, perp):
        with self.assertRaises(NotImplementedError):
            mms.rotate_tensor(inp, flag, vec, perp)

    @data(
        ("fac", generate_ts(64.0, 100, tensor_order=1), "pp"),
        ("fac", generate_ts(64.0, 100, tensor_order=1), "qq"),
        ("rot", np.random.random(3), "pp"),
        ("rot", np.random.random((3, 3)), "pp"),
        ("gse", generate_defatt(64.0, 100), "pp"),
    )
    @unpack
    def test_rotate_tensor_output(self, rot_flag, vec, perp):
        result = mms.rotate_tensor(
            generate_ts(64.0, 100, tensor_order=2), rot_flag, vec, perp
        )
        self.assertIsInstance(result, xr.DataArray)

    @data("gse", "gsm")
    def test_rotate_tensor_dsl_consistent_with_dsl2gse(self, frame):
        # T = a I + v v^T in DSL must rotate to a I + v' v'^T with v' = dsl2gse(v).
        # The spin-axis angles were converted from degrees to radians twice.
        n_t = 50
        time = generate_timeline(1.0, n_t)
        defatt = xr.Dataset(
            {
                "z_ra": pyrf.ts_scalar(time, np.linspace(268.0, 272.0, n_t)),
                "z_dec": pyrf.ts_scalar(time, np.linspace(65.0, 67.0, n_t)),
            }
        )
        v_dsl = pyrf.ts_vec_xyz(time, np.random.default_rng(0).normal(size=(n_t, 3)))
        t_dsl = pyrf.ts_tensor_xyz(
            time, 2.0 * np.eye(3) + np.einsum("ti,tj->tij", v_dsl.data, v_dsl.data)
        )

        v_new = mms.dsl2gse(v_dsl, defatt)
        if frame == "gsm":
            v_new = pyrf.cotrans(v_new, "gse>gsm")

        expected = 2.0 * np.eye(3) + np.einsum("ti,tj->tij", v_new.data, v_new.data)
        result = mms.rotate_tensor(t_dsl, frame, defatt)
        np.testing.assert_allclose(result.data, expected, atol=1e-10)

    @staticmethod
    def _rot_tensor():
        # P = [[1, 0, 0], [0, 2, 0.3], [0, 0.3, 3]] at 5 times
        p_xyz = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.3], [0.0, 0.3, 3.0]]
        return pyrf.ts_tensor_xyz(generate_timeline(1.0, 5), np.tile(p_xyz, (5, 1, 1)))

    def test_rotate_tensor_rot_no_perp_rotation(self):
        # "rot" rotates to the given axes only, as irfu-matlab (the default
        # "pp" also rotated the 2-3 components, even for the identity): the
        # identity leaves P unchanged, (L, M, N) = (y, z, x) permutes it
        inp = self._rot_tensor()
        np.testing.assert_allclose(
            mms.rotate_tensor(inp, "rot", np.eye(3)).data, inp.data, atol=1e-15
        )

        lmn = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        expected = [[2.0, 0.3, 0.0], [0.3, 3.0, 0.0], [0.0, 0.0, 1.0]]
        np.testing.assert_allclose(
            mms.rotate_tensor(inp, "rot", lmn).data[0], expected, atol=1e-15
        )

    def test_rotate_tensor_rot_explicit_perp(self):
        # "pp" and "qq" are for field-aligned coordinates: ignored with a
        # warning elsewhere
        inp = self._rot_tensor()

        with self.assertLogs("pyrfu.mms.rotate_tensor", "WARNING") as logs:
            result = mms.rotate_tensor(inp, "rot", np.eye(3), "pp")

        self.assertIn("field-aligned", "".join(logs.output))
        np.testing.assert_allclose(result.data, inp.data, atol=1e-15)

    def test_rotate_tensor_rot_right_handed_silent(self):
        # A right-handed orthonormal system gives no warning (the check
        # compared the new y axis with the x column of the input)
        with self.assertNoLogs("pyrfu.mms.rotate_tensor", "WARNING"):
            mms.rotate_tensor(self._rot_tensor(), "rot", np.eye(3))

    def test_rotate_tensor_fac_default_pp(self):
        # "fac" keeps "pp" by default: equal perpendicular components
        inp = self._rot_tensor()
        b_xyz = pyrf.ts_vec_xyz(inp.time.data, np.tile([0.2, 0.3, 1.0], (5, 1)))
        result = mms.rotate_tensor(inp, "fac", b_xyz)

        self.assertAlmostEqual(result.data[0, 1, 1], result.data[0, 2, 2])
        np.testing.assert_allclose(
            result.data, mms.rotate_tensor(inp, "fac", b_xyz, "pp").data
        )


@ddt
class SpectrToDatasetTestCase(unittest.TestCase):
    @data(generate_spectr(64.0, 100, 10))
    def test_spectr_to_dataset_output(self, spectr):
        result = mms.spectr_to_dataset(spectr)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class Scpot2NeTestCase(unittest.TestCase):
    @data(None, generate_ts(64.0, 100, tensor_order=0))
    def test_scpot2ne_output(self, i_aspoc):
        result = mms.scpot2ne(
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=2),
            i_aspoc,
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

    @staticmethod
    def _case(dtype=np.float64):
        # Thermal current = two photoelectron populations (60 muA at 5 eV,
        # 8 muA at 20 eV) for potentials of 3 to 40 V and T_e = 1 keV
        i_ph = np.array([60.0, 5.0, 8.0, 20.0])
        time = generate_timeline(1 / 4.5, 400)
        v_sc = np.linspace(3.0, 40.0, 400)
        t_e = np.full(400, 1000.0)
        q_e, m_e = constants.elementary_charge, constants.electron_mass
        current = i_ph[0] * np.exp(-v_sc / i_ph[1]) + i_ph[2] * np.exp(-v_sc / i_ph[3])
        coeff = 1e12 * q_e * 34 / (2 * np.sqrt(np.pi))
        coeff *= np.sqrt(2 * q_e * t_e / m_e) * (1 + v_sc / t_e)
        t_tensor = np.zeros((400, 3, 3))
        t_tensor[:, [0, 1, 2], [0, 1, 2]] = t_e[:, None]
        return (
            i_ph,
            pyrf.ts_scalar(time, v_sc),
            pyrf.ts_scalar(time, (current / coeff).astype(dtype)),
            pyrf.ts_tensor_xyz(time, t_tensor.astype(dtype)),
        )

    @data(np.float64, np.float32)
    def test_scpot2ne_two_populations(self, dtype):
        # The fit finds the two photoelectron populations again, and the
        # density from the potential is the density. The second population
        # was subtracted in the fit, though added in the reconstruction
        i_ph, sc_pot, n_e, t_e = self._case(dtype)
        n_esc, *params = mms.scpot2ne(sc_pot, n_e, t_e)
        np.testing.assert_allclose(params, i_ph, rtol=1e-4)
        np.testing.assert_allclose(n_esc.data, n_e.data, rtol=1e-4)

    def test_scpot2ne_scalar_temperature(self):
        # A scalar temperature gives the result of the tensor with that trace
        # (an assert refused it, though documented)
        _, sc_pot, n_e, t_e = self._case()
        t_scalar = pyrf.ts_scalar(
            t_e.time.data, np.trace(t_e.data, axis1=1, axis2=2) / 3
        )
        result = mms.scpot2ne(sc_pot, n_e, t_scalar)
        expected = mms.scpot2ne(sc_pot, n_e, t_e)
        np.testing.assert_allclose(result[0].data, expected[0].data)
        np.testing.assert_allclose(result[1:], expected[1:])

    def test_scpot2ne_invalid_temperature(self):
        _, sc_pot, n_e, t_e = self._case()
        t_vector = pyrf.ts_vec_xyz(t_e.time.data, np.ones((len(t_e), 3)))

        with self.assertRaises(ValueError):
            mms.scpot2ne(sc_pot, n_e, t_vector)

    def test_scpot2ne_silent(self):
        # scipy's fmin printed its convergence messages
        _, sc_pot, n_e, t_e = self._case()

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            mms.scpot2ne(sc_pot, n_e, t_e)

        self.assertEqual(stdout.getvalue(), "")


class VdfReduceDeprecationTestCase(unittest.TestCase):
    # Deprecated in 2.5.0: FutureWarning (shown to users), pointing at the caller
    def setUp(self):
        self.vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species="ions")

    def test_vdf_frame_transformation_deprecated(self):
        v_gse = generate_ts(64.0, 42, tensor_order=1)

        with self.assertWarnsRegex(FutureWarning, "pyrfu.mms.vdf_shift") as context:
            mms.vdf_frame_transformation(self.vdf, v_gse)

        self.assertEqual(context.filename, __file__)

    def test_vdf_reduce_deprecated(self):
        tint = list(pyrf.datetime642iso8601(self.vdf.time.data[[0, -1]]))

        with self.assertWarnsRegex(FutureWarning, "pyrfu.mms.reduce") as context:
            mms.vdf_reduce(self.vdf, tint, "1d", [1, 0, 0], n_vpt=10)

        self.assertEqual(context.filename, __file__)


class VdfToE64TestCase(unittest.TestCase):
    def test_vdf_to_e64_delta_energy(self):
        # The input attrs are unchanged; the 64-channel widths are the same at
        # every time, the top/bottom edges from the 32-channel widths (they were
        # written into the last/first time step)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=True)
        attrs = {k: np.array(v, copy=True) for k, v in vdf.attrs.items()}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # zero energy channel
            result = mms.vdf_to_e64(vdf)

        for key, value in attrs.items():
            np.testing.assert_array_equal(vdf.attrs[key], value)

        d_plus = result.attrs["delta_energy_plus"]
        d_minus = result.attrs["delta_energy_minus"]
        for d_e in [d_plus, d_minus]:
            np.testing.assert_array_equal(d_e, np.broadcast_to(d_e[0], d_e.shape))
        self.assertEqual(d_plus[0, -1], np.max(attrs["delta_energy_plus"][:, -1]))
        self.assertEqual(d_minus[0, 0], np.min(attrs["delta_energy_minus"][:, 0]))

    def test_vdf_to_e64_one_table(self):
        # energy0 equal to energy1: returned unchanged (a copy), as irfu-matlab
        # PDist.e64; it was merged into 64 duplicated channels
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=False)

        with self.assertLogs("pyrfu", level="INFO"):
            result = mms.vdf_to_e64(vdf)

        xr.testing.assert_identical(result, vdf)
        self.assertIsNot(result.data.data, vdf.data.data)

    def test_vdf_to_e64_delta_time(self):
        # A pair of samples spans the two accumulations: twice their half
        # widths for regular sampling, as irfu-matlab PDist.e64
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=True)
        half = 0.5 / 64.0
        vdf.attrs["delta_time_minus"] = np.full(10, half)
        vdf.attrs["delta_time_plus"] = np.full(10, half)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # zero energy channel
            result = mms.vdf_to_e64(vdf)

        for key in ["delta_time_minus", "delta_time_plus"]:
            self.assertEqual(len(result.attrs[key]), len(result.time))
            np.testing.assert_allclose(result.attrs[key], 2 * half, rtol=1e-6)


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

    def setUp(self):
        # Energies 0, ..., 31 and distinct widths for each channel
        self.vdf = generate_vdf(64.0, 10, [32, 32, 16])
        self.vdf.attrs["delta_energy_plus"] = np.tile(np.arange(32) * 0.1, (10, 1))
        self.vdf.attrs["delta_energy_minus"] = np.arange(32) * 0.2

    def test_vdf_elim_interval(self):
        result = mms.vdf_elim(self.vdf, [5.0, 10.0])

        # Channels strictly within the interval, energy widths clipped too
        np.testing.assert_array_equal(result.energy.data[0], [6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(result.data.data, self.vdf.data.data[:, 6:10])
        np.testing.assert_allclose(
            result.attrs["delta_energy_plus"],
            self.vdf.attrs["delta_energy_plus"][:, 6:10],
        )
        np.testing.assert_allclose(
            result.attrs["delta_energy_minus"], [1.2, 1.4, 1.6, 1.8]
        )

        # The caller's attributes are unchanged
        self.assertEqual(self.vdf.attrs["delta_energy_plus"].shape, (10, 32))

    def test_vdf_elim_two_tables(self):
        # Union of the channels of the two tables (0, 1, ... and 0.5, 1.5, ...)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=True)
        result = mms.vdf_elim(vdf, [5.0, 10.0])
        self.assertEqual(result.energy.shape, (10, 5))
        self.assertEqual(result.attrs["delta_energy_plus"].shape, (10, 5))

    def test_vdf_elim_single_energy(self):
        # One time step: the closest channel of the energy table
        result = mms.vdf_elim(self.vdf.isel(time=[0]), 5.2)
        np.testing.assert_array_equal(result.energy.data, [[5.0]])
        np.testing.assert_allclose(result.attrs["delta_energy_minus"], [1.0])

    def test_vdf_elim_empty(self):
        with self.assertRaisesRegex(ValueError, "No energy channel"):
            mms.vdf_elim(self.vdf, [100.0, 200.0])


@ddt
class VdfOmniTestCase(unittest.TestCase):
    @data("mean", "sum")
    def test_vdf_omni_output(self, method):
        result = mms.vdf_omni(generate_vdf(64.0, 100, (32, 32, 16)), method)
        self.assertIsInstance(result, xr.DataArray)

    @staticmethod
    def _vdf(energy01=False):
        # f = (i_energy + 1) (i_theta + 1) on FPI-like theta (centres of 16 bins)
        vdf = generate_vdf(
            64.0, 4, [32, 32, 16], energy01=energy01, species="ions", units="s^3/cm^6"
        )
        vdf["theta"] = 5.625 + 11.25 * vdf.theta
        vdf["energy"] = vdf.energy + 10.0
        e_idx, t_idx = np.arange(32.0)[:, None, None], np.arange(16.0)[None, None, :]
        vdf.data.data[:] = (e_idx + 1) * (t_idx + 1) * np.ones((32, 32, 16))
        return vdf

    @staticmethod
    def _expected(method):
        g_theta = np.arange(16.0) + 1
        sin_t = np.sin(np.deg2rad(5.625 + 11.25 * np.arange(16)))
        if method == "mean":
            per_e = np.sum(g_theta * sin_t) / np.sum(sin_t)
        else:
            per_e = 32 * np.sum(g_theta)
        return (np.arange(32.0) + 1) * per_e

    def test_vdf_omni_values(self):
        for method in ["mean", "sum", "MEAN"]:
            result = mms.vdf_omni(self._vdf(), method)
            np.testing.assert_allclose(
                result.data, np.broadcast_to(self._expected(method.lower()), (4, 32))
            )
            self.assertEqual(result.attrs["UNITS"], "s^3/cm^6")

    def test_vdf_omni_angular_widths(self):
        # FEEPS-like rings of different widths (delta_theta_*): the mean is
        # weighted by the solid angles of the bands, sum(f Omega) / 4 pi
        n_t, theta = 3, np.array([5.0, 65.0, 115.0, 175.0])
        edges = np.array([0.0, 35.0, 90.0, 145.0, 180.0])
        ring = np.array([1.0, 2.0, 4.0, 8.0])
        vdf = pyrf.ts_skymap(
            generate_timeline(1.0, n_t),
            np.broadcast_to(ring, (n_t, 5, 8, 4)).copy(),
            np.tile(1e3 * np.arange(1, 6), (n_t, 1)),
            np.tile(22.5 + 45.0 * np.arange(8), (n_t, 1)),
            theta,
            energy0=1e3 * np.arange(1, 6),
            energy1=1e3 * np.arange(1, 6),
            esteptable=np.zeros(n_t, dtype=np.uint8),
            attrs={"UNITS": "s^3/cm^6"},
            glob_attrs={
                "species": "electrons",
                "delta_theta_minus": theta - edges[:-1],
                "delta_theta_plus": edges[1:] - theta,
            },
        )
        d_omega = 2 * np.pi * -np.diff(np.cos(np.deg2rad(edges)))
        expected = np.sum(ring * d_omega) / (4 * np.pi)
        np.testing.assert_allclose(mms.vdf_omni(vdf).data, expected, rtol=1e-12)

    def test_vdf_omni_frame_isotropic(self):
        # Isotropic, no frame velocity: the spacecraft-frame omni exactly
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        result = mms.vdf_omni(vdf, v_frame=[0.0, 0.0, 0.0], n_mc=500)
        np.testing.assert_allclose(result.data, mms.vdf_omni(vdf).data, rtol=1e-12)
        np.testing.assert_array_equal(result.attrs["coverage"], 1.0)
        np.testing.assert_array_equal(result.attrs["frame_velocity"], 0.0)
        self.assertEqual(result.attrs["n_mc"], 500)
        self.assertEqual(result.attrs["UNITS"], "s^3/cm^6")

        with self.assertRaises(ValueError):
            mms.vdf_omni(vdf, "sum", v_frame=[0.0, 0.0, 0.0])

    def test_vdf_omni_frame_maxwellian(self):
        # Drifting Maxwellian in its frame: the Maxwellian at rest
        v_kms = np.array([200.0, 150.0, -250.0])
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, list(v_kms))
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        result = mms.vdf_omni(vdf, v_frame=v_kms, random_state=0)

        energy = vdf.energy.data[0]
        v_th2 = 2 * constants.e * 1000.0 / constants.m_p
        expected = np.exp(-energy / 1000.0) * 1e-6 / (np.pi * v_th2) ** 1.5
        low = energy < 2e3
        np.testing.assert_allclose(
            result.data[:, low],
            np.broadcast_to(expected[low], (2, low.sum())),
            rtol=0.03,
        )

        # Time series of the frame velocity, other energies
        v_frame = pyrf.ts_vec_xyz(vdf.time.data, np.tile(v_kms, (2, 1)))
        again = mms.vdf_omni(vdf, v_frame=v_frame, random_state=0)
        np.testing.assert_array_equal(again.data, result.data)
        np.testing.assert_array_equal(
            again.attrs["frame_velocity"], np.tile(v_kms, (2, 1))
        )

        energy = np.array([100.0, 1000.0, 1e6])
        result = mms.vdf_omni(vdf, v_frame=v_kms, energy=energy, random_state=0)
        np.testing.assert_array_equal(result.energy.data, energy)
        np.testing.assert_allclose(
            result.data[:, 1], np.exp(-1.0) * 1e-6 / (np.pi * v_th2) ** 1.5, rtol=0.03
        )

        # Outside the energy range: zero, without coverage
        np.testing.assert_array_equal(result.data[:, 2], 0.0)
        np.testing.assert_array_equal(result.attrs["coverage"][:, 2], 0.0)

        # Bins without data count as zero, with less coverage: without the
        # look directions theta < 90 (v_z < 0), the sphere of radius v_1keV
        # centred on v_frame keeps (v_1keV - |v_z|) / (2 v_1keV) (Archimedes)
        vdf.data.data[:, :, :, :8] = np.nan
        result = mms.vdf_omni(vdf, v_frame=v_kms, energy=energy, random_state=0)
        radius = np.sqrt(v_th2) / 1e3
        expected = (radius - 250.0) / (2 * radius)
        np.testing.assert_allclose(result.attrs["coverage"][:, 1], expected, atol=0.01)

    def test_vdf_omni_energy_tables(self):
        # Alternating tables: Dataset with the units and attributes of vdf, so
        # that the unit conversions work on it
        vdf = self._vdf(energy01=True)
        result = mms.vdf_omni(vdf)

        self.assertIsInstance(result, xr.Dataset)
        np.testing.assert_array_equal(result.energy.data, vdf.energy.data)
        np.testing.assert_allclose(result.data.data[0], self._expected("mean"))
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6")
        self.assertEqual(result.attrs["species"], "ions")

        dpf = mms.psd2dpf(result)
        self.assertEqual(dpf.data.attrs["UNITS"], "1/(cm^2 s sr keV)")

    def test_vdf_omni_shapes(self):
        vdf = self._vdf()

        # One time step
        result = mms.vdf_omni(vdf.isel(time=[0]))
        self.assertEqual(result.shape, (1, 32))

        # 1-D phi
        vdf["phi"] = xr.DataArray(vdf.phi.data[0], dims=["idx1"])
        np.testing.assert_allclose(mms.vdf_omni(vdf).data[0], self._expected("mean"))

    def test_vdf_omni_nan_channel(self):
        # A channel without data: NaN (mean) or 0 (sum), without warnings
        vdf = self._vdf()
        vdf.data.data[:, 3] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            mean = mms.vdf_omni(vdf, "mean").data
            total = mms.vdf_omni(vdf, "sum").data

        self.assertTrue(np.all(np.isnan(mean[:, 3])))
        np.testing.assert_array_equal(total[:, 3], 0.0)

    def test_vdf_omni_invalid_method(self):
        with self.assertRaises(ValueError):
            mms.vdf_omni(self._vdf(), "bazinga")


@ddt
class VdfConvertTestCase(unittest.TestCase):
    def test_vdf_convert_round_trips(self):
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")
        vdf = vdf.assign(energy=(vdf.energy + 1.0) * 10.0)
        units = [
            "s^3/m^6",
            "s^3/km^6",
            "keV/(cm^2 s sr keV)",
            "eV/(cm^2 s sr eV)",
            "1/(cm^2 s sr keV)",
            "1/(cm^2 s sr eV)",
        ]

        for unit in units:
            result = mms.vdf_convert(vdf, unit)
            self.assertEqual(result.data.attrs["UNITS"], unit)
            back = mms.vdf_convert(result, "s^3/cm^6")
            np.testing.assert_allclose(back.data.data, vdf.data.data, rtol=1e-12)

        # The existing converters, and the per eV flux is 1e-3 the per keV one
        np.testing.assert_allclose(
            mms.vdf_convert(vdf, "keV/(cm^2 s sr keV)").data.data,
            mms.psd2def(vdf).data.data,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            mms.vdf_convert(vdf, "1/(cm^2 s sr eV)").data.data,
            1e-3 * mms.psd2dpf(vdf).data.data,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            mms.vdf_convert(vdf, "s^3/km^6").data.data, 1e30 * vdf.data.data
        )

        # Same units: a copy
        result = mms.vdf_convert(vdf, "S^3/CM^6")
        np.testing.assert_array_equal(result.data.data, vdf.data.data)
        self.assertIsNot(result.data.data, vdf.data.data)

    def test_vdf_convert_relativistic(self):
        # Electrons at 100 keV: the relativistic attribute is used by default
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "electrons")
        vdf.data.attrs["UNITS"] = "1/(cm^2 s sr keV)"
        vdf = vdf.assign(energy=(vdf.energy + 1.0) * 1e4)
        vdf.attrs["relativistic"] = True

        psd = mms.vdf_convert(vdf, "s^3/m^6")
        expected = mms.dpf2psd(vdf, relativistic=True)
        np.testing.assert_allclose(psd.data.data, expected.data.data, rtol=1e-12)
        non_rel = mms.vdf_convert(vdf, "s^3/m^6", relativistic=False)
        np.testing.assert_allclose(
            non_rel.data.data, mms.dpf2psd(vdf).data.data, rtol=1e-12
        )

        # Energy fluxes: J = DEF / E, so the same factor
        def_e = mms.vdf_convert(vdf, "keV/(cm^2 s sr keV)")
        np.testing.assert_allclose(
            mms.vdf_convert(def_e, "s^3/m^6").data.data, psd.data.data, rtol=1e-12
        )
        back = mms.vdf_convert(psd, "keV/(cm^2 s sr keV)")
        np.testing.assert_allclose(back.data.data, def_e.data.data, rtol=1e-12)

    def test_vdf_convert_input(self):
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")

        with self.assertRaises(ValueError):
            mms.vdf_convert(vdf, "counts")

        vdf.data.attrs["UNITS"] = "counts"

        with self.assertRaises(ValueError):
            mms.vdf_convert(vdf, "s^3/m^6")


@ddt
class VdfVelocityTestCase(unittest.TestCase):
    @staticmethod
    def _vdf(species):
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, species, "s^3/cm^6")
        return vdf.assign(energy=(vdf.energy + 1.0) * 100.0)

    def test_vdf_velocity_values(self):
        vdf = self._vdf("ions")
        result = mms.vdf_velocity(vdf)
        self.assertEqual(result.vx.attrs["UNITS"], "km/s")

        # Speed of the energy, along minus the look direction
        speed = np.sqrt(result.vx**2 + result.vy**2 + result.vz**2).data
        expected = np.sqrt(2 * constants.e * vdf.energy.data / constants.m_p) / 1e3
        np.testing.assert_allclose(
            speed, np.broadcast_to(expected[..., None, None], speed.shape)
        )
        phi, theta = np.deg2rad(vdf.phi.data[1, 7]), np.deg2rad(vdf.theta.data[3])
        look = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        v_bin = [result[name].data[1, 5, 7, 3] for name in ["vx", "vy", "vz"]]
        np.testing.assert_allclose(v_bin, -expected[1, 5] * look)

        # Rotation: rows are the new unit vectors
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = mms.vdf_velocity(vdf, rotation=rotation)
        np.testing.assert_allclose(rotated.vx.data, result.vz.data)
        np.testing.assert_allclose(rotated.vy.data, result.vx.data)
        rotated = mms.vdf_velocity(vdf, rotation=np.tile(rotation, (4, 1, 1)))
        np.testing.assert_allclose(rotated.vz.data, result.vy.data)

        with self.assertRaises(ValueError):
            mms.vdf_velocity(vdf, rotation=np.eye(2))

    def test_vdf_velocity_sc_pot(self):
        # Ions gain and electrons lose e V, as irfu-matlab PDist.v
        for species, sign in [("ions", 1.0), ("electrons", -1.0)]:
            vdf = self._vdf(species)
            mass = constants.m_p if species == "ions" else constants.m_e
            sc_pot = pyrf.ts_scalar(vdf.time.data, np.full(4, 150.0))
            speed = mms.vdf_velocity(vdf, sc_pot=sc_pot).vx.data
            speed = np.abs(speed[:, :, 0, 8]) / np.sin(np.deg2rad(vdf.theta.data[8]))
            speed /= np.abs(np.cos(np.deg2rad(vdf.phi.data[:, :1])))
            energy = np.clip(vdf.energy.data + sign * 150.0, 0.0, None)
            np.testing.assert_allclose(
                speed, np.sqrt(2 * constants.e * energy / mass) / 1e3, rtol=1e-6
            )
            for other in [150.0, np.full(4, 150.0)]:
                np.testing.assert_array_equal(
                    mms.vdf_velocity(vdf, sc_pot=other).vx.data,
                    mms.vdf_velocity(vdf, sc_pot=sc_pot).vx.data,
                )

        with self.assertRaises(ValueError):
            mms.vdf_velocity(vdf, sc_pot=np.ones(3))

    def test_vdf_velocity_relativistic(self):
        vdf = self._vdf("electrons")
        vdf = vdf.assign(energy=vdf.energy * 100.0)
        speed = np.abs(mms.vdf_velocity(vdf).vz.data[:, -1, 0, 0])
        rel = np.abs(mms.vdf_velocity(vdf, relativistic=True).vz.data[:, -1, 0, 0])
        self.assertTrue(np.all(speed > constants.c / 1e3))
        self.assertTrue(np.all(rel < constants.c / 1e3))

        vdf.attrs["relativistic"] = True
        np.testing.assert_array_equal(
            np.abs(mms.vdf_velocity(vdf).vz.data[:, -1, 0, 0]), rel
        )

        # 10 eV protons: v_rel / v = 1 - 3 eps / 4 (eps = E / m c^2 = 1e-8),
        # without cancellation errors
        vdf = self._vdf("ions")
        vdf = vdf.assign(energy=vdf.energy * 0.0 + 10.0)
        ratio = (
            mms.vdf_velocity(vdf, relativistic=True).vz.data[:, :, 0, 0]
            / mms.vdf_velocity(vdf).vz.data[:, :, 0, 0]
        )
        eps = 10.0 * constants.e / (constants.m_p * constants.c**2)
        np.testing.assert_allclose(ratio, 1 - 3 * eps / 4, rtol=1e-12)


@ddt
class VdfSolidAngleTestCase(unittest.TestCase):
    def test_vdf_solid_angle_skymap(self):
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")
        vdf = vdf.assign(phi=vdf.phi * 0.0 + 5.625 + 11.25 * np.arange(32))
        vdf = vdf.assign(theta=vdf.theta * 0.0 + 5.625 + 11.25 * np.arange(16))
        result = mms.vdf_solid_angle(vdf)
        self.assertEqual(result.data.attrs["UNITS"], "sr")
        self.assertEqual(result.data.shape, vdf.data.shape)
        np.testing.assert_allclose(result.data.data.sum(axis=(2, 3)), 4 * np.pi)

        # Angular widths of the attributes (FEEPS: 4 rings of 8 bins)
        theta, edges = np.array([5.0, 65.0, 115.0, 175.0]), np.array(
            [0.0, 35.0, 90.0, 145.0, 180.0]
        )
        vdf = vdf.isel(idx1=np.arange(8), idx2=np.arange(4))
        vdf = vdf.assign(phi=vdf.phi * 0.0 + 22.5 + 45.0 * np.arange(8))
        vdf = vdf.assign(theta=(["idx2"], theta))
        vdf.attrs["delta_phi_minus"] = vdf.attrs["delta_phi_plus"] = np.full(
            (4, 8), 22.5
        )
        vdf.attrs["delta_theta_minus"] = theta - edges[:-1]
        vdf.attrs["delta_theta_plus"] = edges[1:] - theta
        result = mms.vdf_solid_angle(vdf).data.data
        np.testing.assert_allclose(result.sum(axis=(2, 3)), 4 * np.pi)
        expected = np.pi / 4 * (np.cos(np.deg2rad(35.0)) - np.cos(np.pi / 2))
        np.testing.assert_allclose(result[:, :, :, 1], expected)

    def test_vdf_solid_angle_pad(self):
        pad = _pad_case(np.ones((3, 2, 12)))
        result = mms.vdf_solid_angle(pad).data.data
        np.testing.assert_allclose(result.sum(axis=2), 4 * np.pi)
        np.testing.assert_allclose(
            result[0, 0, 0], 2 * np.pi * (1 - np.cos(np.deg2rad(15.0)))
        )

        with self.assertRaises(ValueError):
            mms.vdf_solid_angle(pad.isel(idx1=0))


@ddt
class VdfD3vTestCase(unittest.TestCase):
    def test_vdf_d3v_sphere(self):
        # The bins of the sphere: 4 pi / 3 (v_2^3 - v_1^3) per energy channel
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        lower = vdf.energy.data - vdf.attrs["delta_energy_minus"]
        upper = vdf.energy.data + vdf.attrs["delta_energy_plus"]
        v_1, v_2 = (
            np.sqrt(2 * constants.e * e / constants.m_p) for e in [lower, upper]
        )
        expected = 4 * np.pi / 3 * (v_2**3 - v_1**3)

        result = mms.vdf_d3v(vdf)
        self.assertEqual(result.data.attrs["UNITS"], "cm^3/s^3")
        np.testing.assert_allclose(
            result.data.data.sum(axis=(2, 3)), 1e6 * expected, rtol=1e-12
        )

        for units, unit_volume, factor in [
            ("s^3/km^6", "km^3/s^3", 1e-9),
            ("s^3/m^6", "m^3/s^3", 1.0),
            ("1/(cm^2 s sr keV)", "m^3/s^3", 1.0),
        ]:
            vdf.data.attrs["UNITS"] = units
            result = mms.vdf_d3v(vdf)
            self.assertEqual(result.data.attrs["UNITS"], unit_volume)
            np.testing.assert_allclose(
                result.data.data.sum(axis=(2, 3)), factor * expected, rtol=1e-12
            )

        # Without the energy widths: midpoints between the channels
        del vdf.attrs["delta_energy_minus"], vdf.attrs["delta_energy_plus"]
        energy = vdf.energy.data[0]
        result = mms.vdf_d3v(vdf).data.data[0, 5].sum()
        e_1, e_2 = (energy[4] + energy[5]) / 2, (energy[5] + energy[6]) / 2
        v_1, v_2 = (np.sqrt(2 * constants.e * e / constants.m_p) for e in [e_1, e_2])
        np.testing.assert_allclose(result, 4 * np.pi / 3 * (v_2**3 - v_1**3))

    def test_vdf_d3v_sc_pot_pad(self):
        # Ions gain e V (irfu-matlab PDist.d3v subtracts V for all species)
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/m^6"
        result = mms.vdf_d3v(vdf, sc_pot=20.0).data.data[0, 3].sum()
        lower = vdf.energy.data[0, 3] - vdf.attrs["delta_energy_minus"][0, 3] + 20.0
        upper = vdf.energy.data[0, 3] + vdf.attrs["delta_energy_plus"][0, 3] + 20.0
        v_1, v_2 = (
            np.sqrt(2 * constants.e * e / constants.m_p) for e in [lower, upper]
        )
        np.testing.assert_allclose(result, 4 * np.pi / 3 * (v_2**3 - v_1**3))

        pad = _pad_case(np.ones((3, 2, 12)))
        result = mms.vdf_d3v(pad)
        self.assertEqual(result.data.attrs["UNITS"], "cm^3/s^3")
        v_1, v_2 = (
            np.sqrt(2 * constants.e * f * pad.energy.data / constants.m_p)
            for f in [0.75, 1.25]
        )
        np.testing.assert_allclose(
            result.data.data.sum(axis=2), 1e6 * 4 * np.pi / 3 * (v_2**3 - v_1**3)
        )


@ddt
class VdfPartialDensityTestCase(unittest.TestCase):
    def test_vdf_partial_density_moments(self):
        # Sums are the density and particle flux of a drifting Maxwellian
        v_kms = [200.0, 150.0, -250.0]
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_kms)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        d_n = mms.vdf_partial_density(vdf)
        self.assertEqual(d_n.data.attrs["UNITS"], "cm^-3")
        n_i = d_n.data.data.sum(axis=(1, 2, 3))
        np.testing.assert_allclose(n_i, 1.0, rtol=0.01)

        # Equal to f d3v
        d3v = mms.vdf_d3v(vdf).data.data
        np.testing.assert_allclose(d_n.data.data, vdf.data.data * d3v, rtol=1e-12)

        for i, comp in enumerate("xyz"):
            d_j = mms.vdf_partial_density(vdf, comp)
            self.assertEqual(d_j.data.attrs["UNITS"], "1/(cm^2 s)")
            j_i = d_j.data.data.sum(axis=(1, 2, 3))
            np.testing.assert_allclose(j_i / n_i, v_kms[i] * 1e5, rtol=0.01)

        # Any units: the same density from the differential particle flux
        dpf = mms.vdf_convert(vdf, "1/(cm^2 s sr keV)")
        np.testing.assert_allclose(
            mms.vdf_partial_density(dpf).data.data, d_n.data.data, rtol=1e-12
        )

    def test_vdf_partial_density_input(self):
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        with self.assertRaises(ValueError):
            mms.vdf_partial_density(vdf, "w")

        pad = _pad_case(np.ones((3, 2, 12)))
        n_pad = mms.vdf_partial_density(pad).data.data
        np.testing.assert_allclose(
            n_pad, pad.data.data * mms.vdf_d3v(pad).data.data, rtol=1e-12
        )

        with self.assertRaises(ValueError):
            mms.vdf_partial_density(pad, "x")


@ddt
class VdfFluxTestCase(unittest.TestCase):
    def test_vdf_flux_maxwellian(self):
        # Isotropic Maxwellian: the flux per steradian summed over the energies
        # is n v_th / (2 pi^1.5)
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, [0.0] * 3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        result = mms.vdf_flux(vdf)
        self.assertEqual(result.data.attrs["UNITS"], "1/(cm^2 s sr)")

        v_th = np.sqrt(2 * constants.e * 1000.0 / constants.m_p) * 1e2  # cm/s
        np.testing.assert_allclose(
            result.data.data.sum(axis=1), v_th / (2 * np.pi**1.5), rtol=0.02
        )

        # Through the solid angle of the bins
        total = mms.vdf_flux(vdf, per_sr=False)
        self.assertEqual(total.data.attrs["UNITS"], "1/(cm^2 s)")
        np.testing.assert_allclose(
            total.data.data,
            result.data.data * mms.vdf_solid_angle(vdf).data.data,
            rtol=1e-12,
        )

    def test_vdf_flux_pad(self):
        # (v_2^4 - v_1^4) / 4 f, the ion mass (irfu-matlab uses m_e)
        pad = _pad_case(np.ones((3, 2, 12)))
        v_1, v_2 = (
            np.sqrt(2 * constants.e * f * pad.energy.data / constants.m_p)
            for f in [0.75, 1.25]
        )
        expected = 1e12 * (v_2**4 - v_1**4) / 4 / 1e4
        np.testing.assert_allclose(
            mms.vdf_flux(pad).data.data[..., 4], expected, rtol=1e-12
        )


@ddt
class VdfMaskTestCase(unittest.TestCase):
    @staticmethod
    def _vdf():
        # Alternating energy tables
        vdf = generate_vdf(64.0, 4, (32, 32, 16), False, "ions", "s^3/cm^6")
        energy = np.tile(10.0 * 1.3 ** np.arange(32), (4, 1))
        energy[1::2] *= 1.1
        return vdf.assign(energy=(["time", "idx0"], energy))

    def test_vdf_mask_limits(self):
        vdf = self._vdf()
        energy = vdf.energy.data

        # Energies of each time
        result = mms.vdf_mask(vdf, below=13.5)
        masked = np.all(np.isnan(result.data.data), axis=(2, 3))
        np.testing.assert_array_equal(masked, energy < 13.5)
        self.assertTrue(masked[0, 1] and not masked[1, 1])
        self.assertFalse(np.any(np.isnan(vdf.data.data)))

        limits = pyrf.ts_scalar(vdf.time.data, np.array([20.0, 30.0, 40.0, 50.0]))
        result = mms.vdf_mask(vdf, below=limits, above=1000.0)
        masked = np.all(np.isnan(result.data.data), axis=(2, 3))
        expected = (energy < limits.data[:, None]) | (energy > 1000.0)
        np.testing.assert_array_equal(masked, expected)
        np.testing.assert_array_equal(
            np.any(np.isnan(result.data.data), axis=(2, 3)), expected
        )

        # Limits on other times are resampled
        shifted = pyrf.ts_scalar(
            vdf.time.data + np.timedelta64(1, "ms"), np.full(4, 30.0)
        )
        result = mms.vdf_mask(vdf, below=shifted)
        np.testing.assert_array_equal(
            np.all(np.isnan(result.data.data), axis=(2, 3)), energy < 30.0
        )

    def test_vdf_mask_matrix(self):
        vdf = self._vdf()
        mask = np.zeros((4, 32), dtype=int)
        mask[:, [2, 5]] = 1
        mask[3, 20] = 1
        result = mms.vdf_mask(vdf, mask=mask)
        np.testing.assert_array_equal(
            np.all(np.isnan(result.data.data), axis=(2, 3)), mask.astype(bool)
        )

        with self.assertRaises(ValueError):
            mms.vdf_mask(vdf, mask=mask[:, :10])

        with self.assertRaises(ValueError):
            mms.vdf_mask(vdf, below=np.ones(3))

        # Pitch-angle distributions
        pad = _pad_case(np.ones((3, 2, 12)))
        result = mms.vdf_mask(pad, above=150.0).data.data
        self.assertTrue(np.all(np.isnan(result[:, 1])))
        self.assertFalse(np.any(np.isnan(result[:, 0])))


@ddt
class VdfFindLowCountsTestCase(unittest.TestCase):
    def test_movmean(self):
        # MATLAB movmean: centred window, shrinking at the ends
        module = sys.modules["pyrfu.mms.vdf_find_low_counts"]
        inp = np.array([1.0, 2.0, 4.0, 8.0, 16.0])[:, None]
        np.testing.assert_allclose(module._movmean(inp, 1)[:, 0], inp[:, 0])
        np.testing.assert_allclose(
            module._movmean(inp, 3)[:, 0], [1.5, 7 / 3, 14 / 3, 28 / 3, 12.0]
        )
        # Even window: k / 2 back, k / 2 - 1 forward
        np.testing.assert_allclose(
            module._movmean(inp, 4)[:, 0], [1.5, 7 / 3, 15 / 4, 30 / 4, 28 / 3]
        )
        np.testing.assert_allclose(
            module._movmean(inp, [0, 2])[:, 0], [7 / 3, 14 / 3, 28 / 3, 12.0, 16.0]
        )

    def test_vdf_find_low_counts(self):
        counts = generate_vdf(64.0, 3, (4, 2, 2), False, "ions", "counts")
        counts = counts.assign(
            energy=(["time", "idx0"], np.tile([10.0, 20.0, 40.0, 80.0], (3, 1)))
        )
        summed = np.array([[0, 2, 8, 12], [1, 6, 9, 3], [0, 1, 2, 3]], dtype=float)
        counts["data"] = counts.data.copy(
            data=np.broadcast_to(summed[..., None, None] / 4, (3, 4, 2, 2)).copy()
        )

        mat = mms.vdf_find_low_counts(counts, 5, output="mat")
        np.testing.assert_array_equal(mat.data, (summed < 5).astype(int))

        energy = mms.vdf_find_low_counts(counts, 5)
        self.assertEqual(energy.attrs["UNITS"], "eV")
        np.testing.assert_array_equal(energy.data, [40.0, 20.0, np.nan])

        smoothed = mms.vdf_find_low_counts(counts, 5, n_mov_mean=3, output="mat")
        expected = np.array(
            [[0.5, 4, 8.5, 7.5], [1 / 3, 3, 19 / 3, 6], [0.5, 3.5, 5.5, 3]]
        )
        np.testing.assert_array_equal(smoothed.data, (expected < 5).astype(int))

        with self.assertRaises(ValueError):
            mms.vdf_find_low_counts(counts, 5, output="index")


@ddt
class VdfMacroparticlesTestCase(unittest.TestCase):
    @staticmethod
    def _vdf():
        v_kms = [200.0, 150.0, -250.0]
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_kms)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        vdf.data.data[:, :, 3, 5] = 0.0
        vdf.data.data[:, :, 4, 6] = np.nan
        return vdf, np.array(v_kms)

    def test_vdf_macroparticles_one_per_bin(self):
        vdf, v_kms = self._vdf()
        result = mms.vdf_macroparticles(vdf, random_state=1)
        self.assertEqual(result.vx.attrs["UNITS"], "km/s")
        self.assertEqual(result.dn.attrs["UNITS"], "cm^-3")
        self.assertEqual(result.dv.attrs["UNITS"], "cm^3/s^3")

        # One per bin with data, the density of each bin
        d_n = mms.vdf_partial_density(vdf).data.data
        self.assertEqual(result.sizes["particle"], np.sum(d_n > 0))
        n_t = result.dn.groupby("time").sum().data
        np.testing.assert_allclose(n_t, np.nansum(d_n, axis=(1, 2, 3)), rtol=1e-12)
        np.testing.assert_allclose(result.dn, result.df * result.dv, rtol=1e-12)

        # The velocities of the bins: speed and look direction in the bin
        v_xyz = np.stack([result[c].data for c in ["vx", "vy", "vz"]], axis=1)
        speed = np.linalg.norm(v_xyz, axis=1) * 1e3
        bins = sys.modules["pyrfu.mms._vdf_mc"]._source_bins(vdf)
        i_t = np.searchsorted(vdf.time.data, result.time.data)
        i_e, i_q = result.idx0.data, result.idx2.data
        self.assertTrue(np.all(speed >= bins["v_lower"][i_t, i_e] * (1 - 1e-12)))
        self.assertTrue(np.all(speed <= bins["v_upper"][i_t, i_e] * (1 + 1e-12)))
        theta = np.rad2deg(np.arccos(-v_xyz[:, 2] * 1e3 / speed))
        theta_lower = vdf.theta.data[i_q] - 5.625
        self.assertTrue(
            np.all((theta >= theta_lower - 1e-9) & (theta <= theta_lower + 11.26))
        )

        # Uniform in the volume of the bins (v^3)
        v3_lower, v3_upper = (bins[k][i_t, i_e] ** 3 for k in ["v_lower", "v_upper"])
        position = (speed**3 - v3_lower) / (v3_upper - v3_lower)
        self.assertAlmostEqual(float(np.mean(position)), 0.5, delta=0.01)

        # Bulk velocity, and reproducible
        v_mean = np.array(
            [
                float((result[c] * result.dn).sum() / result.dn.sum())
                for c in ["vx", "vy", "vz"]
            ]
        )
        np.testing.assert_allclose(v_mean, v_kms, atol=3.0)
        again = mms.vdf_macroparticles(vdf, random_state=np.random.default_rng(1))
        np.testing.assert_array_equal(again.vx.data, result.vx.data)

    def test_vdf_macroparticles_number(self):
        vdf, _ = self._vdf()
        d_n = np.nan_to_num(mms.vdf_partial_density(vdf).data.data)

        # n_tot: in proportion to dn, rounded up
        result = mms.vdf_macroparticles(vdf, n_tot=20000, random_state=1)
        expected = np.ceil(d_n / d_n.sum(axis=(1, 2, 3), keepdims=True) * 20000)
        counts = np.bincount(np.searchsorted(vdf.time.data, result.time.data))
        np.testing.assert_array_equal(counts, expected.sum(axis=(1, 2, 3)))
        np.testing.assert_allclose(
            result.dn.groupby("time").sum().data, d_n.sum(axis=(1, 2, 3)), rtol=1e-12
        )

        # Without skipping the empty bins: one with dn = 0
        result = mms.vdf_macroparticles(vdf, n_tot=20000, skip_zero=False)
        self.assertEqual(result.sizes["particle"], int(np.maximum(expected, 1).sum()))
        self.assertTrue(np.any(result.dn.data == 0.0))

        # One macroparticle per count
        counts = vdf.copy(deep=True)
        counts["data"] = counts.data.copy(
            data=np.random.default_rng(0).poisson(1.0, vdf.data.shape).astype(float)
        )
        counts.data.attrs["UNITS"] = "s^3/cm^6"
        total = int(counts.data.data[0].sum())
        result = mms.vdf_macroparticles(counts, n_tot=total, division="counts")
        first = result.time.data == vdf.time.data[0]
        self.assertEqual(np.sum(first), total)

        result = mms.vdf_macroparticles(vdf, n_tot=1000, division="f")
        self.assertGreater(result.sizes["particle"], 2000)

    def test_vdf_macroparticles_center(self):
        vdf, _ = self._vdf()
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.full(2, 5.0))
        result = mms.vdf_macroparticles(
            vdf, n_bin=2, positioning="center", sc_pot=sc_pot
        )
        velocity = mms.vdf_velocity(vdf, sc_pot=sc_pot)
        i_t = np.searchsorted(vdf.time.data, result.time.data)
        index = (i_t, result.idx0.data, result.idx1.data, result.idx2.data)
        np.testing.assert_array_equal(result.vx.data, velocity.vx.data[index])
        d_n = mms.vdf_partial_density(vdf, sc_pot=sc_pot).data.data
        self.assertEqual(result.sizes["particle"], 2 * np.sum(d_n > 0))

        for kwargs in [{"positioning": "edge"}, {"division": "n"}]:
            with self.assertRaises(ValueError):
                mms.vdf_macroparticles(vdf, **kwargs)

        with self.assertRaises(ValueError):
            mms.vdf_macroparticles(_pad_case(np.ones((3, 2, 12))))


@ddt
class VdfShiftTestCase(unittest.TestCase):
    @staticmethod
    def _vdf(v_kms):
        vdf, sc_pot = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_kms)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        return vdf, sc_pot

    @staticmethod
    def _moments(vdf, sc_pot):
        n_i, v_i, _, _, t_i, _ = mms.psd_moments(vdf, sc_pot, angle_integration="exact")
        return n_i.data, v_i.data, np.trace(t_i.data, axis1=1, axis2=2) / 3

    def test_vdf_shift_no_shift(self):
        # Every particle stays in its bin: the skymap is unchanged, apart from
        # the widened first and last energy bins
        vdf, _ = self._vdf([0.0, 0.0, 0.0])
        vdf.attrs["delta_time_minus"] = vdf.attrs["delta_time_plus"] = np.ones(2)
        result = mms.vdf_shift(vdf, [0.0, 0.0, 0.0], n_mc=20, random_state=0)
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6")
        np.testing.assert_array_equal(result.attrs["delta_time_plus"], np.ones(2))
        np.testing.assert_allclose(
            result.data.data[:, 1:-1], vdf.data.data[:, 1:-1], rtol=1e-9
        )
        np.testing.assert_allclose(
            result.energy.data[:, 1:-1], vdf.energy.data[:, 1:-1]
        )
        np.testing.assert_allclose(result.theta.data, vdf.theta.data)
        np.testing.assert_allclose(result.phi.data, vdf.phi.data)
        np.testing.assert_array_equal(result.attrs["lost"], 0.0)

        # Relativistic speeds: the same bins
        vdf.attrs["relativistic"] = True
        result = mms.vdf_shift(vdf, [0.0, 0.0, 0.0], n_mc=20, random_state=0)
        self.assertTrue(result.attrs["relativistic"])
        np.testing.assert_allclose(
            result.energy.data[:, 1:-1], vdf.energy.data[:, 1:-1], rtol=1e-9
        )
        np.testing.assert_allclose(
            result.data.data[:, 1:-1], vdf.data.data[:, 1:-1], rtol=1e-9
        )

    def test_vdf_shift_bulk_frame(self):
        v_kms = np.array([200.0, 150.0, -250.0])
        vdf, sc_pot = self._vdf(list(v_kms))
        result = mms.vdf_shift(vdf, v_kms, random_state=1)
        np.testing.assert_array_equal(
            result.attrs["frame_velocity"], np.tile(v_kms, (2, 1))
        )

        # The density is conserved, apart from the particles outside the grid
        n_old = mms.vdf_partial_density(vdf).data.data.sum(axis=(1, 2, 3))
        n_new = mms.vdf_partial_density(result).data.data.sum(axis=(1, 2, 3))
        lost = result.attrs["lost"]
        self.assertTrue(np.all((lost > 0) & (lost < 1e-3)))
        np.testing.assert_allclose(n_new, n_old * (1 - lost), rtol=1e-12)

        # At rest in the new frame, with the same density and temperature
        n_i, v_i, t_i = self._moments(result, sc_pot)
        n_0, _, t_0 = self._moments(vdf, sc_pot)
        np.testing.assert_allclose(n_i, n_0, rtol=0.01)
        np.testing.assert_allclose(v_i, 0.0, atol=2.0)
        np.testing.assert_allclose(t_i, t_0, rtol=0.03)

        # A time series of the frame velocity
        v_frame = pyrf.ts_vec_xyz(vdf.time.data, np.tile(v_kms, (2, 1)))
        again = mms.vdf_shift(vdf, v_frame, random_state=1)
        np.testing.assert_array_equal(again.data.data, result.data.data)

        # Any units
        dpf = mms.vdf_convert(vdf, "1/(cm^2 s sr keV)")
        result_dpf = mms.vdf_shift(dpf, v_kms, random_state=1)
        self.assertEqual(result_dpf.data.attrs["UNITS"], "1/(cm^2 s sr keV)")
        np.testing.assert_allclose(
            mms.vdf_convert(result_dpf, "s^3/cm^6").data.data,
            result.data.data,
            rtol=1e-12,
        )

    def test_vdf_shift_rotation_grid(self):
        v_kms = [200.0, 150.0, -250.0]
        vdf, sc_pot = self._vdf(v_kms)
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = mms.vdf_shift(vdf, [0.0, 0.0, 0.0], rotation=rotation, random_state=2)
        _, v_0, _ = self._moments(vdf, sc_pot)
        _, v_i, _ = self._moments(result, sc_pot)
        np.testing.assert_allclose(v_i, v_0 @ rotation.T, atol=3.0)

        # A coarse grid: 10 energies from 100 eV to 10 keV, 8 x 4 angles
        grid = {
            "energy": np.logspace(2, 4, 11),
            "phi": np.linspace(0.0, 360.0, 9),
            "theta": np.linspace(0.0, 180.0, 5),
        }
        result = mms.vdf_shift(vdf, v_kms, grid=grid, random_state=3)
        self.assertEqual(result.data.shape, (2, 10, 8, 4))
        np.testing.assert_allclose(
            result.energy.data[0], np.sqrt(grid["energy"][:-1] * grid["energy"][1:])
        )
        np.testing.assert_allclose(result.attrs["delta_theta_minus"], 22.5)
        n_old = mms.vdf_partial_density(vdf).data.data.sum(axis=(1, 2, 3))
        n_new = mms.vdf_partial_density(result).data.data.sum(axis=(1, 2, 3))
        # Lost: the energies below 100 eV and above 10 keV of the Maxwellian
        # at rest (1 keV), P(3/2, 0.1) + 1 - P(3/2, 10)
        lost = result.attrs["lost"]
        expected = gammainc(1.5, 0.1) + 1 - gammainc(1.5, 10.0)
        np.testing.assert_allclose(lost, expected, rtol=0.05)
        np.testing.assert_allclose(n_new, n_old * (1 - lost), rtol=1e-12)

        with self.assertRaises(ValueError):
            mms.vdf_shift(vdf, [1.0, 2.0])

        with self.assertRaises(ValueError):
            mms.vdf_shift(_pad_case(np.ones((3, 2, 12))), [0.0, 0.0, 0.0])


@ddt
class VdfRebinTestCase(unittest.TestCase):
    @staticmethod
    def _vdf():
        # Drifting Maxwellian with azimuths changing with time (FPI burst)
        v_kms = [200.0, 150.0, -250.0]
        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_kms, n_t=3)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        vdf = vdf.assign(phi=vdf.phi + np.array([0.0, 2.0, 4.0])[:, None])
        return vdf, np.array(v_kms)

    @staticmethod
    def _edges(vdf):
        lower = vdf.energy.data[0] - vdf.attrs["delta_energy_minus"][0]
        upper = vdf.energy.data[0] + vdf.attrs["delta_energy_plus"][0]
        return np.hstack([lower, upper[-1]])

    def test_vdf_rebin_energy(self):
        # Channels of two old channels: the overlap of the bins in v^3 exactly
        vdf, _ = self._vdf()
        edges = self._edges(vdf)
        result = mms.vdf_rebin(vdf, {"energy": edges[::2]}, random_state=0)
        self.assertEqual(result.data.shape, (3, 16, 32, 16))
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6")
        np.testing.assert_allclose(result.phi.data, vdf.phi.data)
        np.testing.assert_allclose(
            result.energy.data[0], np.sqrt(edges[:-2:2] * edges[2::2])
        )
        np.testing.assert_array_equal(result.attrs["lost"], 0.0)
        self.assertNotIn("frame_velocity", result.attrs)

        v_3 = (2 * constants.e * edges / constants.m_p) ** 1.5
        w_3 = np.diff(v_3)[None, :, None, None]
        f_old = vdf.data.data
        expected = (f_old[:, 0::2] * w_3[:, 0::2] + f_old[:, 1::2] * w_3[:, 1::2]) / (
            w_3[:, 0::2] + w_3[:, 1::2]
        )
        np.testing.assert_allclose(result.data.data, expected, rtol=1e-10)

    def test_vdf_rebin_energy_overlap(self):
        # EDI channel 475-525 eV across two FPI channels: the Monte-Carlo
        # estimate of the overlap in v^3
        vdf, _ = self._vdf()
        edges = self._edges(vdf)
        i_e = np.where((edges[1:] > 475.0) & (edges[:-1] < 525.0))[0]
        small = vdf.isel(idx0=i_e, idx1=np.arange(4), idx2=np.arange(6, 10))
        small.attrs = {
            **vdf.attrs,
            "delta_energy_minus": vdf.attrs["delta_energy_minus"][:, i_e],
            "delta_energy_plus": vdf.attrs["delta_energy_plus"][:, i_e],
        }
        result = mms.vdf_rebin(
            small, {"energy": [475.0, 525.0]}, n_mc=20000, random_state=0
        )
        self.assertEqual(result.data.shape, (3, 1, 4, 4))

        v_3 = (2 * constants.e * edges / constants.m_p) ** 1.5
        new_3 = (2 * constants.e * np.array([475.0, 525.0]) / constants.m_p) ** 1.5
        overlap = np.clip(
            np.minimum(v_3[i_e + 1], new_3[1]) - np.maximum(v_3[i_e], new_3[0]),
            0.0,
            None,
        )
        expected = np.sum(small.data.data * overlap[:, None, None], axis=1) / np.diff(
            new_3
        )
        np.testing.assert_allclose(result.data.data[:, 0], expected, rtol=0.03)

    def test_vdf_rebin_cartesian(self):
        vdf, v_kms = self._vdf()
        v_edges = np.linspace(-1500.0, 1500.0, 31)
        result = mms.vdf_rebin(vdf, [v_edges] * 3, base="cart", random_state=0)
        self.assertEqual(result.data.shape, (3, 30, 30, 30))
        self.assertEqual(result.data.attrs["UNITS"], "s^3/cm^6")
        np.testing.assert_allclose(result.vx.data, v_edges[:-1] + 50.0)
        np.testing.assert_array_equal(result.attrs["vz_edges"], v_edges)

        # Density and bulk velocity
        d_v3 = (100.0 * 1e5) ** 3  # cm^3/s^3
        n_old = mms.vdf_partial_density(vdf).data.data.sum(axis=(1, 2, 3))
        n_new = result.data.data.sum(axis=(1, 2, 3)) * d_v3
        lost = result.attrs["lost"]
        self.assertTrue(np.all((lost > 0) & (lost < 1e-3)))
        np.testing.assert_allclose(n_new, n_old * (1 - lost), rtol=1e-10)
        v_x, v_y, v_z = np.meshgrid(
            result.vx.data, result.vy.data, result.vz.data, indexing="ij"
        )
        v_mean = [
            np.sum(result.data.data[0] * v) / np.sum(result.data.data[0])
            for v in [v_x, v_y, v_z]
        ]
        np.testing.assert_allclose(v_mean, v_kms, atol=2.0)

        # The drifting Maxwellian in the boxes
        v_th2 = 2 * constants.e * 1000.0 / constants.m_p / 1e6
        d_v2 = (v_x - v_kms[0]) ** 2 + (v_y - v_kms[1]) ** 2 + (v_z - v_kms[2]) ** 2
        expected = 1e-6 / (np.pi * v_th2 * 1e6) ** 1.5 * np.exp(-d_v2 / v_th2)
        core = expected > 0.1 * expected.max()
        ratio = result.data.data[0][core] / expected[core]
        self.assertAlmostEqual(float(np.mean(ratio)), 1.0, delta=0.01)

        # Rotation: rows are the new axes
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = mms.vdf_rebin(
            vdf, [v_edges] * 3, base="cart", rotation=rotation, random_state=0
        )
        v_mean = [
            np.sum(rotated.data.data[0] * v) / np.sum(rotated.data.data[0])
            for v in [v_x, v_y, v_z]
        ]
        np.testing.assert_allclose(v_mean, rotation @ v_kms, atol=2.0)

    def test_vdf_rebin_cartesian_units(self):
        vdf, _ = self._vdf()
        v_edges = np.linspace(-1500.0, 1500.0, 16)
        expected = mms.vdf_rebin(vdf, [v_edges] * 3, base="cart", random_state=0)

        # Fluxes give the phase-space density in s^3/m^6
        dpf = mms.vdf_convert(vdf, "1/(cm^2 s sr keV)")
        result = mms.vdf_rebin(dpf, [v_edges] * 3, base="cart", random_state=0)
        self.assertEqual(result.data.attrs["UNITS"], "s^3/m^6")
        np.testing.assert_allclose(
            result.data.data, 1e12 * expected.data.data, rtol=1e-10
        )

        # Spacecraft potential: the density of the corrected bins
        result = mms.vdf_rebin(
            vdf, [v_edges] * 3, base="cart", sc_pot=20.0, random_state=0
        )
        d_v3 = (200.0 * 1e5) ** 3
        n_new = result.data.data.sum(axis=(1, 2, 3)) * d_v3
        n_old = mms.vdf_partial_density(vdf, sc_pot=20.0).data.data.sum(axis=(1, 2, 3))
        np.testing.assert_allclose(
            n_new, n_old * (1 - result.attrs["lost"]), rtol=1e-10
        )

        with self.assertRaises(ValueError):
            mms.vdf_rebin(vdf, [v_edges] * 2, base="cart")

        with self.assertRaises(ValueError):
            mms.vdf_rebin(vdf, [v_edges] * 3, base="box")

        with self.assertRaises(ValueError):
            mms.vdf_rebin(_pad_case(np.ones((3, 2, 12))), {"energy": [1.0, 2.0]})


@ddt
class VdfCutsTestCase(unittest.TestCase):
    @staticmethod
    def _bimaxwellian(n_e=32, e_min=10.0, species="ions"):
        # Bi-Maxwellian in field-aligned coordinates (theta from B), n = 1
        # cm^-3, T_par = 2 keV, T_perp = 1 keV, 300 km/s along B
        mass = constants.m_p if species == "ions" else constants.m_e
        ratio = (3e4 / e_min) ** (1 / (n_e - 1))
        energy0 = e_min * ratio ** np.arange(n_e)
        phi = 5.625 + 11.25 * np.arange(32)
        theta = 5.625 + 11.25 * np.arange(16)
        speed = np.sqrt(2 * constants.e * energy0 / mass)[:, None, None]
        ph, th = np.deg2rad(phi)[:, None], np.deg2rad(theta)[None, :]
        v_perp2 = (speed * np.sin(th)) ** 2 * np.ones_like(ph)
        v_par = -speed * np.cos(th) * np.ones_like(ph)
        w_par = 2 * constants.e * 2000.0 / mass
        w_perp = 2 * constants.e * 1000.0 / mass
        f_si = np.exp(-v_perp2 / w_perp - (v_par - 3e5) ** 2 / w_par)
        f_si *= 1e6 / (np.pi**1.5 * w_perp * np.sqrt(w_par))

        energy = np.tile(energy0, (2, 1))
        vdf = pyrf.ts_skymap(
            generate_timeline(1 / 0.15, 2),
            np.tile(1e-12 * f_si, (2, 1, 1, 1)),
            energy,
            np.tile(phi, (2, 1)),
            theta,
            energy0=energy0,
            energy1=energy0,
            esteptable=np.zeros(2, dtype=np.uint8),
            attrs={"UNITS": "s^3/cm^6"},
            glob_attrs={
                "species": species,
                "delta_energy_plus": energy * (np.sqrt(ratio) - 1),
                "delta_energy_minus": energy * (1 - 1 / np.sqrt(ratio)),
            },
        )
        return vdf, f_si

    def test_vdf_cuts_distributions(self):
        vdf, f_si = self._bimaxwellian()
        dists, _ = mms.vdf_cuts(vdf)
        self.assertEqual(dists.f_par.attrs["UNITS"], "s^3/m^6")

        # Along B: the look directions at theta = 174.375 (pitch angle 5.625),
        # the same at all azimuths; against B: theta = 5.625
        np.testing.assert_allclose(
            dists.f_par.data[:, 32:], np.tile(f_si[:, 0, 15], (2, 1)), rtol=1e-12
        )
        np.testing.assert_allclose(
            dists.f_par.data[:, :32], np.tile(f_si[::-1, 0, 0], (2, 1)), rtol=1e-12
        )
        speed = np.sqrt(2 * constants.e * vdf.energy.data[0] / constants.m_p) / 1e3
        np.testing.assert_allclose(
            dists.v_par.data[0], np.hstack([-speed[::-1], speed])
        )

        # Perpendicular plane: theta = 84.375 and 95.625
        expected = (f_si[:, :, 7] + f_si[:, :, 8]) / 2
        np.testing.assert_allclose(dists.f_perp_2d.data[0], expected, rtol=1e-12)
        np.testing.assert_allclose(dists.f_perp.data[0], expected[:, 0], rtol=1e-12)
        np.testing.assert_allclose(dists.v_perp.data[0], speed)

        # Gyro-averaged: pitch angles of the particles, increasing
        np.testing.assert_allclose(dists.pitch_angle.data, 180.0 - vdf.theta.data[::-1])
        np.testing.assert_allclose(dists.f_pa.data[0], f_si[:, 0, ::-1], rtol=1e-12)

        # Wider windows: two bins in each cone
        dists, _ = mms.vdf_cuts(vdf, par_width=20.0, perp_width=20.0)
        expected = (f_si[:, 0, 14] + f_si[:, 0, 15]) / 2
        np.testing.assert_allclose(dists.f_par.data[0, 32:], expected, rtol=1e-12)
        self.assertEqual(dists.attrs["par_width"], 20.0)

    def test_vdf_cuts_moments(self):
        # A grid from 0.1 eV, so that the low speeds are in the cuts
        vdf, _ = self._bimaxwellian(n_e=64, e_min=0.1)
        _, moms = mms.vdf_cuts(vdf)
        self.assertEqual(moms.t.attrs["UNITS"], "eV")
        np.testing.assert_array_equal(moms.comp.data, ["perp1", "perp2", "par"])

        # Of the skymap: psd_moments
        np.testing.assert_allclose(moms.n.data, 1.0, rtol=0.01)
        np.testing.assert_allclose(moms.v.data[:, 2], 300.0, rtol=0.01)
        np.testing.assert_allclose(
            np.diag(moms.t.data[0]), [1000.0, 1000.0, 2000.0], rtol=0.01
        )

        # Of the cuts: T_par, T_perp and the drift
        np.testing.assert_allclose(moms.v_par_cut.data, 300.0, rtol=0.01)
        np.testing.assert_allclose(moms.t_par_cut.data, 2000.0, rtol=0.02)
        np.testing.assert_allclose(moms.v_perp_cut.data, 0.0, atol=1e-6)
        np.testing.assert_allclose(np.diag(moms.t_perp_cut.data[0]), 1000.0, rtol=0.03)
        self.assertTrue(np.all(moms.n_par_cut.data > 0))
        self.assertEqual(moms.n_perp_cut.attrs["UNITS"], "s/m^4")

    def test_vdf_cuts_input(self):
        # Any units, electrons, NaN bins
        vdf, _ = self._bimaxwellian(species="electrons")
        dpf = mms.vdf_convert(vdf, "1/(cm^2 s sr keV)")
        dists, moms = mms.vdf_cuts(dpf)
        expected, _ = mms.vdf_cuts(vdf)
        np.testing.assert_allclose(dists.f_par.data, expected.f_par.data, rtol=1e-10)

        vdf.data.data[:, :, :16, :] = np.nan
        dists, _ = mms.vdf_cuts(vdf)
        np.testing.assert_allclose(dists.f_par.data, expected.f_par.data, rtol=1e-12)

        with self.assertRaises(ValueError):
            mms.vdf_cuts(_pad_case(np.ones((3, 2, 12))))


@ddt
class TokenizeTestCase(unittest.TestCase):
    @data(*random.choices(_mms_keys(), k=10))
    def test_tokenize(self, var_str):
        result = mms.tokenize(var_str)
        self.assertIsInstance(result, dict)

    def test_tokenize_values(self):
        result = mms.tokenize("b_gse_fgm_brst_l2")
        self.assertEqual(result["inst"], "fgm")
        self.assertEqual(result["cdf_name"], "fgm_b_gse_brst_l2")

    @data("vel_gsm_mec_srvy_l2", "vel_gsm_mec_brst_l2")
    def test_tokenize_mec_velocity_gsm(self, var_str):
        # The GSM velocity keys read the GSE velocity (mec_v_gse)
        self.assertEqual(mms.tokenize(var_str)["cdf_name"], "mec_v_gsm")

    @data("brst", "fast", "slow")
    def test_tokenize_edp_probe_potentials(self, data_rate):
        # The 6 probe-to-spacecraft potentials (dcv) are in the scpot files,
        # as irfu-matlab mms.get_data V6_edp_*_l2
        result = mms.tokenize(f"v6_edp_{data_rate}_l2")
        self.assertEqual(result["cdf_name"], f"edp_dcv_{data_rate}_l2")
        self.assertEqual(result["dtype"], "scpot")

    @data("brst", "fast", "slow")
    def test_tokenize_edp_vpsp(self, data_rate):
        # The probe-to-spacecraft potential averaged over the probes (psp) is in
        # the scpot files, as irfu-matlab mms.get_data Vpsp_edp_*_l2 (its name
        # in any case)
        for var_str in [f"vpsp_edp_{data_rate}_l2", f"Vpsp_edp_{data_rate}_l2"]:
            result = mms.tokenize(var_str)
            self.assertEqual(result["cdf_name"], f"edp_psp_{data_rate}_l2")
            self.assertEqual(result["dtype"], "scpot")
            self.assertEqual(result["param"], "vpsp")

    @data("brst", "fast")
    def test_tokenize_fpi_part_index(self, data_rate):
        # The recommended energy index of the FPI partial moments (FPI's
        # moments are the partial moments from this channel)
        for species in ["i", "e"]:
            result = mms.tokenize(f"partidx{species}_fpi_{data_rate}_l2")
            self.assertEqual(result["cdf_name"], f"d{species}s_part_index_{data_rate}")
            self.assertEqual(result["dtype"], f"d{species}s-partmoms")
            self.assertEqual(result["param"], f"partidx{species}")

    @data("srvy", "brst")
    def test_tokenize_hpca_pressure(self, data_rate):
        # The pressure tensors of the HPCA moments files (DBCS only), as
        # irfu-matlab mms.get_data P*_dbcs_hpca_*_l2
        for species in ["hplus", "heplus", "heplusplus", "oplus"]:
            result = mms.tokenize(f"p{species}_dbcs_hpca_{data_rate}_l2")
            self.assertEqual(result["cdf_name"], f"hpca_{species}_ion_pressure")
            self.assertEqual(result["dtype"], "moments")

    @data(
        ("qe_gse_fpi_brst_l2", "des_heatq_gse_brst", "des-moms", "gse"),
        ("errqi_fpi_fast_l2", "dis_heatq_err_fast", "dis-moms", ""),
        ("flagi_fpi_fast_l2", "dis_errorflags_fast", "dis-moms", ""),
        ("padmidene_fpi_brst_l2", "des_pitchangdist_miden_brst", "des-moms", ""),
        ("tparae_fpi_fast_l2", "des_temppara_fast", "des-moms", ""),
        (
            "partpi_dbcs_fpi_brst_l2",
            "dis_prestensor_part_dbcs_brst",
            "dis-partmoms",
            "dbcs",
        ),
        ("qualitye_edp_fast_l2", "edp_quality_fast_l2", "dce", ""),
        ("qualityv_edp_brst_l2", "edp_quality_brst_l2", "scpot", ""),
        ("flag_fgm_srvy_l2", "fgm_flag_srvy_l2", "", ""),
        ("r_gsm_fgm_brst_l2", "fgm_r_gsm_brst_l2", "", "gsm"),
        ("quat_gse2000_mec_srvy_l2", "mec_quat_eci_to_gse2000", "epht89d", "gse2000"),
        ("ldip_mec_brst_l2", "mec_l_dipole", "epht89d", ""),
        ("tparaoplus_hpca_srvy_l2", "hpca_oplus_tparallel", "moments", ""),
        ("b_gsm_hpca_brst_l2", "hpca_B_GSM_spin_avg", "moments", "gsm"),
        ("epsdz_dsp_slow_l2", "dsp_epsd_z", "epsd", ""),
        ("bpsd_dsp_fast_l2", "dsp_bpsd_omni_fast_l2", "bpsd", ""),
        ("vdrift_gse_edi_srvy_l2", "edi_vdrift_gse_srvy_l2", "efield", "gse"),
        ("b_gse_scm_srvy_l2", "scm_acb_gse_scsrvy_srvy_l2", "scsrvy", "gse"),
        ("bschb_gse_scm_brst_l2", "scm_acb_gse_schb_brst_l2", "schb", "gse"),
    )
    @unpack
    def test_tokenize_cmad_keys(self, var_str, cdf_name, dtype, coords):
        # Variables documented in the CMAD and checked on files from AWS
        # (2017-07-11, MMS1): FPI heat flux, flags, electron PADs, extrapolated
        # densities; EDP quality and bitmask of the dce and scpot files; FGM
        # flag and position; MEC quaternions (ECI to the frame), MLT, L, Kp,
        # Dst; HPCA T_para, T_perp, B; DSP spectra; EDI E and drift velocity;
        # SCM survey and high-speed burst waveforms
        result = mms.tokenize(var_str)
        self.assertEqual(result["cdf_name"], cdf_name)
        self.assertEqual(result["dtype"], dtype)
        self.assertEqual(result["cs"], coords)

    def test_get_data_docstring_keys(self):
        # The keys listed in the get_data docstring are those of mms_keys.json
        # (it listed v_gse_mec_* instead of vel_gse_mec_*, and missed 30 keys)
        keys_path = os.path.join(os.path.dirname(mms.__file__), "mms_keys.json")
        with open(keys_path, "r", encoding="utf-8") as file:
            keys = {k for inst in json.load(file).values() for k in inst}

        get_data_module = importlib.import_module("pyrfu.mms.get_data")
        doc = get_data_module.get_data.__doc__.split("Parameters")[0]
        listed = set(
            re.findall(
                r"\b[a-z0-9+-]+_[a-z0-9_+-]*_(?:l2pre|l2a|l2|l3|ql|sitl|l1b)\b", doc
            )
        )
        self.assertSetEqual(listed, keys)

    def test_mms_keys_coordinate_systems(self):
        # When the CDF variable name has a coordinate system, it is the one of
        # the key (some products, e.g. HPCA DBCS moments, have none in the name)
        coordinates = {"gse", "gsm", "dbcs", "dmpa", "bcs", "dsl", "ssc", "gei"}
        keys_path = os.path.join(os.path.dirname(mms.__file__), "mms_keys.json")
        with open(keys_path, "r", encoding="utf-8") as file:
            keys = json.load(file)

        for instrument in keys.values():
            for key, value in instrument.items():
                key_coords = coordinates & set(key.lower().split("_"))
                cdf_coords = coordinates & set(value["cdf_name"].lower().split("_"))
                if cdf_coords:
                    with self.subTest(key=key):
                        self.assertSetEqual(cdf_coords, key_coords)

    @data(
        "es12_dsl_edp_fast_l2a",
        "adcoff_edp_slow_l2a",
        "phplus_gsm_hpca_brst_l2",
        "flux-amb-pm2_edi_brst_l2",
        "energye_fpi_fast_ql",
    )
    def test_tokenize_unsupported(self, var_str):
        # Valid parts, but not in mms_keys.json: a clear ValueError, not KeyError
        with self.assertRaisesRegex(ValueError, "not supported"):
            mms.tokenize(var_str)

    @data(
        "tsi_fpi_brst_l2",
        "tse_fpi_fast_l2",
        "epar_edp_brst_l2",
        "nbge_fpi_brst_l2",
        "pbge_fpi_fast_l2",
        "defbge_fpi_brst_l2",
    )
    def test_tokenize_removed_parameters(self, var_str):
        # Parameters without data (no FPI DES background, scalar temperatures
        # computed in irfu-matlab, epar the e_par keys): invalid parameters
        with self.assertRaisesRegex(ValueError, "invalid parameter"):
            mms.tokenize(var_str)

    @data(
        "B_gse_fgm_srvy_l2",
        "b_GSE_fgm_srvy_l2",
        "B_GSE_FGM_SRVY_L2",
        "Ti_gse_fpi_fast_l2",
        "V_edp_brst_l2",
    )
    def test_tokenize_case(self, var_str):
        # Any case, as the irfu-matlab names of the docstring examples (they
        # raised ValueError: invalid parameter)
        self.assertDictEqual(mms.tokenize(var_str), mms.tokenize(var_str.lower()))

    @data(
        ("b_fgm_srvy", "not a variable key"),
        ("b_gse_fgm_srvy_l2_extra", "not a variable key"),
        ("bx_gse_fgm_srvy_l2", "invalid parameter 'bx'"),
        ("b_xyz_fgm_srvy_l2", "invalid coord. sys. 'xyz'"),
        ("b_gse_fgx_srvy_l2", "invalid instrument 'fgx'"),
        ("b_gse_fgm_srvx_l2", "invalid sampling mode 'srvx'"),
        ("b_gse_fgm_srvy_l9", "invalid data level 'l9'"),
    )
    @unpack
    def test_tokenize_invalid(self, var_str, message):
        # ValueError naming the part (the number of parts and the parameter
        # raised AssertionError)
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            mms.tokenize(var_str)

    def test_tokenize_reads_keys_once(self):
        module = sys.modules["pyrfu.mms.tokenize"]
        module._mms_keys.cache_clear()

        with mock.patch("builtins.open", wraps=open) as opened:
            for _ in range(3):
                mms.tokenize("b_gse_fgm_srvy_l2")

        self.assertEqual(opened.call_count, 1)


@ddt
class ListFilesTestCase(unittest.TestCase):
    @data(*random.choices(_mms_keys(), k=10))
    def test_list_files(self, var_str):
        # Empty data directory, independent of the user's configuration
        with tempfile.TemporaryDirectory() as root:
            result = mms.list_files(
                TEST_TINT, random.randint(1, 4), mms.tokenize(var_str), root
            )

        self.assertListEqual(result, [])

    @staticmethod
    def _touch(root, rel_dir, names):
        path = os.path.join(root, *rel_dir.split("/"))
        os.makedirs(path, exist_ok=True)
        for name in names:
            with open(os.path.join(path, name), "wb"):
                pass

    @data("2019-09-14T07:54:00", "2019-09-14T00:00:00")
    def test_list_files_latest_version_srvy(self, t_start):
        # One file per day, several versions of one of them: only the latest
        # (the old code kept the last listed one, or all from midnight)
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fgm/srvy/l2/2019/09",
                [
                    "mms1_fgm_srvy_l2_20190913_v5.211.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.86.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.211.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.9.0.cdf",
                    "mms1_fgm_srvy_l2_20190915_v5.211.0.cdf",
                ],
            )
            var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
            tint = [t_start, "2019-09-14T08:11:00"]
            result = mms.list_files(tint, 1, var, root)

        self.assertListEqual(
            [os.path.basename(f) for f in result],
            ["mms1_fgm_srvy_l2_20190914_v5.211.0.cdf"],
        )

    def test_list_files_brst_segments(self):
        # Segments starting in the interval and the one covering its start,
        # latest versions only, in time order
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fpi/brst/l2/des-moms/2019/09/14",
                [
                    "mms1_fpi_brst_l2_des-moms_20190914074923_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075223_v3.3.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075223_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075523_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914080023_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914081223_v3.4.0.cdf",
                ],
            )
            var = {"inst": "fpi", "tmmode": "brst", "lev": "l2", "dtype": "des-moms"}
            tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
            result = mms.list_files(tint, 1, var, root)

        self.assertListEqual(
            [os.path.basename(f) for f in result],
            [
                "mms1_fpi_brst_l2_des-moms_20190914075223_v3.4.0.cdf",
                "mms1_fpi_brst_l2_des-moms_20190914075523_v3.4.0.cdf",
                "mms1_fpi_brst_l2_des-moms_20190914080023_v3.4.0.cdf",
            ],
        )

    def test_list_files_brst_across_midnight(self):
        # A segment starting at 23:59 the day before covers the start of the
        # interval: it is in the previous day directory, which was not listed
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fgm/brst/l2/2017/07/11",
                ["mms1_fgm_brst_l2_20170711235900_v5.87.0.cdf"],
            )
            self._touch(
                root,
                "mms1/fgm/brst/l2/2017/07/12",
                ["mms1_fgm_brst_l2_20170712000500_v5.87.0.cdf"],
            )
            var = {"inst": "fgm", "tmmode": "brst", "lev": "l2", "dtype": ""}
            result_start = mms.list_files(
                ["2017-07-12T00:00:30", "2017-07-12T00:01:00"], 1, var, root
            )
            result_both = mms.list_files(
                ["2017-07-12T00:00:30", "2017-07-12T00:06:00"], 1, var, root
            )

        self.assertListEqual(
            [os.path.basename(f) for f in result_start],
            ["mms1_fgm_brst_l2_20170711235900_v5.87.0.cdf"],
        )
        self.assertListEqual(
            [os.path.basename(f) for f in result_both],
            [
                "mms1_fgm_brst_l2_20170711235900_v5.87.0.cdf",
                "mms1_fgm_brst_l2_20170712000500_v5.87.0.cdf",
            ],
        )

    def test_list_files_month_boundary(self):
        # A file starting on the last evening of the previous month (in the
        # previous month directory) covers the start of the interval
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fpi/fast/l2/dis-moms/2017/06",
                ["mms1_fpi_fast_l2_dis-moms_20170630230000_v3.4.0.cdf"],
            )
            var = {"inst": "fpi", "tmmode": "fast", "lev": "l2", "dtype": "dis-moms"}
            tint = ["2017-07-01T00:10:00", "2017-07-01T00:20:00"]
            result = mms.list_files(tint, 1, var, root)

        self.assertListEqual(
            [os.path.basename(f) for f in result],
            ["mms1_fpi_fast_l2_dis-moms_20170630230000_v3.4.0.cdf"],
        )

    def test_list_files_walks_each_directory_once(self):
        # The month directory of survey data was walked for every day
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fgm/srvy/l2/2019/09",
                ["mms1_fgm_srvy_l2_20190912_v5.211.0.cdf"],
            )
            var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
            tint = ["2019-09-10T00:00:00", "2019-09-15T00:00:00"]

            with mock.patch("os.walk", wraps=os.walk) as walk:
                result = mms.list_files(tint, 1, var, root)

        self.assertEqual(walk.call_count, 1)
        self.assertListEqual(
            [os.path.basename(f) for f in result],
            ["mms1_fgm_srvy_l2_20190912_v5.211.0.cdf"],
        )


class _FakeSdcResponse:
    def __init__(self, files, error=None):
        self.files, self.error = files, error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return {"files": self.files}


class _FakeSdcSession:
    # Records the requests and answers with a canned SDC file listing
    def __init__(self, files, error=None):
        self.response = _FakeSdcResponse(files, error)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.response


def _sdc_query(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.path, {
        k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()
    }


def _sdc_science_file(file_name):
    return {"file_name": file_name, "timetag": "2019-09-15T00:00:00", "file_size": 1}


_SDC_URL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"


class ListFilesSdcTestCase(unittest.TestCase):
    # Offline: the SDC session is replaced by a fake one (the tests used to
    # query lasp.colorado.edu with the user's configuration and keyring)

    def _list(self, tint, mms_id, var_str, files, error=None):
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        session = _FakeSdcSession(files, error)
        with mock.patch.object(
            module, "_login_lasp", return_value=(session, {}, _SDC_URL)
        ):
            result = module.list_files_sdc(tint, mms_id, mms.tokenize(var_str))
        return result, session

    def test_list_files_sdc_request(self):
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        for var_str, descriptor in [
            ("b_gse_fgm_brst_l2", None),
            ("pdi_fpi_brst_l2", "dis-dist"),
        ]:
            _, session = self._list(tint, 2, var_str, [])
            path, query = _sdc_query(session.urls[0])
            self.assertTrue(path.endswith("file_info/science"))
            # The base URL ends with "/": it was doubled (".../v1//file_info")
            self.assertNotIn("//", path)
            self.assertEqual(query.get("descriptor"), descriptor)
            expected = {
                "start_date": "2019-09-14",
                "end_date": "2019-09-14-08-10-59",
                "sc_id": "mms2",
                "instrument_id": var_str.split("_")[-3],
                "data_rate_mode": "brst",
                "data_level": "l2",
            }
            self.assertDictEqual({k: query[k] for k in expected}, expected)

    def test_list_files_sdc_files(self):
        # From the file starting before the interval, sorted in time, with
        # their download URLs; other file names are ignored
        names = [
            f"mms2_fgm_brst_l2_20190914{t}_v5.215.0.cdf"
            for t in ["080023", "074523", "075523", "075223"]
        ]
        files = [_sdc_science_file(n) for n in names + ["readme.txt"]]
        result, _ = self._list(
            ["2019-09-14T07:54:00", "2019-09-14T08:11:00"],
            2,
            "b_gse_fgm_brst_l2",
            files,
        )

        expected = [names[3], names[2], names[0]]
        self.assertListEqual([file["file_name"] for file in result], expected)
        self.assertListEqual(
            [file["url"] for file in result],
            [f"{_SDC_URL}download/science?file={name}" for name in expected],
        )

    def test_list_files_sdc_http_error(self):
        with self.assertRaises(requests.exceptions.HTTPError):
            self._list(
                TEST_TINT,
                1,
                "b_gse_fgm_srvy_l2",
                [],
                requests.exceptions.HTTPError("500"),
            )


class ListFilesAncillarySdcTestCase(unittest.TestCase):
    # Offline, as ListFilesSdcTestCase

    def test_list_files_ancillary_sdc(self):
        module = importlib.import_module("pyrfu.mms.list_files_ancillary_sdc")
        name = "MMS3_DEFEPH_2019257_2019258.V01"
        session = _FakeSdcSession([{"file_name": name}])
        with mock.patch.object(
            module, "_login_lasp", return_value=(session, {}, _SDC_URL)
        ):
            result = module.list_files_ancillary_sdc(
                ["2019-09-14T07:54:00", "2019-09-14T08:11:00"], 3, "defeph"
            )

        path, query = _sdc_query(session.urls[0])
        self.assertTrue(path.endswith("file_info/ancillary"))
        # The base URL ends with "/": it was doubled (".../v1//file_info")
        self.assertNotIn("//", path)
        self.assertDictEqual(
            {k: query[k] for k in ["start_date", "end_date", "sc_id", "product"]},
            {
                "start_date": "2019-09-13",
                "end_date": "2019-09-15",
                "sc_id": "mms3",
                "product": "defeph",
            },
        )
        self.assertListEqual(
            result,
            [{"file_name": name, "url": f"{_SDC_URL}download/ancillary?file={name}"}],
        )

    def test_list_files_ancillary_sdc_defq(self):
        # One file for the four spacecraft: sc_id=mms (with mms3, no file)
        module = importlib.import_module("pyrfu.mms.list_files_ancillary_sdc")
        session = _FakeSdcSession([{"file_name": "MMS_DEFQ_2019257_2019258.V00"}])
        with mock.patch.object(
            module, "_login_lasp", return_value=(session, {}, _SDC_URL)
        ):
            module.list_files_ancillary_sdc(
                ["2019-09-14T07:54:00", "2019-09-14T08:11:00"], 3, "defq"
            )

        _, query = _sdc_query(session.urls[0])
        self.assertEqual(query["sc_id"], "mms")
        self.assertEqual(query["product"], "defq")


@pytest.mark.network
@unittest.skipUnless(
    os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
)
class SdcNetworkTestCase(unittest.TestCase):
    # Real queries to the public MMS SDC, with a temporary public
    # configuration (never the user's configuration or keyring)

    def setUp(self):
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        config_dir = tempfile.TemporaryDirectory()
        self.addCleanup(config_dir.cleanup)
        path = os.path.join(config_dir.name, "config.json")
        ConfigCacheTestCase._write_config(path, "sdc", rights="public")
        patch = mock.patch.object(module, "MMS_CFG_PATH", path)
        patch.start()
        self.addCleanup(patch.stop)
        module._login_lasp_cached.cache_clear()
        self.addCleanup(module._login_lasp_cached.cache_clear)

    def test_list_files_sdc_public(self):
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        for var_str in ["b_gse_fgm_srvy_l2", "pdi_fpi_brst_l2"]:
            result = mms.list_files_sdc(tint, 1, mms.tokenize(var_str))
            self.assertGreater(len(result), 0)
            self.assertTrue(all("url" in file for file in result))

        result = mms.list_files_ancillary_sdc(tint, 1, "defeph")
        self.assertGreater(len(result), 0)

    def test_load_ancillary_defq_public(self):
        # One file for the four spacecraft (MMS_DEFQ_*), every 30 s
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        result = mms.load_ancillary("defq", tint, 1, False, source="sdc")
        self.assertEqual(len(result.time), 34)
        self.assertTrue(np.all((result.q.data > 0) & (result.q.data <= 1)))


# DEFEPH layout (MMS1_DEFEPH_2019257_2019258.V01): 14 header lines, no footer
_DEFEPH_HEADER = [
    "Definitive Orbit Ephemeris",
    "Spacecraft = MMS1",
    "StartTime = 2019-257/00:05:52.907 UTC  (MMS TAI: 22536.004512812)",
    "StopTime = 2019-258/00:14:22.907 UTC  (MMS TAI: 22537.010415590)",
    "MMS TAI Reference Epoch = 1958-001/00:00:00 UTC",
    "CentralBody = Earth",
    "ReferenceFrame = Mean of J2000",
    "PrincipalPlane = Equatorial",
    "Project = DefEphemReFormat.m",
    "Source = FDGSS OPS",
    "FileCreationDate = Sep 16 2019 09:45:07.912 UTC",
    "",
    "Epoch (UTC)   Epoch MMS TAI   X   Y   Z   VX   VY   VZ   Mass",
    "                              Km  Km  Km  Km/Sec  Km/Sec  Km/Sec  Kg",
]

_DEFEPH_ROWS = [
    "2019-257/00:05:52.907   22536.004512812   -10539.488308885   2267.641774990"
    "   2762.979162558   -3.379702093   -7.150985755   2.233656130",
    "2019-257/00:06:22.907   22536.004860035   -10639.511569375   2052.828069429"
    "   2829.628149301   -3.288657503   -7.169561589   2.209589514",
    "2019-257/00:06:52.907   22536.005207257   -10736.816612409   1837.489793072"
    "   2895.553467478   -3.198497883   -7.185972813   2.185417405",
]

# DEFATT layout (MMS1_DEFATT_2019257_2019258.V00): 49 header lines (up to the
# COMMENT line with the column names) and a DATA_STOP footer
_DEFATT_ROWS = [
    "2019-257T00:00:00.012 1947110437.012  0.09800 -0.18260  0.80379  0.55765 "
    "-0.026  0.026 18.302 300.435 263.470  66.080 300.430 263.384  65.97 "
    "300.435 263.384 65.970 300.435 0.005  EKF",
    "2019-257T00:00:00.562 1947110437.562  0.08149 -0.19048  0.84953  0.48516 "
    "-0.025  0.027 18.301 310.481 263.431  66.086 310.476 263.386  65.97 "
    "310.481 263.386 65.970 310.481 0.005  EKF",
]


# DEFQ layout (MMS_DEFQ_2017187_2017188.V00, one file for the four
# spacecraft): 11 header lines, a maneuver flag column that is empty in the
# files checked, and "No data exists" for a day without data
# (MMS_DEFQ_2017275_2017276.V00)
_DEFQ_HEADER = [
    "Definitive Tetrahedron Quality Factor",
    "StartTime = 2017-187/00:05:53.151 UTC  (MMS TAI: 21736.004515637)",
    "StopTime = 2017-188/00:14:23.151 UTC  (MMS TAI: 21737.010418414)",
    "MMS TAI Reference Epoch = 1958-001/00:00:00 UTC",
    "Ephemeris Set = MMS1_DEFEPH_2017187_2017188.V01, "
    "MMS2_DEFEPH_2017187_2017188.V01,",
    "MMS3_DEFEPH_2017187_2017188.V01, MMS4_DEFEPH_2017187_2017188.V01",
    "Project = PPFA_ProductGeneration_QualityFactor.m",
    "Source = FDGSS OPS",
    "FileCreationDate = Jul 10 2017 13:26:50.074 UTC",
    " ",
    "Epoch (UTC)              Epoch MMS TAI     Q        Scale      Maneuver Flag",
]

_DEFQ_ROWS = [
    "2017-187/13:50:23.151    21736.577085081   0.927    20                      ",
    "2017-187/13:50:53.151    21736.577432303   0.927    20                      ",
    "2017-187/13:51:23.151    21736.577779525   0.927    20                      ",
]


class LoadAncillaryTestCase(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = root.name
        self.tint = ["2019-09-14T00:00:00", "2019-09-14T01:00:00"]

    def _write(self, product, file_name, lines, sc_name="mms1"):
        path = os.path.join(self.root, "ancillary", sc_name, product)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, file_name), "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    def test_load_ancillary_defeph(self):
        # Without footer, the last sample is data (it was dropped)
        self._write(
            "defeph",
            "MMS1_DEFEPH_2019257_2019258.V01",
            _DEFEPH_HEADER + _DEFEPH_ROWS,
        )
        result = mms.load_ancillary("defeph", self.tint, 1, False, self.root, "local")

        self.assertEqual(len(result.time), 3)
        self.assertEqual(
            result.time.data[-1], np.datetime64("2019-09-14T00:06:52.907", "ns")
        )
        np.testing.assert_allclose(
            result.x.data, [-10539.488308885, -10639.511569375, -10736.816612409]
        )

    def test_load_ancillary_defatt(self):
        # The DATA_STOP footer is not data
        header = ["META_START"] * 48 + ["COMMENT   Time (UTC)    Elapsed Sec"]
        self._write(
            "defatt",
            "MMS1_DEFATT_2019257_2019258.V00",
            header + _DEFATT_ROWS + ["DATA_STOP"],
        )
        result = mms.load_ancillary("defatt", self.tint, 1, False, self.root, "local")

        self.assertEqual(len(result.time), 2)
        np.testing.assert_allclose(result.z_dec.data, [66.080, 66.086])

    def test_load_ancillary_defatt_repeats_and_overlap(self):
        # DEFATT repeats times by design and consecutive files overlap: the
        # times are strictly increasing, with the last of the repeated rows of
        # a file and, in an overlap, the earlier file, as irfu-matlab
        # (mms_load_ancillary, mms_db)
        def row(time, z_dec):
            return (
                _DEFATT_ROWS[0]
                .replace("2019-257T00:00:00.012", time)
                .replace("66.080", z_dec)
            )

        header = ["META_START"] * 48 + ["COMMENT   Time (UTC)    Elapsed Sec"]
        self._write(
            "defatt",
            "MMS1_DEFATT_2019256_2019257.V00",
            header
            + [
                row("2019-256T23:59:59.000", "1.000"),
                row("2019-257T00:00:00.012", "1.500"),
                row("2019-257T00:00:00.012", "2.000"),
            ]
            + ["DATA_STOP"],
        )
        self._write(
            "defatt",
            "MMS1_DEFATT_2019257_2019258.V00",
            header
            + [
                row("2019-257T00:00:00.012", "3.000"),
                row("2019-257T00:00:00.562", "4.000"),
            ]
            + ["DATA_STOP"],
        )
        tint = ["2019-09-13T23:00:00", "2019-09-14T01:00:00"]
        result = mms.load_ancillary("defatt", tint, 1, False, self.root, "local")

        expected_time = np.array(
            [
                "2019-09-13T23:59:59.000",
                "2019-09-14T00:00:00.012",
                "2019-09-14T00:00:00.562",
            ],
            dtype="datetime64[ns]",
        )
        np.testing.assert_array_equal(result.time.data, expected_time)
        np.testing.assert_allclose(result.z_dec.data, [1.0, 2.0, 4.0])

    def test_load_ancillary_no_file(self):
        with self.assertRaises(FileNotFoundError):
            mms.load_ancillary("defeph", self.tint, 1, False, self.root, "local")

    def test_load_ancillary_defq(self):
        # One file for the four spacecraft (it was looked for as MMS1_DEFQ in
        # ancillary/mms1): the same for any mms_id; the maneuver flag is NaN,
        # or its value when set
        rows = _DEFQ_ROWS[:2] + [_DEFQ_ROWS[2].rstrip() + "     1"]
        self._write("defq", "MMS_DEFQ_2017187_2017188.V00", _DEFQ_HEADER + rows, "mms")
        tint = ["2017-07-06T13:50:00", "2017-07-06T13:55:00"]

        for mms_id in range(1, 5):
            result = mms.load_ancillary("defq", tint, mms_id, False, self.root, "local")

            np.testing.assert_allclose(result.q.data, 0.927)
            np.testing.assert_allclose(
                result.epoch_mms_tai.data[0], 21736.577085081, rtol=1e-15
            )
            np.testing.assert_array_equal(result.scale.data, 20)
            np.testing.assert_array_equal(
                result.maneuver_flag.data, [np.nan, np.nan, 1]
            )
            self.assertEqual(
                result.time.data[0], np.datetime64("2017-07-06T13:50:23.151", "ns")
            )

        with self.assertRaisesRegex(FileNotFoundError, "for MMS in"):
            mms.load_ancillary("defq", self.tint, 1, False, self.root, "local")

    def test_load_ancillary_defq_no_data(self):
        # A day without data: no row (the reader raised ValueError)
        self._write(
            "defq",
            "MMS_DEFQ_2017275_2017276.V00",
            _DEFQ_HEADER + ["No data exists"],
            "mms",
        )
        tint = ["2017-10-02T12:00:00", "2017-10-02T12:05:00"]
        result = mms.load_ancillary("defq", tint, 1, False, self.root, "local")

        self.assertEqual(len(result.time), 0)
        self.assertListEqual(
            list(result.data_vars), ["epoch_mms_tai", "q", "scale", "maneuver_flag"]
        )

    def test_load_ancillary_unsupported(self):
        for product in ["predatt", "predeph", "bazinga"]:
            with self.assertRaisesRegex(ValueError, "not supported"):
                mms.load_ancillary(product, self.tint, 1, False, self.root, "local")

    # From the SDC (mocked listing and session)
    _DEFATT_HEADER = ["META_START"] * 48 + ["COMMENT   Time (UTC)    Elapsed Sec"]

    @staticmethod
    def _sdc_file(name, start, end):
        return {
            "file_name": name,
            "url": f"url/{name}",
            "start_date": start,
            "end_date": end,
        }

    def _sdc(self, files, contents, status_error=None):
        module = importlib.import_module("pyrfu.mms.load_ancillary")
        responses = {}

        for file, content in zip(files, contents):
            response = mock.MagicMock(content=content.encode())
            if status_error:
                response.raise_for_status.side_effect = status_error
            responses[file["url"]] = response

        session = mock.MagicMock()
        session.get.side_effect = lambda url, **kwargs: responses[url]

        for name, value in [
            ("list_files_ancillary_sdc", files),
            ("_login_lasp", (session, {}, "")),
        ]:
            patch = mock.patch.object(module, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

        return session

    def _defatt_files(self):
        # The day before (no row in tint), the day of tint, the day after
        day = "\n".join(self._DEFATT_HEADER + _DEFATT_ROWS + ["DATA_STOP"]) + "\n"
        before = day.replace("2019-257T", "2019-256T")
        files = [
            self._sdc_file(
                "MMS1_DEFATT_2019256_2019257.V00",
                "2019-09-13T00:00:00",
                "2019-09-14T00:00:00",
            ),
            self._sdc_file(
                "MMS1_DEFATT_2019257_2019258.V00",
                "2019-09-14T00:00:00",
                "2019-09-15T00:00:00",
            ),
            self._sdc_file(
                "MMS1_DEFATT_2019258_2019259.V00",
                "2019-09-15T00:00:00",
                "2019-09-16T00:00:00",
            ),
        ]
        return files, [before, day, day]

    def test_load_ancillary_sdc(self):
        # Read in memory from the SDC: as the same local file, and only the
        # files covering tint (here near midnight: also the day before)
        files, contents = self._defatt_files()
        session = self._sdc(files, contents)
        self._write("defatt", files[1]["file_name"], contents[1].splitlines())
        local = mms.load_ancillary("defatt", self.tint, 1, False, self.root, "local")

        for source in ["sdc", "aws"]:
            session.get.reset_mock()

            with self.assertLogs("pyrfu.mms.load_ancillary", "INFO") as logs:
                result = mms.load_ancillary("defatt", self.tint, 1, source=source)

            self.assertEqual(
                any("No ancillary files on AWS" in line for line in logs.output),
                source == "aws",
            )

            xr.testing.assert_identical(result, local)
            self.assertListEqual(
                [call.args[0] for call in session.get.call_args_list],
                [files[0]["url"], files[1]["url"]],
            )

    def test_load_ancillary_sdc_files(self):
        # The files covering tint (with a day on their end, as
        # list_files_ancillary: the names give dates only), the latest version
        # of each, in time order
        module = importlib.import_module("pyrfu.mms.load_ancillary")
        files, contents = self._defatt_files()
        newer = dict(files[1], file_name=files[1]["file_name"].replace("V00", "V01"))
        self._sdc([files[2], newer, files[1], files[0]], [contents[2]] * 4)

        for tint, expected in [
            (["2019-09-14T12:00:00", "2019-09-14T13:00:00"], [files[0], newer]),
            (["2019-09-14T00:10:00", "2019-09-14T00:20:00"], [files[0], newer]),
            (
                ["2019-09-14T23:30:00", "2019-09-15T00:30:00"],
                [files[0], newer, files[2]],
            ),
        ]:
            self.assertListEqual(module._list_files_sdc(tint, 1, "defatt"), expected)

    def test_load_ancillary_sdc_file_from_the_day_before(self):
        # Early DEFATT files cover about 16:00 to 16:00 (MMS1 2015361_2015362:
        # 2015-361T15:56 to 362T15:54): the morning of a day is in the file of
        # the day before (it was not read: no row)
        def content(times):
            rows = [
                _DEFATT_ROWS[0].replace("2019-257T00:00:00.012", time) for time in times
            ]
            return "\n".join(self._DEFATT_HEADER + rows + ["DATA_STOP"]) + "\n"

        files = [
            self._sdc_file(
                "MMS1_DEFATT_2019256_2019257.V00",
                "2019-09-13T00:00:00",
                "2019-09-14T00:00:00",
            ),
            self._sdc_file(
                "MMS1_DEFATT_2019257_2019258.V00",
                "2019-09-14T00:00:00",
                "2019-09-15T00:00:00",
            ),
        ]
        before = [
            "2019-256T15:56:14.012",
            "2019-257T03:57:10.012",
            "2019-257T03:57:20.012",
        ]
        self._sdc(files, [content(before), content(["2019-257T15:49:23.062"])])
        tint = ["2019-09-14T03:57:00", "2019-09-14T03:58:00"]
        result = mms.load_ancillary("defatt", tint, 1, False, source="sdc")

        np.testing.assert_array_equal(
            result.time.data,
            np.array(
                ["2019-09-14T03:57:10.012", "2019-09-14T03:57:20.012"],
                dtype="datetime64[ns]",
            ),
        )

    def test_load_ancillary_default_source(self):
        # The default resource of the configuration (a temporary one, also for
        # the local path of the local lister)
        db_get_ts = importlib.import_module("pyrfu.mms.db_get_ts")
        lister = importlib.import_module("pyrfu.mms.list_files_ancillary")
        files, contents = self._defatt_files()
        session = self._sdc(files, contents)
        self._write("defatt", files[1]["file_name"], contents[1].splitlines())

        for default, n_get in [("sdc", 2), ("aws", 2), ("local", 0)]:
            path = os.path.join(self.root, f"config_{default}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"default": default, "local": self.root}, file)

            db_get_ts._load_config_cached.cache_clear()
            session.get.reset_mock()
            with (
                mock.patch.object(db_get_ts, "MMS_CFG_PATH", path),
                mock.patch.object(lister, "MMS_CFG_PATH", path),
            ):
                result = mms.load_ancillary("defatt", self.tint, 1, False)
            db_get_ts._load_config_cached.cache_clear()

            self.assertEqual(len(result.time), 2)
            self.assertEqual(session.get.call_count, n_get)

    def test_load_ancillary_sdc_errors(self):
        files, contents = self._defatt_files()
        self._sdc(files, contents, status_error=requests.HTTPError("404"))

        with self.assertRaises(requests.HTTPError):
            mms.load_ancillary("defatt", self.tint, 1, False, source="sdc")

        self._sdc([], [])

        with self.assertRaises(FileNotFoundError):
            mms.load_ancillary("defatt", self.tint, 1, False, source="sdc")


@ddt
class ListFilesAncillaryTestCase(unittest.TestCase):
    @data("predatt", "predeph", "predq", "defatt", "defeph", "defq")
    def test_list_files_ancillary(self, product):
        # Empty data directory, independent of the user's configuration
        with tempfile.TemporaryDirectory() as root:
            result = mms.list_files_ancillary(
                TEST_TINT, random.randint(1, 4), product, root
            )

        self.assertListEqual(result, [])


@ddt
class VdfProjectionTestCase(unittest.TestCase):
    def test_vdf_projection_output(self):
        result = mms.vdf_projection(
            generate_vdf(64.0, 42, [32, 32, 16], energy01=True), TEST_TINT
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)

    @staticmethod
    def _vdf():
        # Alternating energy tables (positive energies) and a phi table that
        # changes with time, so that each sample's phi is identifiable
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True)
        energy0 = 10.0 * 1.3 ** np.arange(32)
        energy1 = energy0 * np.sqrt(1.3)
        vdf.attrs["energy0"], vdf.attrs["energy1"] = energy0, energy1
        vdf["energy"] = (
            ("time", "idx0"),
            np.where(vdf.attrs["esteptable"][:, None] == 1, energy1, energy0),
        )
        vdf["phi"] = (
            ("time", "idx1"),
            (np.arange(32) * 11.25)[None, :] + 0.1 * np.arange(42)[:, None],
        )
        return vdf

    def test_vdf_projection_phi_clipped(self):
        # With a tint that doesn't start at the first sample, psd_rebin used to
        # get the phi of the first samples instead of those in tint
        vdf = self._vdf()
        tint = list(pyrf.datetime642iso8601(vdf.time.data[[10, 21]]))
        module = importlib.import_module("pyrfu.mms.vdf_projection")
        with mock.patch.object(
            module, "psd_rebin", wraps=module.psd_rebin
        ) as psd_rebin:
            module._init(vdf, tint)

        expected = pyrf.time_clip(vdf.phi, tint).data
        np.testing.assert_allclose(psd_rebin.call_args.args[1], expected)

    @data(0, 1)
    def test_vdf_projection_single_time_energy_table(self, step):
        # A single time used the energy1 edges for both step table values
        vdf = self._vdf()
        t_id = int(np.flatnonzero(vdf.attrs["esteptable"] == step)[0])
        tint = [pyrf.datetime642iso8601(vdf.time.data[t_id])]
        energy_edges = importlib.import_module("pyrfu.mms.vdf_projection")._init(
            vdf, tint
        )[3]

        energy = vdf.attrs[f"energy{step}"]
        # Log-centred edges: the geometric mean of consecutive edges is the energy
        np.testing.assert_allclose(
            np.sqrt(energy_edges[:-1] * energy_edges[1:]), energy, rtol=1e-6
        )

    @staticmethod
    def _maxwellian(v_kms, n_t=2, alternating=False):
        # 100 eV proton Maxwellian on FPI-like bins
        energy1 = None
        if alternating:
            energy1 = 10.0 * 3000.0 ** ((np.arange(32) + 0.5) / 31)

        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(
            1.0, 100.0, v_kms, n_t=n_t, energy1=energy1
        )
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        return vdf

    @staticmethod
    def _peak(v_x, v_y, f_mat):
        # Speed (km/s) and angle (deg) of the centre of the maximum bin
        v_x_c = (v_x[:-1, :-1] + v_x[1:, :-1] + v_x[:-1, 1:] + v_x[1:, 1:]) / 4
        v_y_c = (v_y[:-1, :-1] + v_y[1:, :-1] + v_y[:-1, 1:] + v_y[1:, 1:]) / 4
        idx = np.unravel_index(np.nanargmax(f_mat.T), f_mat.T.shape)
        speed = np.hypot(v_x_c[idx], v_y_c[idx])
        return speed, np.rad2deg(np.arctan2(v_y_c[idx], v_x_c[idx]))

    @data(
        # (in-plane angle of the drift, frame with x, y, z as rows)
        (5.625, np.eye(3)),
        (5.625, np.vstack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
        (95.625, np.vstack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
    )
    @unpack
    def test_vdf_projection_peak(self, angle, coord_sys):
        # 400 km/s drift in the (x, y) plane of coord_sys, at an azimuthal bin
        # centre: the projection peaks there. An orthonormal frame gives no
        # warning (the check compared rows to columns and always warned).
        coord_sys = coord_sys.astype(float)
        rad = np.deg2rad(angle)
        v_d = 400.0 * (np.cos(rad) * coord_sys[0] + np.sin(rad) * coord_sys[1])
        vdf = self._maxwellian(list(v_d))
        tint = [pyrf.datetime642iso8601(vdf.time.data[0])]

        with self.assertNoLogs(level="WARNING"):
            result = mms.vdf_projection(vdf, tint, coord_sys)

        speed, peak_angle = self._peak(*result)
        self.assertAlmostEqual(peak_angle, angle, delta=0.1)
        self.assertAlmostEqual(speed, 400.0, delta=30.0)

    @data(False, True)
    def test_vdf_projection_interval(self, alternating):
        # Intervals with one energy table used to crash (.data on an array);
        # with alternating tables (rebinned to 64 energies) phi was used in
        # degrees as radians, so the peak was misplaced
        rot = np.deg2rad(5.625)
        vdf = self._maxwellian(
            [400.0 * np.cos(rot), 400.0 * np.sin(rot), 0.0], 20, alternating
        )
        tint = list(pyrf.datetime642iso8601(vdf.time.data[[0, -1]]))
        v_x, v_y, f_mat = mms.vdf_projection(vdf, tint)

        self.assertEqual(f_mat.shape[1], 64 if alternating else 32)
        speed, peak_angle = self._peak(v_x, v_y, f_mat)
        self.assertAlmostEqual(peak_angle, 5.625, delta=0.1)
        self.assertAlmostEqual(speed, 400.0, delta=30.0)

    def test_vdf_projection_sc_pot(self):
        # Energy edges below the spacecraft potential gave NaN speeds
        vdf = self._maxwellian([400.0, 0.0, 0.0])
        vdf.attrs["species"] = "electrons"
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.full(2, 20.0))
        tint = [pyrf.datetime642iso8601(vdf.time.data[0])]
        v_x, v_y, _ = mms.vdf_projection(vdf, tint, sc_pot=sc_pot)

        self.assertTrue(np.isfinite(v_x).all() and np.isfinite(v_y).all())
        self.assertEqual(np.hypot(v_x, v_y)[0].max(), 0.0)


@ddt
class FftBandpassTestCase(unittest.TestCase):
    @data(
        (generate_data(100, tensor_order=0), 0.1, 1.0),
        (generate_ts(64.0, 100, tensor_order=0), "bazinga!!", 1.0),
        (generate_ts(64.0, 100, tensor_order=0), 0.1, "bazinga!!"),
    )
    @unpack
    def test_fft_bandpass_input_type(self, inp, f_min, f_max):
        with self.assertRaises(TypeError):
            mms.fft_bandpass(inp, f_min, f_max)

    @data(
        (generate_ts(64.0, 100, tensor_order=0), 1.0, 0.1),
        (generate_ts(64.0, 100, tensor_order=2), 0.1, 1.0),
    )
    @unpack
    def test_fft_bandpass_input_value(self, inp, f_min, f_max):
        with self.assertRaises(ValueError):
            mms.fft_bandpass(inp, f_min, f_max)

    @data(
        (generate_ts(64.0, 101, tensor_order=0), random.random(), random.random()),
        (generate_ts(64.0, 101, tensor_order=1), random.random(), random.random()),
    )
    @unpack
    def test_fft_bandpass_output(self, inp, f_min, f_max):
        result = mms.fft_bandpass(inp, *sorted([f_min, f_max]))
        self.assertIsInstance(result, xr.DataArray)

    @data(0, 1)
    def test_fft_bandpass_nan_input_unchanged(self, tensor_order):
        # NaNs are set to 0 for the FFT in a copy: the caller's data keep them, and
        # the output is NaN there. 5 and 20 Hz sines, band 10-30 Hz.
        times = generate_timeline(128.0, 1024)
        t_s = np.arange(1024) / 128.0
        sig = np.sin(2 * np.pi * 5 * t_s) + np.sin(2 * np.pi * 20 * t_s)
        if tensor_order:
            inp = pyrf.ts_vec_xyz(times, np.tile(sig[:, None], (1, 3)))
        else:
            inp = pyrf.ts_scalar(times, sig)
        inp.data[100] = np.nan
        inp_data = inp.data.copy()

        out = mms.fft_bandpass(inp, 10.0, 30.0)

        np.testing.assert_array_equal(inp.data, inp_data)
        self.assertTrue(np.all(np.isnan(out.data[100])))
        expected = np.sin(2 * np.pi * 20 * t_s)
        if tensor_order:
            expected = np.tile(expected[:, None], (1, 3))
        np.testing.assert_allclose(out.data[300:700], expected[300:700], atol=0.05)

    def test_fft_bandpass_frequency_grid(self):
        # The frequencies of the FFT bins, as irfu-matlab: a 10 Hz sine (a bin
        # of 1024 samples at 128 Hz) is kept whole by a band starting at 10 Hz
        # (it was halved: its negative frequency was taken for 9.95 Hz), and
        # integer frequencies are accepted (they raised TypeError)
        times = generate_timeline(128.0, 1024)
        sig = np.sin(2 * np.pi * 10 * np.arange(1024) / 128.0)
        inp = pyrf.ts_scalar(times, sig)

        np.testing.assert_allclose(mms.fft_bandpass(inp, 10, 30).data, sig, atol=1e-9)
        np.testing.assert_allclose(mms.fft_bandpass(inp, 1, 9).data, 0.0, atol=1e-9)

        with self.assertRaises(TypeError):
            mms.fft_bandpass(inp, True, 30.0)


class HpcaSpinSumTestCase(unittest.TestCase):
    @staticmethod
    def _inputs(start_az):
        # 16 azimuths per spin; the data (time, polar, energy)
        time = generate_timeline(16.0, len(start_az))
        inp = xr.DataArray(
            np.random.random((len(start_az), 2, 3)),
            coords=[time, np.arange(2), np.arange(3)],
            dims=["time", "polar", "energy"],
            attrs={"UNITS": "cm^-2 s^-1 sr^-1 eV^-1"},
        )
        return inp, pyrf.ts_scalar(time, np.asarray(start_az, dtype=float))

    def test_hpca_spin_sum(self):
        # Spins from one start to the next; the samples after the last start
        # are not used (as pyspedas)
        inp, saz = self._inputs(np.arange(50) % 16)

        for method in ["mean", "sum"]:
            result = mms.hpca_spin_sum(inp, saz, method)
            expected = [
                getattr(inp.data[k : k + 16], method)(axis=0) for k in [0, 16, 32]
            ]

            np.testing.assert_allclose(result.data, expected)
            np.testing.assert_array_equal(result.time.data, saz.time.data[[0, 16, 32]])
            self.assertEqual(result.dims, inp.dims)
            self.assertDictEqual(result.attrs, inp.attrs)

    def test_hpca_spin_sum_errors(self):
        # One spin start (it raised IndexError) or none: no complete spin
        for start_az in [np.arange(20) % 32, np.arange(1, 20)]:
            with self.assertRaisesRegex(ValueError, "two spin starts"):
                mms.hpca_spin_sum(*self._inputs(start_az))

        with self.assertRaisesRegex(ValueError, "Invalid method"):
            mms.hpca_spin_sum(*self._inputs(np.arange(50) % 16), method="median")


class WhistlerB2ETestCase(unittest.TestCase):
    b_mag, n_e = 50.0, 5.0  # nT, cm^-3
    f_ce = constants.e * 1e-9 * b_mag / constants.m_e / (2 * np.pi)
    f_pe = np.sqrt(1e6 * n_e * constants.e**2 / (constants.m_e * constants.epsilon_0))
    f_pe /= 2 * np.pi

    def test_whistler_b2e_parallel(self):
        # theta_k = 0: n^2 = R and E^2 = (c^2 / R) B^2, with 1e-12 from nT^2 to
        # T^2 (1e-18) and from (V/m)^2 to (mV/m)^2 (1e6)
        freq = np.array([0.1, 0.3, 0.5]) * self.f_ce
        b2 = np.array([1.0, 2.0, 3.0])
        result = mms.whistler_b2e(b2, freq, 0.0, self.b_mag, self.n_e)

        r_cold = 1 - self.f_pe**2 / (freq * (freq - self.f_ce))
        expected = constants.c**2 / r_cold * b2 * 1e-12
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_whistler_b2e_spectrogram(self):
        # A spectrogram gives, at each time, the spectrum for the local B and n
        time = generate_timeline(1.0, 3)
        freq = np.array([0.1, 0.3, 0.5]) * self.f_ce
        b2_data = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
        b2 = xr.DataArray(b2_data, coords=[time, freq], dims=["time", "frequency"])
        b_mag = pyrf.ts_scalar(time, np.array([50.0, 60.0, 70.0]))
        n_e = pyrf.ts_scalar(time, np.array([5.0, 4.0, 6.0]))
        theta_k = np.deg2rad(30.0)

        result = mms.whistler_b2e(b2, freq, theta_k, b_mag, n_e)

        self.assertIsInstance(result, xr.DataArray)
        self.assertTupleEqual(result.dims, ("time", "frequency"))
        for i in range(3):
            expected = mms.whistler_b2e(
                b2_data[i], freq, theta_k, b_mag.data[i], n_e.data[i]
            )
            np.testing.assert_allclose(result.data[i], expected, rtol=1e-12)

    def test_whistler_b2e_length(self):
        with self.assertRaises(IndexError):
            mms.whistler_b2e(np.ones(3), np.ones(4), 0.0, 50.0, 5.0)


if __name__ == "__main__":
    unittest.main()
