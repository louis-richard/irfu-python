#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, unpack

# Local imports
from .. import dispersion
from ..dispersion.disp_surf_calc import _calc_b, _calc_diel, _calc_e, _calc_vei
from ..dispersion.one_fluid_dispersion import _disprel

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
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

    def test_disp_surf_calc_group_velocity(self):
        kx, kz, wf, extra_param = dispersion.disp_surf_calc(20.0, 20.0, 1836.0, 2.0)
        v_g = extra_param["v_g"]

        # Forward differences on the (kx, kz) grid, as irfu-matlab
        d_k = 20.0 / 34
        v_x = (wf[9, 18, 17] - wf[9, 17, 17]) / d_k
        v_z = (wf[9, 17, 18] - wf[9, 17, 17]) / d_k
        self.assertAlmostEqual(v_g[9, 17, 17], np.hypot(v_x, v_z), delta=1e-5)

        # Light wave branch: v_g = k c^2 / w (k / w in normalized units)
        for i, j in [(17, 17), (30, 30), (30, 5), (5, 30)]:
            k_c = np.hypot(kx[i, j], kz[i, j])
            self.assertAlmostEqual(v_g[9, i, j] / (k_c / wf[9, i, j]), 1.0, delta=0.02)

    def test_disp_surf_calc_particle_field_energy(self):
        kc_max, m_i, wp_e = [1.0, 25.0, 2.0]
        kx, kz, wf, extra_param = dispersion.disp_surf_calc(kc_max, kc_max, m_i, wp_e)

        # log10 of the ratio of the particle (electrons + ions) energy density
        # to the field energy density
        w_f, kc_x, kc_z = [np.transpose(wf, [0, 2, 1]), kx.T, kz.T]
        kc_, theta = [np.hypot(kc_x, kc_z), np.arctan2(kc_x, kc_z)]
        diel = _calc_diel(kc_, w_f, theta, wp_e, wp_e / np.sqrt(m_i), 1 / m_i)
        e_x, e_y, e_z, _, _, e_tot = _calc_e(diel)
        b_tot = _calc_b(kc_x, kc_z, w_f, e_x, e_y, e_z)[-1]
        v_ex, v_ey, v_ez, v_ix, v_iy, v_iz = _calc_vei(m_i, 1 / m_i, w_f, e_x, e_y, e_z)
        en_e = 0.5 * (np.abs(v_ex) ** 2 + np.abs(v_ey) ** 2 + np.abs(v_ez) ** 2)
        en_i = 0.5 * m_i * (np.abs(v_ix) ** 2 + np.abs(v_iy) ** 2 + np.abs(v_iz) ** 2)
        en_field = 0.5 * (np.abs(e_tot) ** 2 + np.abs(b_tot) ** 2)
        ref = np.log10((en_e + en_i) * wp_e**2 / en_field)
        ref = np.transpose(ref, [0, 2, 1])

        np.testing.assert_allclose(
            extra_param["E_part/E_field"][wf > 0], ref[wf > 0], atol=1e-10
        )

    def test_disp_surf_calc_extra_param_keys(self):
        m_i, wp_e = [25.0, 2.0]
        kx, kz, wf, extra_param = dispersion.disp_surf_calc(1.0, 1.0, m_i, wp_e)
        pos = wf > 0

        # (dn_e/n)/(dB/B) - (dn_i/n)/(dB/B) = log10(dn_e/dn_i)
        diff_dn = extra_param["(dn_e/n)/(dB/B)"] - extra_param["(dn_i/n)/(dB/B)"]
        np.testing.assert_allclose(
            diff_dn[pos], extra_param["dn_e/dn_i"][pos], atol=1e-10
        )

        # (W_e + W_i) / W_field = W_e / W_field * (1 + W_i / W_e)
        ratio_part = 10 ** extra_param["W_e/W_field"]
        ratio_part *= 1 + 10 ** -extra_param["E_e/E_i"]
        np.testing.assert_allclose(
            ratio_part[pos], 10 ** extra_param["E_part/E_field"][pos], rtol=1e-10
        )

        # dn_e / (k . E eps0 / e) = (dn_e/n) / (dB/B) * wp_e^2 * dB / |k . E|
        w_f, kc_x, kc_z = [np.transpose(wf, [0, 2, 1]), kx.T, kz.T]
        kc_, theta = [np.hypot(kc_x, kc_z), np.arctan2(kc_x, kc_z)]
        diel = _calc_diel(kc_, w_f, theta, wp_e, wp_e / np.sqrt(m_i), 1 / m_i)
        e_x, e_y, e_z = _calc_e(diel)[:3]
        b_tot = np.real(_calc_b(kc_x, kc_z, w_f, e_x, e_y, e_z)[-1])
        k_dot_e = np.abs(kc_x * e_x + kc_z * e_z)
        ref = extra_param["(dn_e/n)/(dB/B)"]
        ref = ref + np.log10(np.transpose(wp_e**2 * b_tot / k_dot_e, [0, 2, 1]))
        np.testing.assert_allclose(
            extra_param["dn_e/(k E eps0/e)"][pos], ref[pos], atol=1e-10
        )

    def test_disp_surf_calc_extra_param_old_keys(self):
        _, _, _, extra_param = dispersion.disp_surf_calc(1.0, 1.0, 25.0, 2.0)
        for old_key, new_key in [
            ("(dn_e/n)/ (dB/B)", "(dn_e/n)/(dB/B)"),
            ("(dn_e/n)/(dBpar /B)", "(dn_e/n)/(dBpar/B)"),
            (" Spar/Stot", "Spar/Stot"),
        ]:
            self.assertNotIn(old_key, list(extra_param))
            with self.assertWarns(FutureWarning):
                self.assertIs(extra_param[old_key], extra_param[new_key])
            with self.assertWarns(FutureWarning):
                self.assertIs(extra_param.get(old_key), extra_param[new_key])
            with self.assertWarns(FutureWarning):
                self.assertIn(old_key, extra_param)


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

    @data(
        ({"n": 10e6, "t": 10.0, "gamma": 1.0}, {"n": 10e6, "t": 10.0, "gamma": 1.0}),
        (
            {"n": 10e6, "t": 1e3, "gamma": 5 / 3},
            {"n": 10e6, "t": 200.0, "gamma": 5 / 3},
        ),
    )
    @unpack
    def test_one_fluid_dispersion_branches(self, ions, electrons):
        for theta in [5.0, 30.0, 55.0, 85.0]:
            wc_1, wc_2, wc_3 = dispersion.one_fluid_dispersion(
                10e-9, theta, ions, electrons
            )
            k, (v_a, c_s, wc_e, wc_p) = [
                wc_1.k.data,
                [wc_1.attrs[key] for key in ["v_a", "c_s", "wc_e", "wc_p"]],
            ]

            # Three distinct branches, sorted
            self.assertTrue(np.all(wc_1.data > wc_2.data))
            self.assertTrue(np.all(wc_2.data > wc_3.data))

            # Roots of the dispersion relation (relative to its terms)
            for w_c in [wc_1.data, wc_2.data, wc_3.data]:
                res = _disprel(w_c, k, theta, v_a, c_s, wc_e, wc_p)
                scale = (1 + w_c**2 / (k * v_a) ** 2 + w_c**2 / (wc_e * wc_p)) ** 2
                np.testing.assert_array_less(np.abs(res) / scale, 1e-9)

            # Fast, Alfven and slow ideal MHD phase speeds at small k
            cos2 = np.cos(np.deg2rad(theta)) ** 2
            v_2, d_v2 = [v_a**2 + c_s**2, 4 * v_a**2 * c_s**2 * cos2]
            v_mhd = [
                np.sqrt((v_2 + np.sqrt(v_2**2 - d_v2)) / 2),
                v_a * np.sqrt(cos2),
                np.sqrt((v_2 - np.sqrt(v_2**2 - d_v2)) / 2),
            ]
            for w_c, v_ph in zip([wc_1, wc_2, wc_3], v_mhd):
                self.assertAlmostEqual(float(w_c[0]) / k[0] / v_ph, 1.0, delta=1e-2)

    @data(
        (10e-9, {"n": 10e6, "t": 10.0, "gamma": 1.0}),
        (50e-9, {"n": 1e6, "t": 100.0, "gamma": 5 / 3}),
    )
    @unpack
    def test_one_fluid_dispersion_k_range(self, b_0, ions):
        # Default range in units of the ion inertial length V_A / wc_p
        wc_1 = dispersion.one_fluid_dispersion(b_0, 30.0, ions, ions, 50)[0]
        k_vec, v_a, wc_p = [wc_1.k.data, wc_1.attrs["v_a"], wc_1.attrs["wc_p"]]
        self.assertEqual(len(k_vec), 50)
        self.assertAlmostEqual(k_vec[0] * v_a / wc_p, 0.0144, places=12)
        self.assertAlmostEqual(k_vec[-1] * v_a / wc_p, 7.2, places=12)

        # Given wavenumbers, same roots as the default range
        result = dispersion.one_fluid_dispersion(
            b_0, 30.0, ions, ions, k_vec=k_vec[[0, 10, -1]]
        )
        np.testing.assert_array_equal(result[0].k.data, k_vec[[0, 10, -1]])
        np.testing.assert_allclose(result[0].data, wc_1.data[[0, 10, -1]])

    def test_one_fluid_dispersion_k_vec_notebook(self):
        # The default range is 2e-7 to 1e-4 m^-1 for 10 nT and 10 cm^-3
        ions = {"n": 10e6, "t": 10.0, "gamma": 1.0}
        wc_1 = dispersion.one_fluid_dispersion(10e-9, 30.0, ions, ions)[0]
        np.testing.assert_allclose(wc_1.k.data[[0, -1]], [2e-7, 1e-4], rtol=2e-4)

    @data([0.0, 1e-5], [-1e-5], 0.0)
    def test_one_fluid_dispersion_k_vec_input(self, k_vec):
        ions = {"n": 10e6, "t": 10.0, "gamma": 1.0}
        with self.assertRaises(ValueError):
            dispersion.one_fluid_dispersion(10e-9, 30.0, ions, ions, k_vec=k_vec)

    def test_one_fluid_dispersion_parallel(self):
        # At theta = 0, the sound wave decouples from the two circularly
        # polarized waves
        ions = {"n": 10e6, "t": 10.0, "gamma": 1.0}
        result = dispersion.one_fluid_dispersion(10e-9, 0.0, ions, ions)
        w_c = np.stack([wc_.data for wc_ in result])
        k, c_s = [result[0].k.data, result[0].attrs["c_s"]]
        is_sound = np.isclose(w_c, k * c_s, rtol=1e-8)
        np.testing.assert_array_equal(np.sum(is_sound, axis=0), 1)


if __name__ == "__main__":
    unittest.main()
