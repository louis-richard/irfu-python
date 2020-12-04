#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

from pyrfu import mms, pyrf

import unittest

import numpy as np


class TestPyrf(unittest.TestCase):
    def setUp(self):
        """integration test setup."""
        tint = ["2019-09-14T08:00:00.000", "2019-09-14T08:00:30.000"]
        self.b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, 2)
        self.e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, 2)

    def test_calc_fs(self):
        """integration test on sampling frequency computation"""
        fs = pyrf.calc_fs(self.b_xyz)

        self.assertEqual(np.round(fs), 128.0)

    def test_calc_dt(self):
        """integration test on sampling tme step computation"""
        dt = pyrf.calc_dt(self.b_xyz)

        self.assertEqual(int(dt * 1e9), 7813000)

    def test_cross(self):
        """integration test on vector cross product computation"""
        exb = pyrf.cross(self.e_xyz, self.b_xyz)
        res = pyrf.dot(exb, self.e_xyz)
        res = np.mean(res)
        self.assertTrue(res < 1e-5)

    def test_resample(self):
        """integration test on time resampling"""
        e_xyz = pyrf.resample(self.e_xyz, self.b_xyz)

        self.assertTrue((pyrf.resample(e_xyz, self.b_xyz).time.data == self.b_xyz.time.data).all())


if __name__ == "__main__":
    unittest.main()
