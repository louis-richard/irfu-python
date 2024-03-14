#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

# 3rd party imports
import matplotlib.pyplot as plt
from ddt import data, ddt, unpack
from matplotlib.axes import Axes

# Local imports
from .. import plot
from . import generate_data, generate_ts


@ddt
class PlotLineTestCase(unittest.TestCase):
    @data((0.0, generate_ts(64.0, 100)), (plt.subplots(3)[1], generate_ts(64.0, 100)))
    @unpack
    def test_plot_line_axis_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(axis, inp)

    @data((plt.subplots(1)[1], generate_data(100)))
    @unpack
    def test_plot_line_inp_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(axis, inp)

    @data(
        (plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=random.randint(3, 10)))
    )
    @unpack
    def test_plot_line_inp_shape(self, axis, inp):
        with self.assertRaises(NotImplementedError):
            plot.plot_line(axis, inp)

    @data(
        (None, generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        (plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        (
            plt.subplots(3)[1][0],
            generate_ts(64.0, 100, tensor_order=random.randint(0, 2)),
        ),
    )
    @unpack
    def test_plot_line_output(self, axis, inp):
        result = plot.plot_line(axis, inp)
        self.assertIsInstance(result, Axes)


@ddt
class AddPositionTestCase(unittest.TestCase):
    @data(generate_ts(64.0, 100, tensor_order=1))
    def test_add_position_output(self, value):
        result = plot.add_position(plt.subplots(1)[1], value)
        self.assertIsInstance(result, Axes)


if __name__ == "__main__":
    unittest.main()
