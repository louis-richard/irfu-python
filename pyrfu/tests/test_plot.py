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


@ddt
class PlTxTestCase(unittest.TestCase):
    @data(
        ([generate_ts(64.0, 100, tensor_order=3) for _ in range(4)], "cluster"),
        ([generate_ts(64.0, 100, tensor_order=0) for _ in range(4)], "bazinga"),
    )
    @unpack
    def test_pl_tx_input(self, value, colors):
        with self.assertRaises(NotImplementedError):
            plot.pl_tx(plt.subplots(1)[1], value, colors=colors)

    @data(
        (
            None,
            [
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
            ],
        ),
        (
            plt.subplots(1)[1],
            [
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
            ],
        ),
        (
            plt.subplots(1)[1],
            [
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
            ],
        ),
        (
            plt.subplots(1)[1],
            [
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
            ],
        ),
    )
    @unpack
    def test_pl_tx_output(self, ax, value):
        result = plot.pl_tx(ax, value, 0)
        self.assertIsInstance(result, Axes)


if __name__ == "__main__":
    unittest.main()
