#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Type aliases shared by the annotations of pyrfu."""

# Built-in imports
from typing import Any

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Floating-point array of any precision (float32 data, float64 computations)
type FloatArray = NDArray[np.floating[Any]]

# Array of times
type TimeArray = NDArray[np.datetime64]
