#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .initialize import initialize
# from .em_3d_matrix import em_3d_matrix
# from .es_3d_matrix import es_3d_matrix
# from .kernel import kernel
# from .plot_all import plot_all

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.4.8"
__status__ = "Prototype"


class PDRK(object):
    r"""Class of Kinetic Plasma Dispersion Solver run."""

    def __init__(self, file_path: str):
        r"""Creates and setup the kinetic plasma dispersion solver run.

        Parameters
        ----------
        file_path : str
            Path to the config (.yml) file.

        """
        self.config = initialize(file_path)
        self.dispersion = {}