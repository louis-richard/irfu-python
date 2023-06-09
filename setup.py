#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import pathlib

from setuptools import setup, find_packages

# 3rd party imports
# from sphinx.setup_command import BuildDoc

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.34"
__status__ = "Prototype"

HERE = pathlib.Path(__file__).parent

VERSION = __version__
PACKAGE_NAME = "pyrfu"
AUTHOR = "Louis RICHARD"
AUTHOR_EMAIL = "louir@irfu.se"
URL = "https://github.com/louis-richard/irfu-python"

LICENSE = "MIT License"
DESCRIPTION = "Python Space Physics (RymdFysik) Utilities"

with open("README.rst", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


INSTALL_REQUIRES = [
    "cdflib",
    "geopack",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "python-dateutil",
    "requests",
    "scipy",
    "sphinx",
    "tqdm",
    "xarray",
]

PYTHON_REQUIRES = ">=3.7"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
)
