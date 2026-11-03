#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Units of the outputs, in the notation of the MMS files: products separated by
# a space, ratios with "/", powers with "^", a compound unit in parentheses when
# combined ("(mV/m) s"), "" for dimensionless. The strings are composed, not
# simplified. Without UNITS on an input, the output has none (None).


def _units(inp):
    r"""UNITS of a time series, or of the data of a distribution (Dataset),
    None without."""
    if isinstance(inp, Dataset):
        inp = inp["data"] if "data" in inp else inp

    return getattr(inp, "attrs", {}).get("UNITS")


def _outside(units, chars):
    r"""Whether units has one of chars outside parentheses."""
    depth = 0

    for char in units:
        depth += {"(": 1, ")": -1}.get(char, 0)

        if depth == 0 and char in chars:
            return True

    return False


def _wrap(units, power=False):
    r"""Units in parentheses when compound (and, for a power, when they have
    one already), "(mV/m)^2" not being compound."""
    if _outside(units, " /^" if power else " /"):
        return f"({units})"

    return units


def _product(*units):
    r"""Units of a product; identical factors as a power ("nT^2")."""
    if any(unit is None for unit in units):
        return None

    factors = [unit.strip() for unit in units if unit.strip()]
    out = []

    for unit in dict.fromkeys(factors):
        count = factors.count(unit)
        out.append(_power(unit, count) if count > 1 else _wrap(unit))

    return " ".join(out)


def _ratio(numerator, denominator):
    r"""Units of a ratio."""
    if numerator is None or denominator is None:
        return None

    numerator, denominator = numerator.strip(), denominator.strip()

    if not denominator:
        return numerator

    numerator = f"({numerator})" if _outside(numerator, "/") else numerator or "1"

    return f"{numerator}/{_wrap(denominator)}"


def _power(units, exponent):
    r"""Units to a power."""
    if units is None:
        return None

    if not units.strip() or exponent == 1:
        return units.strip()

    return f"{_wrap(units.strip(), power=True)}^{exponent:g}"


def _attrs(attrs, units):
    r"""Copy of attrs with UNITS set to units, or without UNITS for None."""
    attrs = {k: v for k, v in attrs.items() if k != "UNITS"}

    if units is not None:
        attrs["UNITS"] = units

    return attrs


def _check_units(inp, expected, name):
    r"""Warning when inp has UNITS that are not those the computation assumes
    (expected, a str or a list of the accepted spellings); no check without
    UNITS."""
    units = _units(inp)
    expected = [expected] if isinstance(expected, str) else list(expected)

    if units is not None and units.strip() not in expected:
        warnings.warn(
            f"{name} is assumed in {expected[0]}, not {units!r}: the result is in "
            "the units of the documentation",
            UserWarning,
            stacklevel=3,
        )
