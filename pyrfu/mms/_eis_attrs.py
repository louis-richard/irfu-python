#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np


def _is_nan(attr):
    r"""True for a NaN or NaT scalar, False for any other value."""
    try:
        return bool(np.isnan(attr))
    except (TypeError, ValueError):
        return False


def _equal(attr0, attr1):
    r"""True if the attribute values are equal, arrays (and dicts or lists of
    them) included: == compares arrays element by element, which raised
    ValueError in a condition. NaN and NaT (fill values) equal themselves."""
    if isinstance(attr0, dict) and isinstance(attr1, dict):
        return attr0.keys() == attr1.keys() and all(
            _equal(attr0[k], attr1[k]) for k in attr0
        )

    if isinstance(attr0, (list, tuple)) and isinstance(attr1, (list, tuple)):
        return len(attr0) == len(attr1) and all(
            _equal(a_0, a_1) for a_0, a_1 in zip(attr0, attr1)
        )

    if isinstance(attr0, np.ndarray) or isinstance(attr1, np.ndarray):
        try:
            return bool(np.array_equal(attr0, attr1, equal_nan=True))
        except TypeError:  # no NaN in strings and objects
            return bool(np.array_equal(attr0, attr1))

    return bool(attr0 == attr1) or (_is_nan(attr0) and _is_nan(attr1))


def _common_attrs(attrs_list, skip=()):
    r"""Attributes of the first dict of `attrs_list`, except those in `skip`,
    with the same value in all the others."""
    return {
        k: v
        for k, v in attrs_list[0].items()
        if k not in skip
        and all(k in attrs and _equal(attrs[k], v) for attrs in attrs_list[1:])
    }
