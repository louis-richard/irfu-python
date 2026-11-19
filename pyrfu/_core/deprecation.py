#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

# Deprecations of pyrfu (policy in CONTRIBUTING.rst): a deprecated item keeps
# working from "since" to "removed_in" (a major version), with a FutureWarning
# (shown to users, unlike DeprecationWarning) pointing at the caller. The
# ".. deprecated:: since" directive of its docstring is written by hand.

P = ParamSpec("P")
R = TypeVar("R")


def _message(
    name: str,
    since: str,
    removed_in: str,
    alternative: str | None = None,
    reason: str | None = None,
) -> str:
    r"""Message of a deprecation, the same for all of pyrfu."""
    message = (
        f"{name} is deprecated since pyrfu {since} and will be removed in pyrfu "
        f"{removed_in}"
    )

    if alternative is not None:
        message += f"; use {alternative} instead"

    message += "."

    if reason is not None:
        message += f" {reason}"

    return message


def _level(stacklevel: int) -> int:
    r"""Stack level for warnings.warn called in warn_deprecated: the
    `stacklevel`-th frame outside this module, counted from the caller of
    warn_deprecated (the frames of the wrappers of the decorators, however
    many are stacked, are skipped; skip_file_prefixes of warnings.warn does not
    skip them in Python 3.12)."""
    level = 2  # warnings.warn in warn_deprecated, the caller of warn_deprecated
    frame = inspect.currentframe()

    # Skip this function and warn_deprecated
    for _ in range(2):
        frame = None if frame is None else frame.f_back

    remaining = stacklevel

    while frame is not None and frame.f_back is not None:
        if frame.f_code.co_filename != __file__:
            remaining -= 1

            if remaining == 0:
                break

        frame = frame.f_back
        level += 1

    return level


def warn_deprecated(
    name: str,
    since: str,
    removed_in: str,
    alternative: str | None = None,
    reason: str | None = None,
    stacklevel: int = 2,
) -> None:
    r"""Warn (FutureWarning) that `name` is deprecated. The stack level is
    counted from the function that calls warn_deprecated, without the frames
    of this module: 2, the default, is the caller of that function, as
    warnings.warn in it; 1 is that function (for the wrappers of the
    decorators, the caller of the decorated function)."""
    warnings.warn(
        _message(name, since, removed_in, alternative, reason),
        FutureWarning,
        stacklevel=_level(stacklevel),
    )


def deprecated(
    since: str,
    removed_in: str,
    alternative: str | None = None,
    reason: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    r"""Decorator of a deprecated function: it warns at each call."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warn_deprecated(
                func.__name__, since, removed_in, alternative, reason, stacklevel=1
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_kwarg(
    old: str,
    since: str,
    removed_in: str,
    new: str | None = None,
    reason: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    r"""Decorator of a function with a deprecated argument `old`: it warns
    when the argument is given.

    With `new`, the argument was renamed: `old` is no longer a parameter of
    the function, and its value given by keyword is passed as `new`
    (TypeError if both are given). Without, `old` is still a parameter, given
    by keyword or position, and is passed unchanged (the function decides
    what to do with it, e.g. ignore it)."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(func)
        name = f"argument {old!r} of {func.__name__}"

        if new is None:
            alternative = None
        else:
            alternative = f"{new!r}"

        if new is None and old not in signature.parameters:
            raise ValueError(f"{func.__name__} has no argument {old!r}")

        if new is not None and new not in signature.parameters:
            raise ValueError(f"{func.__name__} has no argument {new!r}")

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if new is not None:
                if old in kwargs:
                    warn_deprecated(
                        name, since, removed_in, alternative, reason, stacklevel=1
                    )

                    if new in kwargs:
                        raise TypeError(
                            f"{func.__name__}() got {old!r} and {new!r}: give "
                            f"{new!r} only ({old!r} is its deprecated name)"
                        )

                    kwargs[new] = kwargs.pop(old)

            elif old in signature.bind_partial(*args, **kwargs).arguments:
                warn_deprecated(
                    name, since, removed_in, alternative, reason, stacklevel=1
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_alias(
    func: Callable[P, R], old_name: str, since: str, removed_in: str
) -> Callable[P, R]:
    r"""The function `func` under its former name `old_name`: it warns at
    each call and calls `func`."""

    @functools.wraps(func)
    def alias(*args: P.args, **kwargs: P.kwargs) -> R:
        warn_deprecated(old_name, since, removed_in, func.__name__, stacklevel=1)
        return func(*args, **kwargs)

    alias.__name__ = old_name
    alias.__qualname__ = old_name

    return alias
