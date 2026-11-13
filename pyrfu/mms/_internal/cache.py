#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Cache of the MMS files read from the SDC and AWS.

The files are kept in the user cache directory, in the layout of the SDC (that
of a local MMS data directory), and the file listings for a few hours, so that
reading the same data again needs no network. The files whose last use is
older than the maximum age are removed, then the least recently used ones
while the cache is larger than its maximum size. The settings are those of
:func:`pyrfu.mms.db_init`.
"""

# Built-in imports
import dataclasses
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import time
from typing import Callable, Optional

# 3rd party imports
import platformdirs

# Local imports
from pyrfu._core.user_config import atomic_write as _atomic_write
from pyrfu.mms.db_init import _read_config

logger = logging.getLogger(__name__)

# Sub-directories of the cache: the files of each source and the listings.
# clear_cache removes these only, whatever else is in the cache directory.
_SOURCES = ("aws", "sdc")
_LISTINGS = "listings"

# Time tag and version at the end of the names of the MMS files
_TIME_TAG = re.compile(r"_(\d{4})(\d{2})(\d{2})(\d{6})?_v\d+\.\d+\.\d+\.cdf$")


@dataclasses.dataclass(frozen=True)
class RemoteFile:
    r"""A file of the SDC or AWS, with its path in the cache."""

    source: str
    path: str
    size: Optional[int] = None
    url: str = ""
    bucket: str = ""
    key: str = ""

    def __str__(self):
        return posixpath.basename(self.path)


def file_size(size) -> Optional[int]:
    r"""File size (bytes) of a listing, None if unknown."""
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def _settings() -> dict:
    r"""Cache settings of the MMS configuration: enabled, maximum size
    (bytes), maximum age since last use (hours) and directory."""
    cache = _read_config()["cache"]
    directory = cache["dir"] or os.path.join(
        platformdirs.user_cache_dir("pyrfu"), "mms"
    )

    return {
        "enabled": bool(cache["enabled"]),
        "size": float(cache["size"]),
        "max_age": float(cache["max_age"]),
        "dir": os.path.normpath(os.path.abspath(os.path.expanduser(directory))),
    }


def enabled() -> bool:
    r"""True if the cache is on."""
    return _settings()["enabled"]


def mms_path(mms_id: str, var: dict, file_name: str) -> str:
    r"""Path of an MMS file in the layout of the SDC: spacecraft, instrument,
    rate, level[, data type], year, month[, day for burst], file name."""
    parts = [f"mms{mms_id}", var["inst"], var["tmmode"], var["lev"]]
    parts += [var["dtype"]] if var.get("dtype") else []
    match = _TIME_TAG.search(file_name)

    if match:
        year, month, day = match.groups()[:3]
        parts += [year, month] + ([day] if var["tmmode"] == "brst" else [])

    return posixpath.join(*parts, file_name)


def _ensure_dir(settings: dict, sub_dir: str, path: str) -> None:
    r"""Creates a directory of the cache. The sub-directory of the cache it is
    in (aws, sdc or listings) is readable by the user only, which protects all
    the files below it (the cache can hold SITL data): os.makedirs gives its
    mode to the last directory only. The cache directory itself is left as it
    is (it can be a directory of the user)."""
    os.makedirs(os.path.join(settings["dir"], sub_dir), mode=0o700, exist_ok=True)
    os.makedirs(path, exist_ok=True)


def _listing_path(settings: dict, key: dict) -> str:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return os.path.join(settings["dir"], _LISTINGS, f"{digest}.json")


def listing(key: dict, lister: Callable[[], list]) -> list:
    r"""Listing `lister()` (JSON-serialisable items), from the cache if it was
    made less than the maximum age ago with the same `key`."""
    settings = _settings()
    path = _listing_path(settings, key)

    try:
        with open(path, "r", encoding="utf-8") as file:
            cached = json.load(file)

        if time.time() - cached["created"] < settings["max_age"] * 3600.0:
            return cached["items"]
    except (OSError, ValueError, KeyError, TypeError):
        pass

    items = lister()

    # An empty listing is not kept: the files may come later
    if not items:
        return items

    try:
        _ensure_dir(settings, _LISTINGS, os.path.dirname(path))
        _atomic_write(path, json.dumps({"created": time.time(), "items": items}))
    except OSError as err:
        logger.warning("Cannot write the MMS file listing to the cache: %s", err)

    return items


def _read_cached(path: str, size: Optional[int]) -> Optional[bytes]:
    r"""Content of a cached file, None if it is missing or has not the expected
    size (it is then removed). Refreshes its time of last use."""
    try:
        with open(path, "rb") as file:
            content = file.read()
    except OSError:
        return None

    if size is not None and len(content) != size:
        logger.warning("Cached %s is incomplete: read again", os.path.basename(path))
        _remove(path)
        return None

    try:
        os.utime(path)
    except OSError:
        pass

    return content


def read(remote: RemoteFile, fetch: Callable[[], bytes]) -> bytes:
    r"""Content of `remote`, from the cache or from `fetch()` (then cached if
    it is not larger than the maximum size). A failure of the cache gives a
    warning and the content of `fetch()`."""
    settings = _settings()
    path = os.path.join(settings["dir"], remote.source, *remote.path.split("/"))

    content = _read_cached(path, remote.size)

    if content is not None:
        logger.debug("%s read from the cache", remote)
        return content

    content = fetch()

    if len(content) > settings["size"]:
        return content

    try:
        _ensure_dir(settings, remote.source, os.path.dirname(path))
        _atomic_write(path, content)
        _evict(settings)
    except OSError as err:
        logger.warning("Cannot write %s to the MMS file cache: %s", remote, err)

    return content


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _cached_files(settings: dict) -> list:
    r"""(last use, size, path) of the cached files and listings."""
    files = []

    for sub_dir in (*_SOURCES, _LISTINGS):
        for root, _, names in os.walk(os.path.join(settings["dir"], sub_dir)):
            for name in names:
                path = os.path.join(root, name)

                try:
                    stat = os.stat(path)
                except OSError:
                    continue

                files.append((stat.st_mtime, stat.st_size, path))

    return files


def _evict(settings: dict) -> None:
    r"""Removes the files not used for the maximum age (and the temporary
    files left by interrupted downloads), then the least recently used ones
    while the cache is larger than its maximum size."""
    limit = time.time() - settings["max_age"] * 3600.0
    kept = []

    for last_use, size, path in _cached_files(settings):
        if last_use < limit:
            _remove(path)
        elif not path.endswith(".part"):
            # Downloads in progress (other processes) are not removed by size
            kept.append((last_use, size, path))

    total = sum(size for _, size, _ in kept)

    for _, size, path in sorted(kept):
        if total <= settings["size"]:
            break

        _remove(path)
        total -= size


def cache_info() -> dict:
    r"""Settings and content of the cache of the MMS files read from the SDC
    and AWS.

    Returns
    -------
    dict
        ``enabled``, ``dir``, ``max_size`` (bytes), ``max_age`` (hours),
        ``n_files`` and ``size`` (bytes) of the cached files, ``n_listings``
        (file listings).

    See Also
    --------
    pyrfu.mms.db_init, pyrfu.mms.clear_cache

    Examples
    --------
    >>> from pyrfu import mms
    >>> mms.cache_info()["size"]

    """
    settings = _settings()
    files = _cached_files(settings)
    listings_dir = os.path.join(settings["dir"], _LISTINGS) + os.sep
    n_listings = sum(path.startswith(listings_dir) for _, _, path in files)

    return {
        "enabled": settings["enabled"],
        "dir": settings["dir"],
        "max_size": settings["size"],
        "max_age": settings["max_age"],
        "n_files": len(files) - n_listings,
        "size": sum(
            size for _, size, path in files if not path.startswith(listings_dir)
        ),
        "n_listings": n_listings,
    }


def clear_cache() -> None:
    r"""Empty the cache of the MMS files read from the SDC and AWS (the files
    and the file listings): the next reads use the network.

    See Also
    --------
    pyrfu.mms.cache_info, pyrfu.mms.db_init

    Examples
    --------
    >>> from pyrfu import mms
    >>> mms.clear_cache()

    """
    settings = _settings()

    for sub_dir in (*_SOURCES, _LISTINGS):
        shutil.rmtree(os.path.join(settings["dir"], sub_dir), ignore_errors=True)

    logger.info("MMS file cache %s cleared", settings["dir"])
