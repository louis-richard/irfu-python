#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import copy
import functools
import logging
import math
import numbers
import os
from typing import Literal, Optional

# 3rd party imports
import keyring
import keyring.errors
from keyring.backends.chainer import ChainerBackend
from keyrings.alt.file import PlaintextKeyring

# Local imports
from pyrfu._core.user_config import read_config, user_config_path, write_config

logger = logging.getLogger(__name__)

# Default configuration distributed with the package
_PACKAGE_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


def _user_config_path() -> str:
    r"""Path of the MMS configuration file in the user configuration directory
    (e.g. ~/Library/Application Support/pyrfu on macOS, ~/.config/pyrfu on Linux).
    It is created from the package configuration file the first time, and the
    package file is used if the user directory cannot be written."""
    return user_config_path("mms_config.json", _PACKAGE_CFG_PATH)


def _mms_cfg_path() -> str:
    r"""Path of the MMS configuration: MMS_CFG_PATH if set on the module (e.g.
    patched), else the user configuration file, created on first use (import
    pyrfu created it)."""
    return globals().get("MMS_CFG_PATH") or _user_config_path()


def __getattr__(name):
    # MMS_CFG_PATH computed on first use (PEP 562)
    if name == "MMS_CFG_PATH":
        return _user_config_path()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Service name of the MMS SDC credentials in the keyring
SDC_SERVICE = "mms-sdc"


def _secure_keyring():
    r"""System keyring (e.g. macOS Keychain, Windows Credential Locker, Secret
    Service) if one is available, None otherwise."""
    backend = keyring.get_keyring()
    backends = backend.backends if isinstance(backend, ChainerBackend) else [backend]

    if any(b.priority >= 1 for b in backends):
        return backend

    return None


def _get_credential(username: str):
    r"""MMS SDC credential of username, from the system keyring or, for the
    credentials saved by previous versions or without system keyring, from the
    plaintext keyring file."""
    secure = _secure_keyring()
    credential = secure.get_credential(SDC_SERVICE, username) if secure else None

    if credential is None:
        credential = PlaintextKeyring().get_credential(SDC_SERVICE, username)

    return credential


def _set_password(username: str, password: str) -> None:
    r"""Save the MMS SDC credential in the system keyring, or in the plaintext
    keyring file (with a warning) if there is no system keyring."""
    secure = _secure_keyring()
    plaintext = PlaintextKeyring()

    if secure is not None:
        logger.info("Updating MMS SDC credentials in the system keyring...")
        secure.set_password(SDC_SERVICE, username, password)

        # Remove the copy saved in plain text by previous versions
        try:
            plaintext.delete_password(SDC_SERVICE, username)
        except keyring.errors.PasswordDeleteError:
            pass
    else:
        logger.warning(
            "No system keyring available: MMS SDC credentials saved in plain text "
            "in %s",
            plaintext.file_path,
        )
        plaintext.set_password(SDC_SERVICE, username, password)


# Settings used when there is no configuration yet, or with reset=True
_DEFAULT_CONFIG = {
    "default": "local",
    "local": ".",
    "sdc": {"rights": "public", "username": "username"},
    "aws": "",
    # Cache of the files read from the SDC and AWS: on, 1 GB, files removed 4 h
    # after their last use, in the user cache directory ("")
    "cache": {"enabled": True, "size": 1e9, "max_age": 4.0, "dir": ""},
}


def _config_stamp():
    r"""Path, inode, modification time and size of the MMS configuration file
    (path only if it can't be read): the key of the caches of the
    configuration. db_init writes a new file (new inode) every time."""
    path = _mms_cfg_path()

    try:
        stat = os.stat(path)
    except OSError:
        return (path,)

    return path, stat.st_ino, stat.st_mtime_ns, stat.st_size


def _read_config() -> dict:
    r"""Current MMS configuration, completed with the default settings: the one
    reader of the configuration file, read again when it changes (e.g., after
    db_init). Returns a copy, so that callers cannot change the cache."""
    return copy.deepcopy(_read_config_cached(_config_stamp()))


@functools.lru_cache(maxsize=1)
def _read_config_cached(stamp) -> dict:
    r"""MMS configuration of the file of `stamp` (see _config_stamp)."""
    config = copy.deepcopy(_DEFAULT_CONFIG)

    try:
        current = read_config(stamp[0])
    except (OSError, ValueError):
        return config

    if not isinstance(current, dict):
        return config

    for key in ["default", "local", "aws"]:
        if key in current:
            config[key] = current[key]

    if isinstance(current.get("sdc"), dict):
        config["sdc"].update(
            {k: v for k, v in current["sdc"].items() if k in ["rights", "username"]}
        )

    if isinstance(current.get("cache"), dict):
        config["cache"].update(
            {k: v for k, v in current["cache"].items() if k in config["cache"]}
        )

    return config


def _set_cache(config, enabled, size, max_age, directory):
    r"""Sets the given cache settings of `config` (checked). A directory is
    saved as an absolute path, so that the cache is the same from every
    working directory."""
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError(f"cache must be True or False, not {enabled!r}")

    for name, value in [("cache_size", size), ("cache_max_age", max_age)]:
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a number, not {value!r}")

        if math.isnan(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")

    if directory is not None and not isinstance(directory, (str, os.PathLike)):
        raise TypeError(f"cache_dir must be a path, not {directory!r}")

    if directory:
        directory = os.path.normpath(
            os.path.abspath(os.path.expanduser(os.fspath(directory)))
        )

    for key, value in [
        ("enabled", enabled),
        ("size", None if size is None else float(size)),
        ("max_age", None if max_age is None else float(max_age)),
        ("dir", None if directory is None else os.fspath(directory)),
    ]:
        if value is not None:
            config[key] = value


def db_init(
    default: Optional[Literal["local", "sdc", "aws"]] = None,
    local: Optional[str] = None,
    sdc: Optional[Literal["public", "sitl"]] = None,
    sdc_username: Optional[str] = None,
    sdc_password: Optional[str] = None,
    aws: Optional[str] = None,
    cache: Optional[bool] = None,
    cache_size: Optional[float] = None,
    cache_max_age: Optional[float] = None,
    cache_dir: Optional[str] = None,
    reset: bool = False,
) -> None:
    r"""Manage the MMS data access configuration.

    The default resource to access MMS data, the local path to use, the MMS SDC
    rights and username, and the Amazon Web Services (AWS) bucket name are
    saved in the MMS configuration file of the user configuration directory
    (`pyrfu.mms.MMS_CFG_PATH`). The MMS SDC password is saved in the system
    keyring (macOS Keychain, Windows Credential Locker, Secret Service on
    Linux) or, if there is none, in plain text in the keyring file in your
    home directory. It is never written in the configuration file.

    Only the settings that are given are changed; the others keep their
    current value (or the default one if there is no configuration yet, or
    with `reset=True`).

    Parameters
    ----------
    default : {"local", "sdc", "aws"}, Optional
        Name of the default resource to access the MMS data. Default is local.
    local : str, Optional
        Local path to MMS data. Default is the current directory.
    sdc : {"public", "sitl"}, Optional
        Rights to access MMS data from SDC. Default is public. "sitl" needs the
        credentials of `sdc_username` in the keyring.
    sdc_username : str, Optional
        MMS SDC username. Given alone, the configuration uses this user, whose
        password must already be in the keyring.
    sdc_password : str, Optional
        MMS SDC password of `sdc_username`, saved in the keyring. It needs
        `sdc_username`.
    aws : str, Optional
        Bucket name and prefix to MMS data in AWS S3, as "bucket/prefix". Empty
        (the default) uses the public MMS archive on NASA HelioCloud
        ("gov-nasa-hdrl-data1/spdf/cdaweb/data/mms", no AWS credentials needed).
    cache : bool, Optional
        Keep the files read from the SDC and AWS, and the file listings for
        `cache_max_age`, in a cache, so that reading them again needs no
        network. Default is True. A file reprocessed on the SDC or AWS is seen
        when the listing expires, or after :func:`pyrfu.mms.clear_cache`.
    cache_size : float, Optional
        Maximum size of the cache (bytes): the least recently used files are
        removed above it, and larger files are not cached. Default is 1e9.
    cache_max_age : float, Optional
        Hours after their last use when the cached files are removed, and
        after which the file listings are made again. Default is 4.
    cache_dir : str, Optional
        Directory of the cache, in the layout of the SDC (that of a `local`
        directory), saved as an absolute path. Empty (the default) uses the
        user cache directory, e.g. ~/Library/Caches/pyrfu/mms on macOS,
        ~/.cache/pyrfu/mms on Linux.
    reset : bool, Optional
        Start from the default settings instead of the current ones. The
        keyring is not changed. Default is False.

    Raises
    ------
    NotImplementedError
        If the default resource is not implemented.
    FileNotFoundError
        If the local path doesn't exist.
    TypeError
        If `cache` is not a bool, `cache_size` or `cache_max_age` not a number,
        or `cache_dir` not a path.
    ValueError
        If the SDC rights are not "public" or "sitl", if `sdc_password` is
        given without `sdc_username`, or if `cache_size` or `cache_max_age` is
        not positive.

    Examples
    --------
    >>> from pyrfu import mms

    Use the local data in /data/mms by default (the other settings are kept)

    >>> mms.db_init(default="local", local="/data/mms")

    Save the MMS SDC team credentials and use them

    >>> mms.db_init(sdc="sitl", sdc_username="user", sdc_password="password")

    Keep up to 5 GB of files for a day, or turn the cache off

    >>> mms.db_init(cache_size=5e9, cache_max_age=24)
    >>> mms.db_init(cache=False)

    """
    config = copy.deepcopy(_DEFAULT_CONFIG) if reset else _read_config()

    # Check and set the given settings only
    if default is not None:
        if default.lower() not in ["local", "sdc", "aws"]:
            raise NotImplementedError(f"Resource {default} is not implemented!!")

        config["default"] = default.lower()

    if local is not None or reset:
        # Normalize the path and make sure that it exists
        local = os.path.normpath(
            os.path.abspath(
                os.path.expanduser(config["local"] if local is None else local)
            )
        )

        if not os.path.exists(local):
            raise FileNotFoundError(f"{local} doesn't exists!!")

        config["local"] = local

    if sdc is not None:
        if sdc.lower() not in ["public", "sitl"]:
            raise ValueError("sdc must be 'public' or 'sitl'!!")

        config["sdc"]["rights"] = sdc.lower()

    if sdc_password is not None and sdc_username is None:
        raise ValueError("sdc_password needs sdc_username")

    if sdc_username is not None:
        config["sdc"]["username"] = sdc_username

    if aws is not None:
        config["aws"] = aws

    _set_cache(config["cache"], cache, cache_size, cache_max_age, cache_dir)

    # First configuration: the current directory, as an absolute path
    if not os.path.isabs(config["local"]):
        config["local"] = os.path.normpath(os.path.abspath(config["local"]))

    # Save the password in the keyring only if given (never a placeholder)
    if sdc_password is not None:
        _set_password(sdc_username, sdc_password)

    if config["sdc"]["rights"] == "sitl" and not _get_credential(
        config["sdc"]["username"]
    ):
        logger.warning(
            "No MMS SDC credentials for %s in the keyring: use "
            "mms.db_init(sdc_username=..., sdc_password=...)",
            config["sdc"]["username"],
        )

    path = _mms_cfg_path()
    logger.info("Updating MMS data access configuration in %s...", path)

    # At once: a write that fails leaves the previous configuration (a partial
    # file was read as the default settings)
    write_config(path, config)
