import argparse
import os
import sys
from datetime import datetime

import click
import GooseEPM as epm
import h5py

from . import tag
from ._version import version


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:
    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])


def read_version(file: h5py.File, path: str) -> str:
    """
    Read version of this library from file.

    :param file: HDF5 archive.
    :param path: Path in ``file`` to read version from, as attribute "dependencies".
    :return: Version string.
    """

    if path not in file:
        return None

    ret = file[path].attrs["dependencies"]
    return [i for i in ret if i.startswith("mycode_thermal_epm")][0].split("=")[1]


def path_meta(module: str, function: str) -> str:
    """
    Create a path for metadata.

    :param module: Name of module.
    :param function: Name of function.
    :return: "/meta/{stamp}_module={module}_function={function}"
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    return f"/meta/{stamp}_{module}_{function}"


def create_check_meta(
    file: h5py.File,
    path: str,
    dev: bool = False,
) -> None:
    """
    Create, update, or read/check metadata.
    This function creates a group ``file["meta"][path]`` and sets the attributes::

        "dependencies": The current version of all relevant dependencies.
        "compiler": Compiler information.

    :param file: HDF5 archive.
    :param path: Path relative to ``file["meta"]``.
    :param dev: Allow uncommitted changes.
    """

    deps = sorted(list(set(list(epm.version_dependencies()) + ["mycode_thermal_epm=" + version])))
    compiler = epm.version_compiler()

    assert dev or not tag.any_has_uncommitted(deps)

    if "meta" not in file:
        meta = file.create_group("meta")
    else:
        meta = file["meta"]

    groups = sorted([i for i in meta])[::-1]
    for existing in groups:
        if sorted([i for i in meta[existing].attrs]) == ["dependencies", "compiler"]:
            a = meta[existing].attrs["compiler"] == compiler
            b = meta[existing].attrs["dependencies"] == deps
            if a == b:
                print("linking", path, "to", existing)
                meta[path] = meta[existing]
                return

    group = meta.create_group(path)
    group.attrs["dependencies"] = deps
    group.attrs["compiler"] = epm.version_compiler()


def _check_overwrite_file(filepath: str, force: bool):
    if force or not os.path.isfile(filepath):
        return

    if not click.confirm(f'Overwrite "{filepath}"?'):
        raise OSError("Cancelled")
