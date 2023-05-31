import argparse
import sys
import uuid

import GooseEPM as epm
import h5py

from . import tag
from ._version import version


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:
    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])


def create_check_meta(
    file: h5py.File = None,
    path: str = None,
    dev: bool = False,
) -> h5py.Group:
    """
    Create, update, or read/check metadata. This function creates metadata as attributes to a group
    ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations.
        "version": The current version of this code (updated).
        "dependencies": The current version of all relevant dependencies (updated).
        "compiler": Compiler information (updated).

    :param file: HDF5 archive.
    :param path: Path in ``file`` to store/read metadata.
    :param dev: Allow uncommitted changes.
    :return: Group to metadata.
    """

    deps = sorted(
        list(
            set(
                list(epm.version_dependencies())
                + [
                    "mycode_thermal_epm=" + version,
                ]
            )
        )
    )

    assert dev or not tag.any_has_uncommitted(deps)

    if file is None:
        return None

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = epm.version_compiler()
        return meta

    meta = file[path]
    if file.mode in ["r+", "w", "a"]:
        assert dev or tag.all_greater_equal(deps, meta.attrs["dependencies"])
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = epm.version_compiler()
    else:
        assert dev or tag.all_equal(deps, meta.attrs["dependencies"])
    return meta
