import argparse
import os
import re
import sys
import textwrap
from datetime import datetime

import click
import GooseEPM as epm
import h5py

from . import tag
from ._version import version


class MyFmt(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter,
):
    pass


def _fmt_doc(text: str) -> str:
    """
    Format the docstring:

    *   Dedent.
    *   Strip.
    """
    return textwrap.dedent(text).split(":param")[0].strip()


def _parse(parser: argparse.ArgumentParser, cli_args: list) -> argparse.ArgumentParser:
    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])


# https://stackoverflow.com/a/68901244/2646505
def docstring_copy(source_function):
    """
    Copies the docstring of the given function to another.
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @docstring_copy(foo)
        def bar():
            ...
    """

    def wrapped(func):
        func.__doc__ = source_function.__doc__
        return func

    return wrapped


def docstring_append_cli():
    """
    Append the docstring with the default CLI help.
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @docstring_append_cli
        def bar():
            ...
    """

    args = textwrap.dedent(
        """
    :param cli_args:
        Command line arguments, see ``--help`` for details. Default: ``sys.argv[1:]`` is used.

    :param _return_parser: Return parser instead of executing (for documentation).
    """
    )

    ret = ":return: ``None`` if executed, parser if ``return_parser``."

    def wrapped(func):
        doc = textwrap.dedent(func.__doc__)
        s = re.split(r"(\n\s*:param \w*:)(.*)", doc)
        base = s[0]
        arguments = "".join(s[1:])
        doc = "\n\n".join([base, args, arguments, ret])
        func.__doc__ = textwrap.indent(re.sub(r"(\n\n+)", r"\n\n", doc), "    ")
        return func

    return wrapped


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
    ver = [i for i in ret if i.startswith("depinning_creep_2024")]

    if len(ver) == 0:
        ver = [i for i in ret if i.startswith("mycode_thermal_epm=")]
        assert len(ver) == 1

    return ver[0].split("=")[1]


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

    deps = sorted(list(set(list(epm.version_dependencies()) + ["depinning_creep_2024=" + version])))
    compiler = sorted(epm.version_compiler())

    assert dev or not tag.any_has_uncommitted(deps)

    if "meta" not in file:
        meta = file.create_group("meta")
    else:
        meta = file["meta"]

    groups = sorted([i for i in meta])[::-1]
    for existing in groups:
        if sorted([i for i in meta[existing].attrs]) == ["compiler", "dependencies"]:
            a = sorted(meta[existing].attrs["compiler"]) == compiler
            b = sorted(meta[existing].attrs["dependencies"]) == deps
            if a and b:
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


def default_chunks(shape):
    if len(shape) == 1:
        n = shape[0]
        if n % 2 == 0:
            if n < 128:
                return (1, n)
            else:
                m = n // 128
                return (1, int(n / m))
    elif len(shape) == 2:
        n = shape[1]
        if shape[0] != n:
            return True
        if n % 2 == 0:
            if n <= 8:
                return (1, n, n)
            elif n == 16:
                return (1, 8, 16)
            elif n == 32:
                return (1, 4, 32)
            elif n == 64:
                return (1, 2, 64)
            elif n.bit_count() == 1:
                return (1, 1, 128)
            elif n < 128:
                return (1, 1, n)
            else:
                m = n // 128
                return (1, 1, int(n / m))

    return True
