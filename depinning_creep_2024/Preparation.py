from __future__ import annotations

import argparse
import inspect
import logging
import os
import pathlib
import re
import shutil
import tempfile
from datetime import datetime

import enstat
import GooseEPM as epm
import GooseHDF5 as g5
import h5py
import numpy as np
import shelephant
import tqdm
from numpy.typing import ArrayLike

from . import storage
from . import tag
from . import tools
from ._version import version
from .tools import MyFmt

data_version = "3.0"
m_name = "Preparation"
m_exclude = ["AQS", "Extremal", "Thermal"]


def convert_A_to_ell(A: np.ndarray, dim: int) -> np.ndarray:
    """
    Convert number of blocks that yielded at least once to a proxy for the linear extent.
    :param A: Number of blocks that yielded at least once.
    :param dim: Dimension.
    """
    if dim == 1:
        return A
    if dim == 2:
        return np.sqrt(A)
    raise NotImplementedError


def propagator(group: h5py.Group) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Return the propagator and the distances.

    :param group: Opened group ``"param"`` (read ``"interactions"``).
    :return: Propagator and distances.
    """
    if group["interactions"].asstr()[...] == "monotonic-shortrange":
        return epm.laplace_propagator()
    raise NotImplementedError(f"Unknown interactions type '{group['interactions']}'")


def get_dynamics(group: h5py.Group) -> str:
    """
    Read the dynamics from the file.

    :param group: Opened group ``"param"`` (read ``"dynamics"``).
    :return: Dynamics (``"default"`` or ``"depinning"``)
    """
    if "dynamics" in group:
        return group["dynamics"].asstr()[...]
    return "default"


def allocate_System(
    group: h5py.Group,
    random_stress: bool = True,
    loading: str = "stress",
    thermal: bool = False,
    **kwargs,
) -> epm.SystemClass:
    """
    Allocate the system.

    .. note::

        You need to additionally restore a state if so desired.

    :param group: Opened group ``"param"``.
    :param random_stress: ``True`` if a random, compatible, stress should be generated.
    :param loading: Loading type.
    :param thermal: Simulate at finite temperature.
    :return: Allocated system.
    """
    prop, dist = propagator(group)
    opts = dict(
        rules="default",
        loading=loading,
        thermal=thermal,
        propagator=prop,
        distances=dist,
        sigmay_mean=np.ones(group["shape"][...]) * group["sigmay"][0],
        sigmay_std=np.ones(group["shape"][...]) * group["sigmay"][1],
        seed=group["seed"][...],
        alpha=group["alpha"][...],
        random_stress=random_stress,
    )

    if get_dynamics(group) == "depinning":
        opts["rules"] = "depinning"
        opts.pop("sigmay_mean")
        assert np.isclose(group["sigmay"][0], 0)

    return epm.allocate_System(**opts, **kwargs)


def compute_x(dynamics: str, sigma: ArrayLike, sigmay: ArrayLike) -> np.ndarray:
    """
    Compute the distance to yielding.

    :param dynamics: Dynamics (``"default"`` or ``"depinning"``)
    :param sigma: Stresses.
    :param sigmay: Yield stresses.
    :return: Distance to yielding.
    """
    if dynamics == "default":
        return sigmay - np.abs(sigma)

    if dynamics == "depinning":
        return sigmay - sigma

    raise NotImplementedError(f"Unknown dynamics '{dynamics}'")


def store_histogram(group: h5py.Group, hist: enstat.histogram) -> None:
    """
    Store the histogram.

    :param group: Data group.
    :param hist: Histogram.
    """
    storage.dump_overwrite(group, "bin_edges", hist.bin_edges)
    storage.dump_overwrite(group, "count", hist.count)
    storage.dump_overwrite(group, "count_left", hist.count_left)
    storage.dump_overwrite(group, "count_right", hist.count_right)


def get_data_version(file: h5py.File) -> str:
    """
    Read the current data version from the file.

    :param file: Opened file.
    :return: Data version.
    """
    if "/param/data_version" in file:
        return str(file["/param/data_version"].asstr()[...])
    return "0.0"


def check_copy(
    src: h5py.File,
    dst: h5py.File,
    rename: list = None,
    shallow: bool = True,
    attrs: bool = True,
    allow: dict = {},
) -> None:
    """
    Check that all datasets in ``src`` are also in ``dst``
    (check that ``"!="`` and ``"->"`` are empty).

    :param src: Source file.
    :param dst: Destination file.
    :param rename: List of renamed datasets.
    :param shallow: Do not check the contents of the datasets.
    :param attrs: Check (existence of) attributes.
    :param allow: List of datasets (regex) that are allowed to be different.
    """
    if rename is None:
        diff = g5.compare(src, dst, shallow=shallow)
    else:
        diff, diff_a, diff_b = g5.compare_rename(
            src, dst, rename=rename, regex=True, shallow=shallow, attrs=attrs, only_datasets=False
        )
        for key in diff_a:
            diff[key] += diff_a[key]
        for key in diff_b:
            diff[key] += diff_b[key]

    for key in allow:
        for item in allow[key]:
            diff[key] = [x for x in diff[key] if not re.match(item, x)]

    for key in ["!=", "->"]:
        assert len(diff[key]) == 0, f"Key '{key}' not empty: {diff[key]}"


def _upgrade_data(filename: pathlib.Path, temp_dir: pathlib.Path) -> pathlib.Path | None:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :return: New file in temporary directory if the data is upgraded, ``None`` otherwise.
    """
    with h5py.File(filename) as src:
        assert not any(x in src for x in m_exclude)
        ver = get_data_version(src)
        assert tag.greater_equal(ver, "2.0")

    if tag.greater_equal(ver, "3.0"):
        return None

    temp_file = temp_dir / "new_file.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        g5.copy(src, dst, ["/meta", "/param"])
        dst["/param/seed"] = src["/init/state"].attrs["seed"]
        dst["/param/data_version"][...] = data_version

        if "init" in src:
            dst["/param/sigmay"] = [
                src["init"]["sigmay"].attrs["mean"],
                src["init"]["sigmay"].attrs["std"],
            ]
        else:
            dst["/param/sigmay"] = (
                [0.0, 1.0] if get_dynamics(dst["param"]) == "depinning" else [1.0, 0.3]
            )

        if "init" in src:
            system = allocate_System(dst["param"], random_stress=False)
            system.sigma = src["init"]["sigma"][...]
            system.sigmay = src["init"]["sigmay"][...]
            system.state = src["init"]["state"][...]
            system.epsp = (
                src["init"]["epsp"][...] if "epsp" in src["init"] else np.zeros_like(system.epsp)
            )
            system.t = src["init"]["t"][...] if "t" in src["init"] else 0.0
        else:
            system = allocate_System(dst["param"])
            system.epsp = np.zeros_like(system.epsp)
            system.t = 0.0

        dump_snapshot(0, dst.create_group(m_name).create_group("snapshots"), system)
        check_copy(src, dst, rename=[["/init", "/Preparation/snapshots"]], attrs=False)

    return _upgrade_metadata(temp_file, temp_dir)


def _upgrade_metadata(filename: pathlib.Path, temp_dir: pathlib.Path) -> pathlib.Path | None:
    """
    Upgrade storage of metadata: encode a timestamp in the key.

    .. note::

        Since the original timestamp cannot be recovered, the timestamp is set to the current time.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :return: The old/new file name.
    """
    with h5py.File(filename) as src:
        if "meta" not in src:
            return filename

    temp_file = temp_dir / "new_meta.h5"
    assert temp_file != filename
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        groups = [i for i in src]
        if "meta" in groups:
            groups.remove("meta")
        g5.copy(src, dst, groups)

        known = [
            "Preparation/Generate",
            "Preparation/Run",
            "Extremal/BranchPreparation",
            "Extremal/Run",
            "ExtremalAvalanche/BranchExtremal",
            "ExtremalAvalanche/Run",
            "Thermal/BranchPreparation",
            "Thermal/Run",
            "AQS/BranchPreparation",
            "AQS/Run",
        ]

        in_src = [i.split("/meta/")[1] for i in g5.getdatapaths(src, root="/meta")]
        dst.create_group("meta")
        deps = []
        comp = []
        link = None

        for key in known:
            if key in in_src:
                d = src[f"/meta/{key}"]
                if isinstance(d, h5py.Group):
                    if sorted([i for i in d.attrs]) == ["compiler", "dependencies", "uuid"]:
                        a = list(d.attrs["compiler"]) == comp
                        b = list(d.attrs["dependencies"]) == deps
                        if a and b:
                            dst[tools.path_meta(*key.split("/"))] = dst[f"/meta/{link}"]
                            continue
                        else:
                            name = tools.path_meta(*key.split("/")).split("/")[2]
                            deps = list(d.attrs["dependencies"])
                            comp = list(d.attrs["compiler"])
                            link = name
                            g = dst["meta"].create_group(name)
                            g.attrs["compiler"] = comp
                            g.attrs["dependencies"] = deps
                            continue

                g5.copy(src, dst, f"/meta/{key}", tools.path_meta(*key.split("/")))

        for key in in_src:
            if key not in known:
                logging.warning(f"Unknown metadata key: {key}", RuntimeWarning)
                stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
                g5.copy(src, dst, f"/meta/{key}", f"/meta/{stamp}_{key}")

    return temp_file


@tools.docstring_append_cli()
def UpgradeData(
    cli_args: list = None,
    _return_parser: bool = False,
    _module: str = m_name,
    _upgrade_function=_upgrade_data,
    _combine: bool = False,
) -> None:
    """
    Upgrade data to the current version.

    :param _module: Name of the module calling this function.
    :param _upgrade_function: Function that upgrades the data.
    :param _combine:
        Combine two data-files.

        -   If ``_combine=False``::

                temp_file = _upgrade_function(filename, temp_dir)

        -   If ``_combine=True``::

                temp_file = _upgrade_function(filename, temp_dir, args.insert)

        The return value ``temp_file`` is then handled as follows:

        -   If ``ret is None``: The file is not upgraded.
        -   Otherwise a backup is created, and the file is replaced by ``temp_file``.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--no-bak", action="store_true", help="Do not backup before modifying")
    if _combine:
        parser.add_argument("--insert", type=pathlib.Path, help="Extra data to insert")
        parser.add_argument("file", type=pathlib.Path, help="File (overwritten)")
    else:
        parser.add_argument("files", type=pathlib.Path, nargs="*", help="File (overwritten)")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    if _combine:
        files = [args.file]
    else:
        files = args.files

    assert all([f.is_file() for f in files]), "File not found"
    assert args.no_bak or not any([(f.parent / (f.name + ".bak")).exists() for f in files])
    assert args.develop or not tag.has_uncommitted(version), "Uncommitted changes"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        pbar = tqdm.tqdm(files)
        for filename in pbar:
            pbar.set_description(str(filename))
            pbar.refresh()

            if _combine:
                temp_file = _upgrade_function(filename, temp_dir, args.insert)
            else:
                temp_file = _upgrade_function(filename, temp_dir)

            if temp_file is None:
                continue

            with h5py.File(temp_file, "a") as file:
                tools.create_check_meta(file, tools.path_meta(_module, funcname), dev=args.develop)

            if not args.no_bak:
                bakname = filename.parent / (filename.name + ".bak")
                shutil.copy2(filename, bakname)
            shutil.copy2(temp_file, filename)


@tools.docstring_append_cli()
def VerifyData(cli_args: list = None, _return_parser: bool = False) -> None:
    """
    Check that the data is of the correct version.
    Filenames of incorrect files are printed to stdout.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("files", type=pathlib.Path, nargs="*", help="File (overwritten)")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert all([f.is_file() for f in args.files]), "File not found"
    assert args.develop or not tag.has_uncommitted(version), "Uncommitted changes"

    ret = []
    pbar = tqdm.tqdm(args.files)
    for filename in pbar:
        pbar.set_description(str(filename))
        pbar.refresh()
        with h5py.File(filename) as file:
            if get_data_version(file) != data_version:
                ret.append(filename)
    print("\n".join(list(map(str, ret))))


def dump_snapshot(index: int, group: h5py.Group, system: epm.SystemClass):
    """
    Add/overwrite snapshot of the current state (fully recoverable).

    .. note::

        The snapshot is added to a multi-dimensional dataset containing all snapshots.

    :param index: Index of the snapshot to overwrite.
    :param group: Group to store the snapshot in.
    :param system: System.
    """
    kwargs = dict(shape=system.shape, dtype=np.float64, chunks=tools.default_chunks(system.shape))
    with g5.ExtendableSlice(group, "epsp", **kwargs) as dset:
        dset[index] = system.epsp
    with g5.ExtendableSlice(group, "sigma", **kwargs) as dset:
        dset[index] = system.sigma
    with g5.ExtendableSlice(group, "sigmay", **kwargs) as dset:
        dset[index] = system.sigmay
    with g5.ExtendableList(group, "state", np.uint64, chunks=(16,)) as dset:
        dset[index] = system.state
    with g5.ExtendableList(group, "t", np.float64, chunks=(16,)) as dset:
        dset[index] = system.t


def load_snapshot(index: int, group: h5py.Group, system: epm.SystemClass) -> epm.SystemClass:
    """
    Recover system from a snapshot.

    :param index: Index of the snapshot to read (``None`` if snapshots are plain datasets).
    :param group: Group of the snapshots.
    :param system: System (modified).
    :return: System (modified).
    """
    if index is None:
        system.epsp = group["epsp"][...]
        system.sigma = group["sigma"][...]
        system.sigmay = group["sigmay"][...]
        system.t = group["t"][...]
        system.state = group["state"][...]
        return system

    system.epsp = group["epsp"][index, ...]
    system.sigma = group["sigma"][index, ...]
    system.sigmay = group["sigmay"][index, ...]
    system.t = group["t"][index]
    system.state = group["state"][index]
    return system


def Generate(cli_args: list = None, _return_parser: bool = False) -> None:
    """
    Generate IO files, and compute and write initial states.
    In addition, write common simulation files.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)

    parser.add_argument("--shape", type=int, nargs=2, help="System shape")
    parser.add_argument("--alpha", type=float, default=1.5, help="Potential type")
    parser.add_argument(
        "--interactions",
        type=str,
        choices=["monotonic-shortrange", "monotonic-longrange", "eshelby"],
        help="Interaction type",
        default="monotonic-shortrange",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        choices=["default", "depinning"],
        help="Dynamics",
    )
    parser.add_argument("--all", action="store_true", help="Generate a suggestion of runs")
    parser.add_argument("outdir", type=pathlib.Path, help="Output directory")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = [args.outdir / f"id={i:04d}.h5" for i in range(args.start, args.start + args.nsim)]
    assert not any([i.exists() for i in files])

    n = np.prod(args.shape)
    files = []
    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n
        with h5py.File(args.outdir / files[-1], "w") as file:
            tools.create_check_meta(file, tools.path_meta(m_name, funcname), dev=args.develop)
            param = file.create_group("param")
            param["alpha"] = args.alpha
            param["shape"] = args.shape
            param["dynamics"] = args.dynamics
            param["interactions"] = args.interactions
            param["data_version"] = data_version
            param["sigmay"] = [0.0, 1.0] if args.dynamics == "depinning" else [1.0, 0.3]
            param["seed"] = seed

            system = allocate_System(file["param"])
            system.epsp = np.zeros_like(system.epsp)
            system.t = 0.0
            dump_snapshot(0, file.create_group(m_name).create_group("snapshots"), system)

    if not args.all:
        return

    assert len(os.path.split(args.outdir)) > 1

    for name in ["AQS", "Extremal"]:
        base = args.outdir / ".." / name
        base.mkdir(parents=True, exist_ok=True)

        exe = f"{name}_BranchPreparation"
        commands = [f"{exe} ../{args.outdir.name}/{f} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exe = f"{name}_Run"
        if name == "Extremal":
            commands = [f"{exe} -n 112 {f}" for f in files]
        else:
            commands = [f"{exe} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)

    name = "Thermal"
    temperatures = {
        "temperature=0,002": 0.002,
        "temperature=0,003": 0.003,
        "temperature=0,005": 0.005,
        "temperature=0,007": 0.007,
        "temperature=0,01": 0.01,
        "temperature=0,02": 0.02,
        "temperature=0,03": 0.03,
        "temperature=0,05": 0.05,
        "temperature=0,07": 0.07,
        "temperature=0,1": 0.1,
        "temperature=0,2": 0.2,
        "temperature=0,3": 0.3,
        "temperature=0,5": 0.5,
        "temperature=0,7": 0.7,
    }
    for key, temp in temperatures.items():
        base = args.outdir / ".." / name / key
        base.mkdir(parents=True, exist_ok=True)

        exe = f"{name}_BranchPreparation"
        commands = [f"{exe} ../../{args.outdir.name}/{f} {f} --temperature {temp}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exe = f"{name}_Run"
        commands = [f"{exe} -n 112 {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)


# <autodoc> generated by docs/conf.py


def _Generate_parser() -> argparse.ArgumentParser:
    return Generate(_return_parser=True)


def _UpgradeData_parser() -> argparse.ArgumentParser:
    return UpgradeData(_return_parser=True)


def _VerifyData_parser() -> argparse.ArgumentParser:
    return VerifyData(_return_parser=True)


# </autodoc>
