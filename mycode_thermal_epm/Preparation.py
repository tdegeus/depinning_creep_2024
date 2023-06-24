import argparse
import inspect
import os
import pathlib
import shutil
import tempfile
import textwrap

import GooseEPM as epm
import GooseHDF5 as g5
import h5py
import numpy as np
import shelephant
import tqdm

from . import storage
from . import tag
from . import tools
from ._version import version

data_version = "2.0"
m_exclude = ["AQS", "Extremal", "ExtremalAvalanche", "Thermal"]


def propagator(param: h5py.Group):
    if param["interactions"].asstr()[...] == "monotonic-shortrange":
        return epm.laplace_propagator()
    raise NotImplementedError("Unknown interactions type '%s'" % param["interactions"])


def get_data_version(file: h5py.File) -> str:
    """
    Read the current data version from the file.

    :param file: Opened file.
    :return: Data version.
    """
    if "/param/data_version" in file:
        return str(file["/param/data_version"].asstr()[...])
    return "0.0"


def _libname_pre_v2(libs: list[str]) -> list[str]:
    """
    Rename library names from data_version == 2.0 to those used in data_version < 2.0.

    :param libs: List of new library names.
    :return: Corresponding list of old library names.
    """
    rename = {
        "Preparation": "AthermalPreparation",
        "AQS": "AthermalQuasiStatic",
        "Extremal": "ExtremeValue",
        "ExtremalAvalanche": "ExtremeValue/Avalanche",
        "Thermal": "Thermal",
    }
    return [rename[i] for i in libs]


def _copy_metadata_pre_v2(src: h5py.File, dst: h5py.File) -> None:
    """
    Copy and rename metadata from data_version < 2.0 to data_version == 2.0.

    :param src: Source file.
    :param dst: Destination file.
    """
    a = g5.getdatapaths(src, root="/meta")
    b = []
    for path in a:
        _, meta, lib, func = path.split("/")
        if lib == "AthermalPreparation":
            b.append(g5.join(meta, "Preparation", func, root=True))
        elif lib == "AthermalQuasiStatic":
            b.append(g5.join(meta, "AQS", func, root=True))
        elif path == "/meta/ExtremeValue/RunAvalanche":
            b.append("/meta/ExtremalAvalanche/Run")
        elif path == "/meta/ExtremeValue/BranchRun":
            b.append("/meta/ExtremalAvalanche/BranchExtremal")
        elif lib == "ExtremeValue":
            b.append(g5.join(meta, "Extremal", func, root=True))
        else:
            b.append(path)
    g5.copy(src, dst, a, b)
    storage.dump_overwrite(dst, "/param/data_version", data_version)


def _upgrade_data(filename: pathlib.Path, temp_dir: pathlib.Path) -> bool:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :return: ``temp_file`` if the data is upgraded, ``None`` otherwise.
    """
    with h5py.File(filename) as src:
        assert not any(x in src for x in m_exclude + _libname_pre_v2(m_exclude))
        ver = get_data_version(src)

    if tag.greater_equal(ver, "2.0"):
        return None

    temp_file = temp_dir / "from_older.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        g5.copy(src, dst, ["/param", "/init"])
        _copy_metadata_pre_v2(src, dst)

    return temp_file


def UpgradeData(cli_args=None, upgrade_function=_upgrade_data):
    r"""
    Upgrade data to the current version.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--no-bak", action="store_true", help="Do not backup before modifying")
    parser.add_argument("files", type=pathlib.Path, nargs="*", help="File (overwritten)")

    args = tools._parse(parser, cli_args)
    assert all([f.is_file() for f in args.files]), "File not found"
    assert args.no_bak or not any([(f.parent / (f.name + ".bak")).exists() for f in args.files])
    assert args.develop or not tag.has_uncommitted(version), "Uncommitted changes"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        pbar = tqdm.tqdm(args.files)
        for filename in pbar:
            pbar.set_description(str(filename))
            pbar.refresh()
            temp_file = upgrade_function(filename, temp_dir)
            if temp_file is None:
                continue
            if not args.no_bak:
                bakname = filename.parent / (filename.name + ".bak")
                shutil.copy2(filename, bakname)
            shutil.copy2(temp_file, filename)


class SystemStressControl(epm.SystemStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        init = file["init"]
        epm.SystemStressControl.__init__(
            self,
            *propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * init["sigmay"].attrs["mean"],
            sigmay_std=np.ones(param["shape"][...]) * init["sigmay"].attrs["std"],
            seed=init["state"].attrs["seed"],
            alpha=param["alpha"][...],
            random_stress=True,
        )


def Generate(cli_args=None):
    """
    Generate IO file of the following structure::

        |-- param   # ensemble parameters
        |   |-- alpha
        |   |-- shape
        |   `-- interactions
        `-- init    # initial realisation -> use "Run" to fill
            |-- sigma
            |-- sigmay
            `-- state

    .. note::

        You can take `t` and `epsp` equal to zero.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
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
    )
    parser.add_argument("--all", action="store_true", help="Generate a suggestion of runs")

    parser.add_argument("outdir", type=pathlib.Path, help="Output directory")

    args = tools._parse(parser, cli_args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n = np.prod(args.shape)
    assert not any(
        [
            (args.outdir / f"id={i:04d}.h5").exists()
            for i in range(args.start, args.start + args.nsim)
        ]
    )
    files = []
    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n
        with h5py.File(args.outdir / files[-1], "w") as file:
            tools.create_check_meta(file, f"/meta/Preparation/{funcname}", dev=args.develop)

            param = file.create_group("param")
            param["alpha"] = args.alpha
            param["shape"] = args.shape
            param["interactions"] = args.interactions
            param["data_version"] = data_version

            init = file.create_group("init")
            init.create_dataset("sigma", shape=args.shape, dtype=np.float64)

            init.create_dataset("sigmay", shape=args.shape, dtype=np.float64)
            init["sigmay"].attrs["mean"] = 1.0
            init["sigmay"].attrs["std"] = 0.3

            init.create_dataset("state", shape=[], dtype=np.uint64)
            init["state"].attrs["seed"] = seed

    exec = "Preparation_Run"
    commands = [f"{exec} {f}" for f in files]
    shelephant.yaml.dump(args.outdir / "commands_run.yaml", commands, force=True)

    if not args.all:
        return

    assert len(os.path.split(args.outdir)) > 1

    for name in ["AQS", "Extremal"]:
        base = args.outdir / ".." / name
        base.mkdir(parents=True, exist_ok=True)

        exec = f"{name}_BranchPreparation"
        commands = [f"{exec} ../{args.outdir.name}/{f} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exec = f"{name}_Run"
        if name == "Extremal":
            commands = [f"{exec} -n 100 {f}" for f in files]
        else:
            commands = [f"{exec} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)

    name = "Thermal"
    temperatures = {
        "temperature=0,005": 0.005,
        "temperature=0,007": 0.007,
        "temperature=0,01": 0.01,
        "temperature=0,02": 0.02,
        "temperature=0,03": 0.03,
        "temperature=0,05": 0.05,
        "temperature=0,07": 0.07,
        "temperature=0,1": 0.1,
        "temperature=0,2": 0.2,
        "temperature=0,5": 0.5,
    }
    for key, temp in temperatures.items():
        base = args.outdir / ".." / name / key
        base.mkdir(parents=True, exist_ok=True)

        exec = f"{name}_BranchPreparation"
        commands = [f"{exec} ../../{args.outdir.name}/{f} {f} --temperature {temp}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exec = f"{name}_Run"
        commands = [f"{exec} -n 100 {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)


def Run(cli_args=None):
    """
    Initialize system, and store state.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/Preparation/{funcname}", dev=args.develop)
        system = SystemStressControl(file)
        init = file["init"]
        init["sigma"][...] = system.sigma
        init["sigmay"][...] = system.sigmay
        init["state"][...] = system.state
