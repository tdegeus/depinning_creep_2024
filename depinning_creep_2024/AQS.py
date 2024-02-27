"""
Athermal Quasistatic Shear (AQS) simulations.
"""

import argparse
import inspect
import pathlib
import sys
import time

import enstat
import GooseEPM as epm
import GooseHDF5 as g5
import GooseSLURM as slurm
import h5py
import numpy as np
import tqdm

from . import Preparation
from . import tag
from . import tools
from ._version import version
from .Preparation import data_version
from .tools import MyFmt

f_info = "EnsembleInfo.h5"
m_name = "AQS"
m_exclude = ["Extremal", "ExtremalAvalanche", "Thermal"]


def _upgrade_data(filename: pathlib.Path, temp_dir: pathlib.Path) -> bool:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :return: ``temp_file`` if the data is upgraded, ``None`` otherwise.
    """
    with h5py.File(filename) as src:
        assert not any(x in src for x in m_exclude)
        ver = Preparation.get_data_version(src)

    assert tag.equal(ver, "2.0")

    with h5py.File(filename) as src:
        if m_name not in src:
            return None

    temp_file = temp_dir / "new_file.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        g5.copy(src, dst, ["/meta", "/param"])
        dst["/param/seed"] = src["/restart/state"].attrs["seed"]
        dst["/param/data_version"][...] = data_version

        rename = [
            ["/AQS/A", "/AQS/data/A"],
            ["/AQS/S", "/AQS/data/S"],
            ["/AQS/T", "/AQS/data/T"],
            ["/AQS/sigma", "/AQS/data/sigma"],
            ["/AQS/uframe", "/AQS/data/uframe"],
            ["/AQS/restore", "/AQS/snapshots"],
        ]
        for entry in rename:
            g5.copy(src, dst, *entry)

        with g5.ExtendableList(dst["/AQS/snapshots"], "systemspanning", bool, chunks=(16,)) as dset:
            dset.append(np.ones(dst["/AQS/snapshots/step"].size, dtype=bool))

    return Preparation._upgrade_metadata(temp_file, temp_dir)


@tools.docstring_append_cli()
def UpgradeData(cli_args: list = None, _return_parser: bool = False):
    """
    Upgrade data to the current version.
    """
    return Preparation.UpgradeData(
        cli_args=cli_args,
        _return_parser=_return_parser,
        _module=m_name,
        _upgrade_function=_upgrade_data,
    )


def allocate_System(file: h5py.File, index: int) -> epm.SystemClass:
    """
    Allocate the system, and restore snapshot.

    :param param: Opened file.
    :param index: Index of the snapshot to load.
    :return: System.
    """
    system = Preparation.allocate_System(
        group=file["param"],
        random_stress=False,
        thermal=False,
        loading="spring",
        kframe=file["param"]["kframe"][...],
    )
    system = Preparation.load_snapshot(index, file[m_name]["snapshots"], system)
    system.epsframe = file[m_name]["snapshots"]["uframe"][index]
    return system


def dump_snapshot(
    index: int, group: h5py.Group, system: epm.SystemClass, step: int, systemspanning: bool
) -> None:
    """
    Add/overwrite snapshot of the current state (fully recoverable).

    :param index: Index of the snapshot to overwrite.
    :param group: Group to store the snapshot in.
    :param system: System.
    :param step: Current AQS step.
    :param systemspanning: If snapshot follows a system-spanning event.
    """
    Preparation.dump_snapshot(index, group, system)

    with g5.ExtendableList(group, "uframe") as dset:
        dset[index] = system.epsframe

    with g5.ExtendableList(group, "step") as dset:
        dset[index] = step

    with g5.ExtendableList(group, "systemspanning") as dset:
        dset[index] = systemspanning


@tools.docstring_append_cli()
def BranchPreparation(cli_args: list = None, _return_parser: bool = False):
    """
    Branch from prepared stress state and add parameters.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--kframe", type=float, help="Frame stiffness (default 1 / L**2)")
    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(m in src for m in m_exclude), "Wrong file type"
        g5.copy(src, dest, ["/meta", "/param"])
        tools.create_check_meta(dest, tools.path_meta(m_name, funcname), dev=args.develop)

        g5.copy(src, dest, f"/{Preparation.m_name}/snapshots", f"/{m_name}/snapshots")
        group = dest[m_name]["snapshots"]
        g5.ExtendableList(group, "step", np.int64, chunks=(16,)).setitem(index=0, data=0).flush()
        g5.ExtendableList(group, "uframe", np.float64, chunks=(16,)).setitem(
            index=0, data=0
        ).flush()
        g5.ExtendableList(group, "systemspanning", bool, chunks=(16,)).setitem(
            index=0, data=True
        ).flush()

        if args.kframe is not None:
            dest["param"]["kframe"] = args.kframe
        else:
            dest["param"]["kframe"] = 1.0 / (np.min(dest["param"]["shape"][...]) ** 2)

        group = dest[m_name].create_group("data")
        with g5.ExtendableList(group, "uframe", np.float64, chunks=(16,)) as dset:
            dset[0] = 0
        with g5.ExtendableList(group, "sigma", np.float64, chunks=(16,)) as dset:
            dset[0] = 0
        with g5.ExtendableList(group, "T", np.float64, chunks=(16,)) as dset:
            dset[0] = 0
        with g5.ExtendableList(group, "S", np.int64, chunks=(16,)) as dset:
            dset[0] = 0
        with g5.ExtendableList(group, "A", np.int64, chunks=(16,)) as dset:
            dset[0] = 0


def Run(cli_args: list = None, _return_parser: bool = False) -> None:
    """
    Run simulation for a fixed number of steps.

    -   Write global output per step (``uframe``, ``sigma``, ``T``, ``S``, ``A``).

    -   Write snapshot every time a system-spanning event occurs.
        Added temporary snapshots based on ``--buffer`` to be able to restart.
    """
    tic = time.time()
    ticb = time.time()

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument(
        "-n",
        "--nstep",
        type=int,
        default=312 * 16,
        help="Total #load-steps to run (save storage by using a multiple of 16)",
    )
    parser.add_argument(
        "--buffer",
        type=slurm.duration.asSeconds,
        default=30 * 60,
        help="Write interval to write partial (restartable) data",
    )
    parser.add_argument(
        "--walltime",
        type=slurm.duration.asSeconds,
        default=sys.maxsize,
        help="Walltime at which to stop",
    )
    parser.add_argument(
        "--save-duration",
        type=slurm.duration.asSeconds,
        default=0,
        help="Duration to reserve for saving data",
    )
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    args.walltime -= args.save_duration
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, tools.path_meta(m_name, funcname), dev=args.develop)

        data = file[m_name]["data"]
        snap = file[m_name]["snapshots"]
        index_snapshot = snap["state"].size - 1
        system = allocate_System(file, index_snapshot)
        start = snap["step"][index_snapshot] + 1
        if snap["systemspanning"][index_snapshot]:
            index_snapshot += 1
        # if output/restore fields exceed restart step; remove excess
        for key in ["uframe", "sigma", "T", "S", "A"]:
            data[key].resize((start,))

        pbar = tqdm.tqdm(range(start, args.nstep), desc=str(args.file))
        duration = enstat.scalar()

        for step in pbar:
            pbar.refresh()
            n = np.copy(system.nfails)
            t0 = system.t

            # event-driven protocol
            if step % 2 == 1:
                system.shiftImposedShear()  # leaves >= 1 block unstable
            else:
                if duration.mean() > args.walltime - (time.time() - tic):
                    return
                tici = time.time()
                system.relaxAthermal()  # leaves 0 blocks unstable
                duration += time.time() - tici

            if time.time() - tic >= args.walltime:
                return

            # global output
            with g5.ExtendableList(data, "uframe") as dset:
                dset[step] = system.epsframe
            with g5.ExtendableList(data, "sigma") as dset:
                dset[step] = system.sigmabar
            with g5.ExtendableList(data, "T") as dset:
                dset[step] = system.t - t0
            with g5.ExtendableList(data, "S") as dset:
                dset[step] = np.sum(system.nfails - n)
            with g5.ExtendableList(data, "A") as dset:
                dset[step] = np.sum(system.nfails != n)

            # full state
            if np.sum(system.nfails != n) == system.size:
                dump_snapshot(index_snapshot, snap, system, step, True)
                index_snapshot += 1
            elif step == args.nstep - 1 or time.time() - ticb > args.buffer:
                dump_snapshot(index_snapshot, snap, system, step, False)
                ticb = time.time()


def _norm_uframe(file: h5py.File) -> float:
    """
    Return the normalisation of uframe.

    :param file: File to read from.
    :return: Normalisation of uframe.
    """
    kframe = file["/param/kframe"][...]
    return (kframe + 1.0) / kframe  # mu = 1


def _steady_state(file: h5py.File) -> int:
    """
    Return the first steady-state step.

    :param file: File to read from.
    :return: First steady-state step.
    """

    kframe = file["/param/kframe"][...]
    u0 = (kframe + 1.0) / kframe  # mu = 1

    res = file[m_name]["data"]
    uframe = res["uframe"][...] / u0
    sigma = res["sigma"][...]  # mu, mean(sigmay) = 1

    tmp = uframe[0]
    uframe[0] = 1
    tangent = sigma / uframe
    tangent[0] = np.inf
    uframe[0] = tmp
    test = tangent < 0.95
    if not np.any(test):
        return len(test)
    return np.argmax(tangent < 0.95) + 1


@tools.docstring_append_cli()
def EnsembleInfo(cli_args: list = None, _return_parser: bool = False) -> None:
    """
    Basic interpretation of the ensemble.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_info)
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, f"/meta/{m_name}/{funcname}", dev=args.develop)
        elastic = {"uframe": [], "sigma": [], "S": [], "A": [], "ell": [], "T": []}
        plastic = {"uframe": [], "sigma": [], "S": [], "A": [], "ell": [], "T": []}
        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    u0 = _norm_uframe(file)
                    output["/norm/uframe"] = u0
                    output["/norm/sigma"] = 1.0
                    pdfx = enstat.histogram(bin_edges=np.linspace(0, 3, 2001), bound_error="norm")

                if m_name not in file:
                    assert not any(m in file for m in m_exclude), "Wrong file type"
                    continue

                res = file[m_name]["data"]
                uframe = res["uframe"][...] / u0
                sigma = res["sigma"][...]
                i = _steady_state(file)
                if i % 2 == 0:
                    i += 1
                if (uframe.size - i) % 2 == 1:
                    end = -1
                else:
                    end = None

                # copy loading/avalanche data from the 'steady state'
                elastic["uframe"] += uframe[i:end:2].tolist()
                elastic["sigma"] += sigma[i:end:2].tolist()
                plastic["uframe"] += uframe[i + 1 : end : 2].tolist()
                plastic["sigma"] += sigma[i + 1 : end : 2].tolist()
                plastic["S"] += res["S"][i + 1 : end : 2].tolist()
                plastic["A"] += res["A"][i + 1 : end : 2].tolist()
                plastic["ell"] += np.sqrt(res["A"][i + 1 : end : 2]).tolist()
                plastic["T"] += res["T"][i + 1 : end : 2].tolist()

                if file[m_name]["snapshots"]["state"].size > 1:
                    pdfx += Preparation.compute_x(
                        dynamics=Preparation.get_dynamics(file),
                        sigma=file[m_name]["snapshots"]["sigma"][1:, ...],
                        sigmay=file[m_name]["snapshots"]["sigmay"][1:, ...],
                    ).ravel()

        for key in elastic:
            output["/elastic/" + key] = elastic[key]
        for key in plastic:
            output["/plastic/" + key] = plastic[key]

        Preparation.store_histogram(output.create_group("hist_x"), pdfx)


@tools.docstring_append_cli()
def Plot(cli_args: list = None, _return_parser: bool = False) -> None:
    """
    Basic of the ensemble.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="Simulation file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    with h5py.File(args.file) as file:
        i = _steady_state(file)
        res = file[m_name]["data"]
        S = res["S"][i:].tolist()
        if "/restore/sigma" in res:
            x = Preparation.get_x(file, res["restore"])
        else:
            x = None
        uframe = res["uframe"][...] / _norm_uframe(file)
        sigma = res["sigma"][...]

    S = np.array(S)
    S = S[S > 0]

    fig, axes = gplt.subplots(ncols=3)

    ax = axes[0]
    ax.plot(uframe, sigma, marker=".")
    if i < uframe.size:
        ax.axvline(uframe[i], ls="-", c="r", lw=1)
    ax.set_xlabel(r"$\bar{u}$")
    ax.set_ylabel(r"$\bar{\sigma}$")

    ax = axes[1]
    if len(S) > 0:
        pdfs = enstat.histogram.from_data(S, bins=100, mode="log")
        ax.plot(pdfs.x, pdfs.p)
        keep = np.logical_and(pdfs.x > 1e1, pdfs.x < 1e5)
        gplt.fit_powerlaw(pdfs.x[keep], pdfs.p[keep], axis=ax, auto_fmt="S")
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$P(S)$")

    ax = axes[2]
    if x is not None:
        pdfx = enstat.histogram.from_data(x, bins=100, mode="log")
        ax.plot(pdfx.x, pdfx.p)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")

    plt.show()
    plt.close(fig)


# <autodoc> generated by docs/conf.py


def _BranchPreparation_parser() -> argparse.ArgumentParser:
    return BranchPreparation(_return_parser=True)


def _EnsembleInfo_parser() -> argparse.ArgumentParser:
    return EnsembleInfo(_return_parser=True)


def _Plot_parser() -> argparse.ArgumentParser:
    return Plot(_return_parser=True)


def _Run_parser() -> argparse.ArgumentParser:
    return Run(_return_parser=True)


def _UpgradeData_parser() -> argparse.ArgumentParser:
    return UpgradeData(_return_parser=True)


# </autodoc>
