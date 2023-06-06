import argparse
import inspect
import pathlib
import textwrap
import time

import enstat
import GooseEPM as epm
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import AthermalPreparation
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"
m_name = "AthermalQuasiStatic"


class SystemSpringLoading(epm.SystemSpringLoading):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemSpringLoading.__init__(
            self,
            *AthermalPreparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            kframe=param["kframe"][...],
            random_stress=False,
        )

        self.epsp = restart["epsp"][...]
        self.sigma = restart["sigma"][...]
        self.sigmay = restart["sigmay"][...]
        self.state = restart["state"][...]
        self.epsframe = restart["uframe"][...]
        self.step = restart["step"][...]


def BranchPreparation(cli_args=None):
    r"""
    Branch from prepared stress state and add parameters.

    1.  Copy ``\param``.
        Add ``\param\kframe``, ``\param\sigmay``.

    2.  Copy ``\init`` to ``\restart``.
        Add ``\restart\epsp``, ``\restart\step``, and ``\restart\uframe``.
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
    parser.add_argument("--kframe", type=float, help="Frame stiffness (default 1 / L**2)")
    parser.add_argument(
        "--sigmay", type=float, nargs=2, default=[1.0, 0.3], help="Mean and std of sigmay"
    )
    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        g5.copy(src, dest, ["/meta", "/param"])
        g5.copy(src, dest, "/init", "/restart")
        dest["restart"]["epsp"] = np.zeros(src["param"]["shape"][...], dtype=np.float64)
        dest["restart"]["step"] = 0
        dest["restart"]["uframe"] = 0.0
        dest["param"]["sigmay"] = args.sigmay
        if args.kframe is not None:
            dest["param"]["kframe"] = args.kframe
        else:
            dest["param"]["kframe"] = 1.0 / (np.min(dest["param"]["shape"][...]) ** 2)
        tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)


def Run(cli_args=None):
    r"""
    Run simulation for a fixed number of steps.

    -   Write global output per step (``uframe``, ``sigma``, ``T``, ``S``, ``A``) in
        ``\AthermalQuasiStatic``.

    -   Write state to ``\AthermalQuasiStatic\restore`` every time a system-spanning event occurs
        (can be used to uniquely restore the system in this state).

    -   Backup state every ``backup_interval`` minutes by overwriting ``\restart``.
        (can be used to uniquely restore the system in this state).
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
    parser.add_argument("-n", "--nstep", type=int, default=5000, help="Total #load-steps to run")
    parser.add_argument("--backup-interval", default=5, type=int, help="Backup interval in minutes")
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/{m_name}/{funcname}", dev=args.develop)
        system = SystemSpringLoading(file)
        restart = file["restart"]

        if m_name not in file:
            res = file.create_group(m_name)
            restore = res.create_group("restore")
            start = 0
        else:
            res = file[m_name]
            restore = res["restore"]
            start = restart["step"][...] + 1
            # previous run was interrupted -> output arrays exceed restart step -> remove excess
            for key in ["uframe", "sigma", "T", "S", "A"]:
                res[key].resize((start,))
            if "step" in restore:
                n = np.argmax(restore["step"][...] == start - 1) + 1
                for key in ["uframe", "state", "step"]:
                    restore[key].resize((n,))
                for key in ["epsp", "sigma", "sigmay"]:
                    restore[key].resize((n, *file["param"]["shape"][...]))

        tic = time.time()
        pbar = tqdm.tqdm(range(start, start + args.nstep), desc=str(args.file))

        for step in pbar:
            pbar.refresh()
            n = np.copy(system.nfails)
            t0 = system.t

            # event-driven protocol
            if step % 2 == 1:
                system.shiftImposedShear()  # leaves >= 1 block unstable
            else:
                system.relaxAthermal()  # leaves 0 blocks unstable

            # global output
            with g5.ExtendableList(res, "uframe", np.float64) as dset:
                dset.append(system.epsframe)
            with g5.ExtendableList(res, "sigma", np.float64) as dset:
                dset.append(system.sigmabar)
            with g5.ExtendableList(res, "T", np.float64) as dset:
                dset.append(system.t - t0)
            with g5.ExtendableList(res, "S", np.int64) as dset:
                dset.append(np.sum(system.nfails - n))
            with g5.ExtendableList(res, "A", np.int64) as dset:
                dset.append(np.sum(system.nfails != n))

            # full state
            if np.sum(system.nfails != n) == system.size:
                with g5.ExtendableSlice(restore, "epsp", system.shape, np.float64) as dset:
                    dset += system.epsp
                with g5.ExtendableSlice(restore, "sigma", system.shape, np.float64) as dset:
                    dset += system.sigma
                with g5.ExtendableSlice(restore, "sigmay", system.shape, np.float64) as dset:
                    dset += system.sigmay
                with g5.ExtendableList(restore, "uframe", np.float64) as dset:
                    dset.append(system.epsframe)
                with g5.ExtendableList(restore, "state", np.uint64) as dset:
                    dset.append(system.state)
                with g5.ExtendableList(restore, "step", np.uint64) as dset:
                    dset.append(step)

            # full state
            if step == start + args.nstep - 1 or time.time() - tic > args.backup_interval * 60:
                tic = time.time()
                restart["epsp"][...] = system.epsp
                restart["sigma"][...] = system.sigma
                restart["sigmay"][...] = system.sigmay
                restart["state"][...] = system.state
                restart["uframe"][...] = system.epsframe
                restart["step"][...] = step
                file.flush()


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

    res = file[m_name]
    uframe = res["uframe"][...] / u0
    sigma = res["sigma"][...]  # mu, mean(sigmay) = 1

    tmp = uframe[0]
    uframe[0] = 1
    tangent = sigma / uframe
    tangent[0] = np.inf
    uframe[0] = tmp
    return np.argmax(tangent < 0.95) + 1


def EnsembleInfo(cli_args=None):
    """
    Basic interpretation of the ensemble.
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

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_info)
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        elastic = {"uframe": [], "sigma": [], "S": [], "A": [], "T": []}
        plastic = {"uframe": [], "sigma": [], "S": [], "A": [], "T": []}
        for ifile, f in enumerate(args.files):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    u0 = _norm_uframe(file)
                    output["/norm/uframe"] = u0
                    output["/norm/sigma"] = 1.0

                res = file[m_name]
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
                plastic["T"] += res["T"][i + 1 : end : 2].tolist()

        for key in elastic:
            output["/elastic/" + key] = elastic[key]
        for key in plastic:
            output["/plastic/" + key] = plastic[key]


def Plot(cli_args=None):
    """
    Basic of the ensemble.
    """

    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file) as file:
        i = _steady_state(file)
        res = file[m_name]
        S = file["/AthermalQuasiStatic/S"][i:].tolist()
        sigma = file["/AthermalQuasiStatic/restore/sigma"][...]
        sigmay = file["/AthermalQuasiStatic/restore/sigmay"][...]
        x = sigmay - np.abs(sigma)
        uframe = res["uframe"][...] / _norm_uframe(file)
        sigma = res["sigma"][...]

    S = np.array(S)
    S = S[S > 0]
    hist = enstat.histogram.from_data(S, bins=100, mode="log")

    fig, axes = gplt.subplots(ncols=3)

    ax = axes[0]
    ax.plot(uframe, sigma, marker=".")
    ax.axvline(uframe[i], ls="-", c="r", lw=1)
    ax.set_xlabel(r"$\bar{u}$")
    ax.set_ylabel(r"$\bar{\sigma}$")

    ax = axes[1]
    ax.plot(hist.x, hist.p)
    keep = np.logical_and(hist.x > 1e1, hist.x < 1e5)
    gplt.fit_powerlaw(hist.x[keep], hist.p[keep], axis=ax, auto_fmt="S")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$P(S)$")

    ax = axes[2]
    hist = enstat.histogram.from_data(x, bins=100, mode="log")
    ax.plot(hist.x, hist.p)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")

    plt.show()
    plt.close(fig)
