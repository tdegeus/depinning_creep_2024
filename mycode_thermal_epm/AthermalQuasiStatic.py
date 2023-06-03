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
from . import tag
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"


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

        self.sigma = restart["sigma"][...]
        self.sigmay = restart["sigmay"][...]
        self.state = restart["state"][...]
        self.epsframe = restart["uframe"][...]
        self.step = restart["step"][...]


def BranchPreparation(cli_args=None):
    """
    Branch from prepared stress state and add parameters.
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
    parser.add_argument(
        "--sigmay", type=float, nargs=2, default=[1.0, 0.1], help="Mean and std of sigmay"
    )
    parser.add_argument("--kframe", type=float, help="Frame stiffness (default 1 / L**2)")
    parser.add_argument("input", type=pathlib.Path, help="Input file")
    parser.add_argument("output", type=pathlib.Path, help="Output file")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        g5.copy(src, dest, ["/meta", "/param"])
        g5.copy(src, dest, "/init", "/restart")
        dest["restart"]["step"] = 0
        dest["restart"]["uframe"] = 0.0
        dest["param"]["sigmay"] = args.sigmay
        if args.kframe is not None:
            dest["param"]["kframe"] = args.kframe
        else:
            dest["param"]["kframe"] = 1.0 / (np.min(dest["param"]["shape"][...]) ** 2)
        tools.create_check_meta(dest, f"/meta/AthermalQuasiStatic/{funcname}", dev=args.develop)


def Run(cli_args=None):
    """
    Run simulation for a fixed number of steps.
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
        tools.create_check_meta(file, f"/meta/AthermalQuasiStatic/{funcname}", dev=args.develop)
        system = SystemSpringLoading(file)

        if "AthermalQuasiStatic" not in file:
            res = file.create_group("AthermalQuasiStatic")
            end = args.nstep - 1
            sigma = res.create_dataset("sigma", (args.nstep,), maxshape=(None,), dtype=np.float64)
            uframe = res.create_dataset("uframe", (args.nstep,), maxshape=(None,), dtype=np.float64)
            S = res.create_dataset("S", (args.nstep,), maxshape=(None,), dtype=np.int64)
            A = res.create_dataset("A", (args.nstep,), maxshape=(None,), dtype=np.int64)
            T = res.create_dataset("T", (args.nstep,), maxshape=(None,), dtype=np.int64)
            sigma[system.step] = system.sigmabar
            uframe[system.step] = system.epsframe
            S[system.step] = 0
            A[system.step] = 0
            T[system.step] = 0
            system.step += 1
        else:
            res = file["AthermalQuasiStatic"]
            system.step += 1
            end = system.step + args.nstep - 1
            sigma = res["sigma"]
            uframe = res["uframe"]
            S = res["S"]
            A = res["A"]
            T = res["T"]
            for dset in [sigma, uframe, S, A, T]:
                dset.resize((system.step + args.nstep,))

        restart = file["restart"]
        tic = time.time()
        pbar = tqdm.tqdm(range(system.step, end + 1), desc=str(args.file))

        for step in pbar:
            pbar.refresh()
            n = np.copy(system.nfails)
            t0 = system.t
            if step % 2 == 1:
                system.shiftImposedShear()
            else:
                system.relaxAthermal()

            uframe[step] = system.epsframe
            sigma[step] = system.sigmabar
            S[step] = np.sum(system.nfails - n)
            A[step] = np.sum(system.nfails != n)
            T[step] = system.t - t0

            if step == end or time.time() - tic > args.backup_interval * 60:
                tic = time.time()
                restart["sigma"][...] = system.sigma
                restart["sigmay"][...] = system.sigmay
                restart["state"][...] = system.state
                restart["uframe"][...] = system.epsframe
                restart["step"][...] = step
                file.flush()


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
                    kframe = file["/param/kframe"][...]
                    u0 = (kframe + 1.0) / kframe  # mu = 1
                    output["/norm/uframe"] = u0
                    output["/norm/sigma"] = 1.0
                    ver = tools.read_version(file, "/meta/AthermalQuasiStatic/Run")

                uframe = file["/AthermalQuasiStatic/uframe"][...] / u0
                sigma = file["/AthermalQuasiStatic/sigma"][...]
                tmp = uframe[0]
                uframe[0] = 1
                tangent = sigma / uframe
                tangent[0] = np.inf
                uframe[0] = tmp

                i = np.argmax(tangent < 0.95) + 1
                if i % 2 == 0:
                    i += 1
                if (uframe.size - i) % 2 == 1:
                    end = -1
                else:
                    end = None

                elastic["uframe"] += uframe[i:end:2].tolist()
                elastic["sigma"] += sigma[i:end:2].tolist()
                plastic["uframe"] += uframe[i + 1 : end : 2].tolist()
                plastic["sigma"] += sigma[i + 1 : end : 2].tolist()
                plastic["S"] += file["/AthermalQuasiStatic/S"][i + 1 : end : 2].tolist()
                plastic["A"] += file["/AthermalQuasiStatic/A"][i + 1 : end : 2].tolist()
                if tag.greater(ver, "1.1"):
                    plastic["T"] += file["/AthermalQuasiStatic/T"][i + 1 : end : 2].tolist()

        for key in elastic:
            output["/elastic/" + key] = elastic[key]
        for key in plastic:
            output["/plastic/" + key] = plastic[key]


def Plot(cli_args=None):
    """
    Basic plot.
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
        kframe = file["/param/kframe"][...]
        u0 = (kframe + 1.0) / kframe  # mu = 1
        uframe = file["/AthermalQuasiStatic/uframe"][...] / u0
        sigma = file["/AthermalQuasiStatic/sigma"][...]
        tmp = uframe[0]
        uframe[0] = 1
        tangent = sigma / uframe
        tangent[0] = np.inf
        uframe[0] = tmp
        i = np.argmax(tangent < 0.95) + 1
        S = file["/AthermalQuasiStatic/S"][i:].tolist()

    S = np.array(S)
    S = S[S > 0]
    hist = enstat.histogram.from_data(S, bins=100, mode="log")

    fig, axes = gplt.subplots(ncols=2)

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

    plt.show()
    plt.close(fig)
