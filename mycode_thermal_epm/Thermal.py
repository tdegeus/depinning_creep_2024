import argparse
import inspect
import pathlib
import textwrap

import enstat
import GooseEPM as epm
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import Preparation
from . import tag
from . import tools
from ._version import version
from .Preparation import data_version

f_info = "EnsembleInfo.h5"
m_name = "Thermal"
m_exclude = ["Extremal", "AQS"]


class SystemThermalStressControl(epm.SystemThermalStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemThermalStressControl.__init__(
            self,
            *Preparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            random_stress=False,
        )

        self.epsp = restart["epsp"][...]
        self.sigma = restart["sigma"][...]
        self.sigmay = restart["sigmay"][...]
        self.t = restart["t"][...]
        self.state = restart["state"][...]
        self.sigmabar = param["sigmabar"][...]
        self.temperature = param["temperature"][...]


def _upgrade_data(filename: pathlib.Path, temp_file: pathlib.Path) -> bool:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_file: Temporary filename.
    :return: True if the file was upgraded.
    """
    with h5py.File(filename) as src:
        assert "Thermal" in src, "Not a 'Thermal' file"

        if tag.greater_equal(Preparation.get_data_version(src), "1.0"):
            return False

        with h5py.File(temp_file, "w") as dst:
            paths = g5.getdatapaths(src)
            paths.remove("/Thermal/S")

            S = src["Thermal/S"][...]
            expect = np.arange(S.shape[1]).reshape((1, -1))
            assert np.all(np.equal(S, expect)), "S must arange"

            g5.copy(src, dst, paths)
            dst["/param/data_version"] = data_version

    return True


def UpgradeData(cli_args=None):
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, _upgrade_data)


def BranchPreparation(cli_args=None):
    r"""
    Branch from prepared stress state and add parameters.

    1.  Copy ``\param``.
        Add ``\param\sigmabar``, ``\param\temperature``, ``\param\sigmay``.

    2.  Copy ``\init`` to ``\restart``.
        Add ``\restart\epsp``, ``\restart\t``.
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
    parser.add_argument("--sigmabar", type=float, default=0.3, help="Stress")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument(
        "--sigmay", type=float, nargs=2, default=[1.0, 0.3], help="Mean and std of sigmay"
    )
    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(m in src for m in m_exclude), "Wrong file type"
        g5.copy(src, dest, ["/meta", "/param"])
        g5.copy(src, dest, "/init", "/restart")
        dest["restart"]["epsp"] = np.zeros(src["param"]["shape"][...], dtype=np.float64)
        dest["restart"]["t"] = 0.0
        dest["param"]["sigmay"] = args.sigmay
        dest["param"]["sigmabar"] = args.sigmabar
        dest["param"]["temperature"] = args.temperature
        tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)


def Run(cli_args=None):
    """
    Run simulation at fixed stress.
    Measure:

        -   The state every ``--interval`` events.
            The ``--interval`` is the number of times that all blocks have to have failed between
            measurements.

        -   Avalanches during ``--ninc`` steps.
            This measurement can be switched off with ``--flow`` to get minimal output
            (``mean_epsp`` and ``t``)
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
    parser.add_argument("--interval", type=int, default=100, help="Measure every #events")
    parser.add_argument("-n", "--measurements", type=int, default=100, help="Total #measurements")
    parser.add_argument("--ninc", type=int, help="#increments to measure (default: ``20 N``)")
    parser.add_argument("--flow", action="store_true", help="Flow only (min. out., no avalanches)")
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/{m_name}/{funcname}", dev=args.develop)
        system = SystemThermalStressControl(file)
        restart = file["restart"]
        if args.ninc is None:
            args.ninc = 20 * system.size
        else:
            assert args.ninc > 0

        if m_name not in file:
            assert not any(m in file for m in m_exclude), "Wrong file type"
            res = file.create_group(m_name)
        else:
            res = file[m_name]

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            nfails = system.nfails.copy()
            system.makeThermalFailureSteps(args.interval * system.size)
            while True:
                if np.all(system.nfails - nfails >= args.interval):
                    break
                system.makeThermalFailureSteps(system.size)

            with g5.ExtendableList(res, "mean_epsp", np.float64) as dset:
                dset.append(np.mean(system.epsp))
            with g5.ExtendableList(res, "t", np.float64) as dset:
                dset.append(system.t)

            if not args.flow:
                with g5.ExtendableSlice(res, "epsp", system.shape, np.float64) as dset:
                    dset += system.epsp
                with g5.ExtendableSlice(res, "sigma", system.shape, np.float64) as dset:
                    dset += system.sigma
                with g5.ExtendableSlice(res, "sigmay", system.shape, np.float64) as dset:
                    dset += system.sigmay
                with g5.ExtendableList(res, "state", np.uint64) as dset:
                    dset.append(system.state)

            if args.ninc > 0 and not args.flow:
                avalanche = epm.Avalanche()
                avalanche.makeThermalFailureSteps(system, args.ninc)
                with g5.ExtendableSlice(res, "idx", [args.ninc], np.uint64) as dset:
                    dset += avalanche.idx
                with g5.ExtendableSlice(res, "T", [args.ninc], np.float64) as dset:
                    dset += avalanche.T

            restart["epsp"][...] = system.epsp
            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
            restart["t"][...] = system.t
            restart["state"][...] = system.state
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

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_info)
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, f"/meta/{m_name}/{funcname}", dev=args.develop)
        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                assert m_name in file, "Wrong file type"
                assert not any(m in file for m in m_exclude), "Wrong file type"
                res = file[m_name]

                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    xc = file["param"]["sigmay"][0] - file["param"]["sigmabar"][...]
                    alpha = file["param"]["alpha"][...]
                    temperature = file["param"]["temperature"][...]
                    size = np.prod(file["param"]["shape"][...])
                    t0 = np.exp(xc**alpha / temperature)
                    output["/norm/xc"] = xc
                    output["/norm/t0"] = t0
                    hist = enstat.histogram(bin_edges=np.linspace(0, 3, 2001), bound_error="ignore")
                    tmax = res["T"][-1, -1] / t0
                    bin_edges = np.linspace(0, 2 * tmax, 2001)  # todo: logspace, larger?
                    S = enstat.binned(bin_edges, names=["x", "y"], bound_error="ignore")
                    Ssq = enstat.binned(bin_edges, names=["x", "y"], bound_error="ignore")
                    A = enstat.binned(bin_edges, names=["x", "y"], bound_error="ignore")
                    Asq = enstat.binned(bin_edges, names=["x", "y"], bound_error="ignore")
                    N = np.prod(file["param"]["shape"][...])

                hist += (res["sigmay"][...] - np.abs(res["sigma"][...])).ravel()
                for i in range(res["T"].shape[0]):
                    ti = res["T"][i, ...] / t0
                    ai = epm.cumsum_n_unique(res["idx"][i, ...]) / N
                    si = np.arange(ti.size) / N
                    S.add_sample(ti, si)
                    Ssq.add_sample(ti, si**2)
                    A.add_sample(ti, ai)
                    Asq.add_sample(ti, ai**2)

        output["files"] = sorted([f.name for f in args.files])

        res = output.create_group("hist_x")
        res["x"] = hist.x
        res["p"] = hist.p
        res["count"] = hist.count

        for name, variable in zip(
            ["S", "Ssq", "A", "Asq", "t"], [S["y"], Ssq["y"], A["y"], Asq["y"], S["x"]]
        ):
            for key, value in variable:
                output[f"/restore/{name}/{key}"] = value

        output["chi4_S"] = N * (Ssq["y"].mean() - S["y"].mean() ** 2)
        output["chi4_A"] = N * (Asq["y"].mean() - A["y"].mean() ** 2)
        output["t"] = S["x"].mean()


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
        x = file["hist_x"]["x"][...]
        p = file["hist_x"]["p"][...]
        xc = file["param"]["sigmay"][0] - file["param"]["sigmabar"][...]

        chi4_S = file["chi4_S"][...]
        chi4_A = file["chi4_A"][...]
        t = file["t"][...]

    fig, axes = gplt.subplots(ncols=3)

    ax = axes[0]
    ax.plot(x, p)
    ax.axvline(xc, color="r", ls="-", label=r"$\langle \sigma_y \rangle - \bar{\sigma}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    ax.legend()

    ax = axes[1]
    ax.plot(t, chi4_S, label=r"$\chi_4^S$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\chi_4^S$")

    ax = axes[2]
    ax.plot(t, chi4_A, label=r"$\chi_4^A$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\chi_4^A$")

    plt.show()
    plt.close(fig)
