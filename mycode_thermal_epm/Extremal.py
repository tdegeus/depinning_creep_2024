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
from . import Thermal
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"
m_name = "Extremal"
m_exclude = ["AQS", "ExtremalAvalanche", "Thermal"]


class SystemStressControl(epm.SystemStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemStressControl.__init__(
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


def _upgrade_data(filename: pathlib.Path, temp_dir: pathlib.Path) -> bool:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :return: ``temp_file`` if the data is upgraded, ``None`` otherwise.
    """
    with h5py.File(filename) as src:
        assert not any(x in src for x in m_exclude + Preparation._libname_pre_v2(m_exclude))
        ver = Preparation.get_data_version(src)

    if tag.greater_equal(ver, "2.0"):
        return None

    temp_file = temp_dir / "from_older.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        if "ExtremeValue" in src:
            g5.copy(src, dst, "/ExtremeValue", "/Extremal")
        g5.copy(src, dst, ["/param", "/restart"])
        Preparation._copy_metadata_pre_v2(src, dst)

    return temp_file


def UpgradeData(cli_args=None):
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, _upgrade_data)


def BranchPreparation(cli_args=None):
    r"""
    Branch from prepared stress state and add parameters.

    1.  Copy ``\param``.
        Add ``\param\sigmabar``, ``\param\sigmay``.

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
        tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)


def Run(cli_args=None):
    """
    Run simulation at fixed stress.
    Measure:

        -   The state every ``--interval`` events.
            The ``--interval`` is the number of times that all blocks have to have failed between
            measurements.
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
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/{m_name}/{funcname}", dev=args.develop)
        system = SystemStressControl(file)
        restart = file["restart"]

        if m_name not in file:
            assert not any(m in file for m in m_exclude), "Wrong file type"
            res = file.create_group(m_name)
        else:
            res = file[m_name]

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            nfails = system.nfails.copy()
            system.makeWeakestFailureSteps(args.interval * system.size, allow_stable=True)
            while True:
                if np.all(system.nfails - nfails >= args.interval):
                    break
                system.makeWeakestFailureSteps(system.size, allow_stable=True)

            with g5.ExtendableSlice(res, "epsp", system.shape, np.float64) as dset:
                dset += system.epsp
            with g5.ExtendableSlice(res, "sigma", system.shape, np.float64) as dset:
                dset += system.sigma
            with g5.ExtendableSlice(res, "sigmay", system.shape, np.float64) as dset:
                dset += system.sigmay
            with g5.ExtendableList(res, "state", np.uint64) as dset:
                dset.append(system.state)
            with g5.ExtendableList(res, "t", np.float64) as dset:
                dset.append(system.t)

            Preparation.overwrite_restart(restart, system)


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
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    alpha = file["param"]["alpha"][...]
                    dE = enstat.scalar()
                    pdfx = enstat.histogram(bin_edges=np.linspace(0, 3, 2001))

                if m_name not in file:
                    assert not any(m in file for m in m_exclude), "Wrong file type"
                    continue

                res = file[m_name]
                x = (res["sigmay"][...] - np.abs(res["sigma"][...])).ravel()
                sorter = np.argsort(x)
                dE += x[sorter[1]] ** alpha - x[sorter[0]] ** alpha
                pdfx += x

        output["files"] = sorted([f.name for f in args.files])
        output["/dE/first"] = dE.first
        output["/dE/second"] = dE.second
        output["/dE/norm"] = dE.norm
        output["/hist_x/bin_edges"] = pdfx.bin_edges
        output["/hist_x/count"] = pdfx.count


def EnsembleHeightHeight(cli_args=None):
    """
    Get the height-height correlation function.
    """
    return Thermal.EnsembleHeightHeight(cli_args, m_name)


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

    fig, ax = gplt.subplots()
    ax.plot(x, p)
    ax.axvline(xc, color="r", ls="-", label=r"$\langle \sigma_y \rangle - \bar{\sigma}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    ax.legend()
    plt.show()
    plt.close(fig)
