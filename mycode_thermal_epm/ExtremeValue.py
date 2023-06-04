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

from . import AthermalPreparation
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"


class SystemStressControl(epm.SystemStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemStressControl.__init__(
            self,
            *AthermalPreparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            random_stress=False,
        )

        self.sigma = restart["sigma"][...]
        self.sigmay = restart["sigmay"][...]
        self.state = restart["state"][...]
        self.sigmabar = param["sigmabar"][...]


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
    parser.add_argument("--sigmabar", type=float, default=0.3, help="Stress")
    parser.add_argument(
        "--sigmay", type=float, nargs=2, default=[1.0, 0.1], help="Mean and std of sigmay"
    )
    parser.add_argument("input", type=pathlib.Path, help="Input file")
    parser.add_argument("output", type=pathlib.Path, help="Output file")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        g5.copy(src, dest, ["/meta", "/param"])
        g5.copy(src, dest, "/init", "/restart")
        dest["param"]["sigmay"] = args.sigmay
        dest["param"]["sigmabar"] = args.sigmabar
        tools.create_check_meta(dest, f"/meta/ExtremeValue/{funcname}", dev=args.develop)


def Run(cli_args=None):
    """
    Measure stability "x" after all sites have failed ``n`` times.
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
        tools.create_check_meta(file, f"/meta/ExtremeValue/{funcname}", dev=args.develop)
        system = SystemStressControl(file)

        if "ExtremeValue" not in file:
            res = file.create_group("ExtremeValue")
            res.create_group("sigma")
            res.create_group("sigmay")
            res["n"] = 0
        else:
            res = file["ExtremeValue"]

        restart = file["restart"]

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            nfails = system.nfails.copy()
            system.makeWeakestFailureSteps(args.interval * system.size, allow_stable=True)
            while True:
                if np.all(system.nfails - nfails >= args.interval):
                    break
                system.makeWeakestFailureSteps(system.size, allow_stable=True)

            n = res["n"][...]
            res["n"][...] = n + 1
            res["sigma"][str(n)] = system.sigma
            res["sigmay"][str(n)] = system.sigmay

            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
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

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_info)
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    x = []

    with h5py.File(args.output, "w") as output:
        for ifile, f in enumerate(args.files):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])

                res = file["ExtremeValue"]
                n = res["n"][...]
                for i in range(n):
                    x += (res["sigmay"][str(i)][...] - res["sigma"][str(i)][...]).tolist()

    # import GooseMPL as gplt  # noqa: F401
    # import matplotlib.pyplot as plt  # noqa: F401

    # plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    # fig, ax = plt.subplots()
    # hist = enstat.histogram.from_data(x, bins=100, mode="log")
    # ax.plot(hist.x, hist.p)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.show()
