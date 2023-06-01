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
from . import tag
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"


class SystemAthermal(epm.SystemAthermal):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemAthermal.__init__(
            self,
            *AthermalPreparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            init_random_stress=False,
            init_relax=False,
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
        system = SystemAthermal(file)

        if "ExtremeValue" not in file:
            res = file.create_group("ExtremeValue")
            res.create_group("x")
            res["n"] = 0
        else:
            res = file["ExtremeValue"]

        restart = file["restart"]

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            system.sigmabar = file["param"]["sigmabar"][...]  # avoid drift due to numerical errors
            n = system.nfails.copy()
            system.makeWeakestFailureSteps(args.interval * system.size, allow_stable=True)
            while True:
                if np.all(system.nfails - n >= args.interval):
                    break
                system.makeWeakestFailureSteps(system.size, allow_stable=True)

            n = res["n"][...]
            res["n"][...] = n + 1
            res["x"][str(n)] = system.sigmay - system.sigma

            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
            restart["state"][...] = system.state
            file.flush()
