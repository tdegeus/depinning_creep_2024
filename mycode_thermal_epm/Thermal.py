import argparse
import inspect
import pathlib
import textwrap

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
        self.temperature = param["temperature"][...]


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
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
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
        dest["param"]["temperature"] = args.temperature
        tools.create_check_meta(dest, f"/meta/Thermal/{funcname}", dev=args.develop)


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
    parser.add_argument("--ninc", type=int, help="#increments to measure (default: system size)")
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/Thermal/{funcname}", dev=args.develop)
        system = SystemStressControl(file)
        if args.ninc is None:
            args.ninc = system.size
        else:
            assert args.ninc > 0

        if "Thermal" not in file:
            res = file.create_group("Thermal")
            res.create_group("x")
            res.create_group("S")
            res.create_group("A")
            res["n"] = 0
        else:
            res = file["Thermal"]

        restart = file["restart"]
        S = np.empty(args.ninc, dtype=np.int64)
        A = np.empty(args.ninc, dtype=np.int64)

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            nfails = system.nfails.copy()
            system.makeThermalFailureSteps(args.interval * system.size)
            while True:
                if np.all(system.nfails - nfails >= args.interval):
                    break
                system.makeThermalFailureSteps(system.size)

            nfails = system.nfails.copy()
            S[0] = 0
            A[0] = 0
            for inc in range(1, args.ninc):
                system.makeThermalFailureSteps(system.size)
                S[inc] = np.sum(system.nfails - nfails)
                A[inc] = np.sum(system.nfails != nfails)

            n = res["n"][...]
            res["n"][...] = n + 1
            res["x"][str(n)] = system.sigmay - system.sigma
            res["S"][str(n)] = S
            res["A"][str(n)] = A

            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
            restart["state"][...] = system.state
            file.flush()
