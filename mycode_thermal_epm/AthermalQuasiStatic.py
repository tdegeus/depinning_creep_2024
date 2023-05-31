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


class SystemDrivenAthermal(epm.SystemDrivenAthermal):
    def __init__(self, file: h5py.File):
        param = file["param"]
        init = file["init"]

        epm.SystemDrivenAthermal.__init__(
            self,
            *AthermalPreparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=init["state"].attrs["seed"],
            alpha=param["alpha"][...],
            kframe=param["kframe"][...],
            init_random_stress=False,
            init_relax=False,
        )

        self.sigma = init["sigma"][...]
        self.sigmabar = init["sigma"].attrs["mean"]
        self.sigmay = init["sigmay"][...]
        self.state = init["state"][...]


def BranchPreparation(cli_args=None):
    """
    Branch from prepared stress state.
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
        g5.copy(src, dest, ["/meta", "/param", "/init"])
        dest["param"]["sigmay"] = args.sigmay
        if args.kframe is not None:
            dest["param"]["kframe"] = args.kframe
        else:
            dest["param"]["kframe"] = 1.0 / (np.min(dest["param"]["shape"][...]) ** 2)


def Run(cli_args=None):
    """
    Run simulation.
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
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        system = SystemDrivenAthermal(file)
        res = file.create_group("AthermalQuasiStatic")
        sigma = res.create_dataset("sigma", (args.nstep,), dtype=np.float64)
        uframe = res.create_dataset("uframe", (args.nstep,), dtype=np.float64)
        S = res.create_dataset("S", (args.nstep,), dtype=np.float64)
        A = res.create_dataset("A", (args.nstep,), dtype=np.float64)
        sigma[0] = system.sigmabar
        uframe[0] = system.epsframe
        S[0] = 0
        A[0] = 0

        for i in tqdm.tqdm(range(1, args.nstep)):
            n = np.copy(system.nfails)
            if i % 2 == 1:
                system.shiftImposedShear()
            else:
                system.relaxAthermal()

            uframe[i] = system.epsframe
            sigma[i] = system.sigmabar
            S[i] = np.sum(system.nfails - n)
            A[i] = np.sum(system.nfails != n)
