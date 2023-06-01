import argparse
import inspect
import pathlib
import textwrap
import time

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
        restart = file["restart"]

        epm.SystemDrivenAthermal.__init__(
            self,
            *AthermalPreparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            kframe=param["kframe"][...],
            init_random_stress=False,
            init_relax=False,
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
        system = SystemDrivenAthermal(file)

        if "AthermalQuasiStatic" not in file:
            res = file.create_group("AthermalQuasiStatic")
            end = args.nstep - 1
            sigma = res.create_dataset("sigma", (args.nstep,), maxshape=(None,), dtype=np.float64)
            uframe = res.create_dataset("uframe", (args.nstep,), maxshape=(None,), dtype=np.float64)
            S = res.create_dataset("S", (args.nstep,), maxshape=(None,), dtype=np.int64)
            A = res.create_dataset("A", (args.nstep,), maxshape=(None,), dtype=np.int64)
            t = res.create_dataset("t", (args.nstep,), maxshape=(None,), dtype=np.int64)
            sigma[system.step] = system.sigmabar
            uframe[system.step] = system.epsframe
            S[system.step] = 0
            A[system.step] = 0
            t[system.step] = 0
            system.step += 1
        else:
            res = file["AthermalQuasiStatic"]
            system.step += 1
            end = system.step + args.nstep - 1
            sigma = res["sigma"]
            uframe = res["uframe"]
            S = res["S"]
            A = res["A"]
            t = res["t"]
            for dset in [sigma, uframe, S, A, t]:
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
            t[step] = system.t - t0

            if step == end or time.time() - tic > args.backup_interval * 60:
                tic = time.time()
                restart["sigma"][...] = system.sigma
                restart["sigmay"][...] = system.sigmay
                restart["state"][...] = system.state
                restart["uframe"][...] = system.epsframe
                restart["step"][...] = step
                file.flush()
