import argparse
import inspect
import pathlib
import textwrap

import GooseEPM as epm
import h5py
import numpy as np
import tqdm

from . import tools
from ._version import version


class SystemDrivenAthermal(epm.SystemAthermal):
    def __init__(self, file: h5py.File):
        param = file["param"]
        init = file["init"]

        if param["interactions"].asstr()[...] == "laplace":
            propagator = epm.laplace_propagator()
        else:
            raise NotImplementedError

        epm.SystemDrivenAthermal.__init__(
            self,
            *propagator,
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"]["mean"][...],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"]["std"][...],
            seed=init["state"].attrs["seed"],
            alpha=param["alpha"][...],
            kframe=param["kframe"][...],
            init_random_stress=False,
            init_relax=False,
            sigmabar=init["sigma"].attrs["mean"],
        )

        self.sigma = init["sigma"][...]
        self.sigmabar = init["sigma"].attrs["mean"]
        self.sigmay = init["sigmay"][...]
        self.state = init["state"][...]


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
        res = file["result"]
        sigma = res.create_dataset("sigma", (args.nstep,), dtype=np.float64)
        uframe = res.create_dataset("uframe", (args.nstep,), dtype=np.float64)
        sigma[0] = system.sigmabar
        uframe[0] = system.epsframe

        for i in tqdm.tqdm(range(1, args.nstep)):
            if i % 2 == 1:
                system.shiftImposedShear()
            else:
                system.relaxAthermal()

            uframe[i] = system.epsframe
            sigma[i] = system.sigmabar
