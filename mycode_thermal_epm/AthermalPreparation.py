import argparse
import inspect
import pathlib
import textwrap

import GooseEPM as epm
import h5py
import numpy as np
import shelephant

from . import tools
from ._version import version


def propagator(param: h5py.Group):
    if param["interactions"].asstr()[...] == "monotonic-shortrange":
        return epm.laplace_propagator()
    raise NotImplementedError("Unknown interactions type '%s'" % param["interactions"])


class SystemAthermal(epm.SystemAthermal):
    def __init__(self, file: h5py.File):
        param = file["param"]
        init = file["init"]
        epm.SystemAthermal.__init__(
            self,
            *propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * init["sigmay"].attrs["mean"],
            sigmay_std=np.ones(param["shape"][...]) * init["sigmay"].attrs["std"],
            seed=init["state"].attrs["seed"],
            alpha=param["alpha"][...],
            init_random_stress=True,
            init_relax=True,
            sigmabar=init["sigma"].attrs["mean"],
        )


def Generate(cli_args=None):
    """
    Generate IO file of the following structure:

        |-- param   # ensemble parameters
        |   |-- alpha
        |   |-- shape
        |   `-- interactions
        `-- init    # initial realisation -> use "Run" to get the full realisation
            |-- sigma
            |-- sigmay
            `-- state
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

    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)

    parser.add_argument("--alpha", type=float, default=1.5, help="Potential type")
    parser.add_argument(
        "--interactions",
        type=str,
        choices=["monotonic-shortrange", "monotonic-longrange", "eshelby"],
        help="Interaction type",
    )
    parser.add_argument("--shape", type=int, nargs=2, help="System shape")

    parser.add_argument("outdir", type=pathlib.Path, help="Output directory")

    args = tools._parse(parser, cli_args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n = args.size if args.shape is None else np.prod(args.shape)
    assert not any(
        [
            (args.outdir / f"id={i:04d}.h5").exists()
            for i in range(args.start, args.start + args.nsim)
        ]
    )
    files = []
    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n
        with h5py.File(args.outdir / files[-1], "w") as file:
            tools.create_check_meta(file, f"/meta/AthermalPreparation/{funcname}", dev=args.develop)

            param = file.create_group("param")
            param["alpha"] = args.alpha
            param["shape"] = args.shape
            param["interactions"] = args.interactions

            init = file.create_group("init")

            init.create_dataset("sigma", shape=args.shape, dtype=np.float64)
            init["sigma"].attrs["mean"] = 0.0

            init.create_dataset("sigmay", shape=args.shape, dtype=np.float64)
            init["sigmay"].attrs["mean"] = 1.0
            init["sigmay"].attrs["std"] = 0.1

            init.create_dataset("state", shape=[], dtype=np.uint64)
            init["state"].attrs["seed"] = seed

    executable = "AthermalPreparation_Run"
    commands = [f"{executable} {file}" for file in files]
    shelephant.yaml.dump(args.outdir / "commands_run.yaml", commands, force=True)


def Run(cli_args=None):
    """
    Initialize system, and store state.
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
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/AthermalPreparation/{funcname}", dev=args.develop)
        system = SystemAthermal(file)
        init = file["init"]
        init["sigma"][...] = system.sigma
        init["sigmay"][...] = system.sigmay
        init["state"][...] = system.state
