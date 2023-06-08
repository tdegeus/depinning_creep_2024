import argparse
import inspect
import os
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


class SystemStressControl(epm.SystemStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        init = file["init"]
        epm.SystemStressControl.__init__(
            self,
            *propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * init["sigmay"].attrs["mean"],
            sigmay_std=np.ones(param["shape"][...]) * init["sigmay"].attrs["std"],
            seed=init["state"].attrs["seed"],
            alpha=param["alpha"][...],
            random_stress=True,
        )


def Generate(cli_args=None):
    """
    Generate IO file of the following structure::

        |-- param   # ensemble parameters
        |   |-- alpha
        |   |-- shape
        |   `-- interactions
        `-- init    # initial realisation -> use "Run" to fill
            |-- sigma
            |-- sigmay
            `-- state

    .. note::

        You can take `t` and `epsp` equal to zero.
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

    parser.add_argument("--shape", type=int, nargs=2, help="System shape")
    parser.add_argument("--alpha", type=float, default=1.5, help="Potential type")
    parser.add_argument(
        "--interactions",
        type=str,
        choices=["monotonic-shortrange", "monotonic-longrange", "eshelby"],
        help="Interaction type",
    )
    parser.add_argument("--all", action="store_true", help="Generate a suggestion of runs")

    parser.add_argument("outdir", type=pathlib.Path, help="Output directory")

    args = tools._parse(parser, cli_args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n = np.prod(args.shape)
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

            init.create_dataset("sigmay", shape=args.shape, dtype=np.float64)
            init["sigmay"].attrs["mean"] = 1.0
            init["sigmay"].attrs["std"] = 0.3

            init.create_dataset("state", shape=[], dtype=np.uint64)
            init["state"].attrs["seed"] = seed

    exec = "AthermalPreparation_Run"
    commands = [f"{exec} {f}" for f in files]
    shelephant.yaml.dump(args.outdir / "commands_run.yaml", commands, force=True)

    if not args.all:
        return

    assert len(os.path.split(args.outdir)) > 1

    for name in ["AthermalQuasiStatic", "ExtremeValue"]:
        base = args.outdir / ".." / name
        base.mkdir(parents=True, exist_ok=True)

        exec = f"{name}_BranchPreparation"
        commands = [f"{exec} ../{args.outdir.name}/{f} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exec = f"{name}_Run"
        if name == "ExtremeValue":
            commands = [f"{exec} -n 100 {f}" for f in files]
        else:
            commands = [f"{exec} {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)

    name = "Thermal"
    temperatures = {
        "temperature=0,005": 0.005,
        "temperature=0,007": 0.007,
        "temperature=0,01": 0.01,
        "temperature=0,05": 0.05,
        "temperature=0,07": 0.07,
        "temperature=0,1": 0.1,
        "temperature=0,5": 0.5,
    }
    for key, temp in temperatures.items():
        base = args.outdir / ".." / name / key
        base.mkdir(parents=True, exist_ok=True)

        exec = f"{name}_BranchPreparation"
        commands = [f"{exec} ../../{args.outdir.name}/{f} {f} --temperature {temp}" for f in files]
        shelephant.yaml.dump(base / "commands_branch.yaml", commands, force=True)

        exec = f"{name}_Run"
        commands = [f"{exec} -n 100 {f}" for f in files]
        shelephant.yaml.dump(base / "commands_run.yaml", commands, force=True)


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
        system = SystemStressControl(file)
        init = file["init"]
        init["sigma"][...] = system.sigma
        init["sigmay"][...] = system.sigmay
        init["state"][...] = system.state
