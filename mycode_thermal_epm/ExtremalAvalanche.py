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

from . import Extremal
from . import Preparation
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"
m_name = "Extremal"
m_exclude = ["Thermal", "AQS"]


def BranchExtremal(cli_args=None):
    r"""
    Branch from prepared stress state using :py:func:`Run`:
    Copy ``\param`` and ``\restart``.
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
    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(m in src for m in m_exclude), "Wrong file type"
        g5.copy(src, dest, ["/meta", "/param", "/restart"])
        meta = tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)
        meta.attrs["t"] = src["restart"]["t"][...]


def Run(cli_args=None):
    """
    Measure 'avalanche' by running ``--ninc`` steps at fixed stress.
    The output is written every ``--ncache`` steps.
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
    parser.add_argument(
        "-n", "--ninc-size", type=int, help="--ninc in units of system size", default=100
    )
    parser.add_argument("--ninc", type=int, help="#increments to measure (default: ``100 N``)")
    parser.add_argument("--ncache", type=int, help="write output every #increments (default: N)")
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, f"/meta/{m_name}/{funcname}", dev=args.develop)
        system = Extremal.SystemStressControl(file)
        restart = file["restart"]
        if args.ninc is None:
            args.ninc = args.ninc_size * system.size
        if args.ncache is None:
            args.ncache = system.size

        assert args.ninc > 0
        assert args.ninc % args.ncache == 0

        if m_name not in file:
            assert not any(m in file for m in m_exclude), "Wrong file type"
            res = file.create_group(m_name).create_group("Avalanche")
        else:
            res = file[m_name]

        for _ in tqdm.tqdm(range(args.ninc // args.ncache), desc=str(args.file)):
            avalanche = epm.AvalancheWeakest()
            avalanche.makeWeakestFailureSteps(system, args.ncache)
            with g5.ExtendableList(res, "idx", np.uint64) as dset:
                dset += avalanche.idx
            with g5.ExtendableList(res, "xmin", np.float64) as dset:
                dset += avalanche.xmin

            restart["epsp"][...] = system.epsp
            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
            restart["t"][...] = system.t
            restart["state"][...] = system.state
            file.flush()
