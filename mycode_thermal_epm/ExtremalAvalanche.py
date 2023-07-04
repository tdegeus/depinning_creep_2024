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
from scipy import stats

from . import Extremal
from . import Preparation
from . import tag
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"
m_name = "ExtremalAvalanche"
m_exclude = ["AQS", "Thermal"]


def _upgrade_data_v1_to_v2(src: h5py.File, dst: h5py.File):
    if "ExtremeValue" in src:
        g5.copy(src, dst, "/ExtremeValue/Avalanche", "/ExtremalAvalanche")
    g5.copy(src, dst, ["/param", "/restart"])
    Preparation._copy_metadata_pre_v2(src, dst)


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

    if tag.less(ver, "2.0"):
        temp_file = temp_dir / "from_2_0.h5"
        with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
            _upgrade_data_v1_to_v2(src, dst)

    return temp_file


def UpgradeData(cli_args=None):
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, _upgrade_data)


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
            res = file.create_group(m_name)
            dt = 0
        else:
            res = file[m_name]
            dt = res["idx"].size

        # check data integrity
        # todo: make independent of BranchExtremal
        assert file[f"/meta/{m_name}/BranchExtremal"].attrs["t"] + dt == restart["t"][...]

        for _ in tqdm.tqdm(range(args.ninc // args.ncache), desc=str(args.file)):
            measurement = epm.Avalanche()
            measurement.makeWeakestFailureSteps(system, args.ncache, allow_stable=True)
            with g5.ExtendableList(res, "idx", np.uint64) as dset:
                dset += measurement.idx
            with g5.ExtendableList(res, "xmin", np.float64) as dset:
                dset += measurement.x

            restart["epsp"][...] = system.epsp
            restart["sigma"][...] = system.sigma
            restart["sigmay"][...] = system.sigmay
            restart["t"][...] = system.t
            restart["state"][...] = system.state
            file.flush()


def EnsembleInfo(cli_args=None, myname=m_name):
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
    parser.add_argument("--nbins", type=int, help="Number of bins", default=100)
    parser.add_argument("--ndx", type=int, help="Number of x_c - x_0 to sample", default=100)
    parser.add_argument("--xc", type=float, help="Value of x_c")
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, f"/meta/{myname}/{funcname}", dev=args.develop)
        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    size = np.prod(file["param"]["shape"][...])
                    idx = []
                    xmin = []
                    smax = 0
                if myname not in file:
                    assert not any(m in file for m in m_exclude), "Wrong file type"
                    continue
                res = file[myname]
                idx += [res["idx"][...]]
                xmin += [res["xmin"][...]]
                smax = max(smax, res["xmin"].size)

        opts = dict(bins=args.nbins, mode="log", integer=True)
        A_bin_edges = enstat.histogram.from_data(np.array([1, size]), **opts).bin_edges
        S_bin_edges = enstat.histogram.from_data(np.array([1, smax]), **opts).bin_edges
        A_bins = A_bin_edges.size - 1
        S_bins = S_bin_edges.size - 1
        x0_list = args.xc - np.logspace(-4, np.log10(args.xc), args.ndx)
        output["x0"] = x0_list
        output["files"] = sorted([f.name for f in args.files])

        for x0 in tqdm.tqdm(x0_list):
            S = []
            A = []
            for x, i in zip(xmin, idx):
                si, ai = epm.segment_avalanche(x0 >= x, i)
                S += si.tolist()
                A += ai.tolist()

            S = np.array(S)
            A = np.array(A)

            if S.size == 0 or np.all(np.equal(A, size)):
                s_x = np.NaN * np.ones(S_bins)
                s_p = np.NaN * np.ones(S_bins)
                s_m = np.NaN * np.ones(5)
                a_x = np.NaN * np.ones(A_bins)
                a_p = np.NaN * np.ones(A_bins)
                a_m = np.NaN * np.ones(5)
            else:
                hist = enstat.histogram(bin_edges=S_bin_edges)
                hist += S
                s_x = hist.x
                s_p = hist.p
                m = np.mean(S)
                s_m = np.array([m] + [stats.moment(S, moment=i) for i in range(2, 6)])

                hist = enstat.histogram(bin_edges=A_bin_edges)
                hist += A
                a_x = hist.x
                a_p = hist.p
                m = np.mean(A)
                a_m = np.array([m] + [stats.moment(A, moment=i) for i in range(2, 6)])

            with g5.ExtendableSlice(output, "/S/pdf/x", [S_bins], np.float64) as dset:
                dset += s_x
            with g5.ExtendableSlice(output, "/S/pdf/p", [S_bins], np.float64) as dset:
                dset += s_p
            with g5.ExtendableSlice(output, "/S/moments", [5], np.float64) as dset:
                dset += s_m
            with g5.ExtendableSlice(output, "/A/pdf/x", [A_bins], np.float64) as dset:
                dset += a_x
            with g5.ExtendableSlice(output, "/A/pdf/p", [A_bins], np.float64) as dset:
                dset += a_p
            with g5.ExtendableSlice(output, "/A/moments", [5], np.float64) as dset:
                dset += a_m
