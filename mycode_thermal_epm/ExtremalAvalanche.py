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
from . import storage
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
    Copy ``/param`` and ``/restart``.
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
        dest["restart"]["nstep"] = 0
        meta = tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)
        meta.attrs["t"] = src["restart"]["t"][...]
        meta.attrs["snapshot"] = src[Extremal.m_name]["t"].size - 1

        for key in ["epsp", "sigma", "sigmay"]:
            assert np.all(dest["restart"][key][...] == src[Extremal.m_name][key][-1, ...])
        for key in ["t", "state"]:
            assert np.all(dest["restart"][key][...] == src[Extremal.m_name][key][-1])


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
        system = Extremal.allocate_System(file)
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
        else:
            res = file[m_name]
            assert restart["nstep"][...] == res["idx"].size == res["xmin"].size

        for _ in tqdm.tqdm(range(args.ninc // args.ncache), desc=str(args.file)):
            measurement = epm.Avalanche()
            measurement.makeWeakestFailureSteps(system, args.ncache, allow_stable=True)
            with g5.ExtendableList(res, "idx", np.uint64) as dset:
                dset += measurement.idx
            with g5.ExtendableList(res, "xmin", np.float64) as dset:
                dset += measurement.x

            Preparation.dump_restart(restart, system)
            restart["nstep"][...] += args.ncache


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
    parser.add_argument("--nbins", type=int, help="Number of bins", default=60)
    parser.add_argument("--ndx", type=int, help="Number of x_c - x_0 to sample", default=100)
    parser.add_argument("--xc", type=float, help="Value of x_c")
    parser.add_argument(
        "--means", type=int, default=4, help="Compute <S, A, ell>**(i + 1) for i in range(means)"
    )
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
                    L = np.max(file["param"]["shape"][...])
                    N = np.prod(file["param"]["shape"][...])
                    smax = 0
                    xmin = np.inf
                    xmax = -np.inf
                if myname not in file:
                    assert not any(m in file for m in m_exclude), "Wrong file type"
                    continue
                res = file[myname]
                x = res["xmin"][...]
                smax = max(smax, x.size)
                xmin = min(xmin, np.min(x))
                xmax = max(xmax, np.max(x))

        opts = dict(bins=args.nbins, mode="log", integer=True)
        x_bin_edges = enstat.histogram.from_data(np.array([xmin, xmax]), bins=500).bin_edges
        S_bin_edges = enstat.histogram.from_data(np.array([1, smax]), **opts).bin_edges
        A_bin_edges = enstat.histogram.from_data(np.array([1, N]), **opts).bin_edges
        ell_bin_edges = enstat.histogram.from_data(np.array([1, L]), **opts).bin_edges
        if args.xc is None:
            n = args.ndx // 2
            x0_list = np.linspace(xmin, xmax, n)[:-2]
            x0_list = np.concatenate((x0_list, np.linspace(x0_list[-1], xmax, n + 3)[1:]))
        else:
            x0_list = args.xc - np.logspace(-4, np.log10(args.xc), args.ndx)
            x0_list = np.sort(np.concatenate(([args.xc], x0_list)))
            output["xc"] = args.xc
        output["x0"] = x0_list
        output["files"] = sorted([f.name for f in args.files])

        x_hist = enstat.histogram(bin_edges=x_bin_edges)
        S_hist = [enstat.histogram(bin_edges=S_bin_edges) for _ in x0_list]
        A_hist = [enstat.histogram(bin_edges=A_bin_edges) for _ in x0_list]
        ell_hist = [enstat.histogram(bin_edges=ell_bin_edges) for _ in x0_list]
        S_mean = [[enstat.scalar(dtype=int) for _ in range(args.means)] for _ in x0_list]
        A_mean = [[enstat.scalar(dtype=int) for _ in range(args.means)] for _ in x0_list]
        ell_mean = [[enstat.scalar(dtype=int) for _ in range(args.means)] for _ in x0_list]
        fractal_A = [enstat.binned(bin_edges=A_bin_edges, names=["A", "S"]) for _ in x0_list]
        fractal_ell = [enstat.binned(bin_edges=ell_bin_edges, names=["ell", "S"]) for _ in x0_list]

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                res = file[myname]
                idx = res["idx"][...]
                xmin = res["xmin"][...]
                x_hist += xmin
                for i, x0 in enumerate(tqdm.tqdm(x0_list)):
                    S, A, _ = epm.segment_avalanche(x0 >= xmin, idx, first=False, last=False)
                    ell = np.sqrt(A)
                    A_hist[i] += A
                    S_hist[i] += S
                    ell_hist[i] += ell
                    fractal_A[i].add_sample(A, S)
                    fractal_ell[i].add_sample(ell, S)
                    S = S.astype(int).astype("object")
                    A = A.astype(int).astype("object")
                    for p in range(args.means):
                        S_mean[i][p] += S ** (p + 1)
                        A_mean[i][p] += A ** (p + 1)
                        ell_mean[i][p] += ell ** (p + 1)

            for n, v in zip(["x_hist"], [x_hist]):
                storage.dump_overwrite(output, f"/{n}/bin_edges", v.bin_edges)
                storage.dump_overwrite(output, f"/{n}/count", v.count)

            for n, v in zip(["S_hist", "A_hist", "ell_hist"], [S_hist, A_hist, ell_hist]):
                storage.dump_overwrite(output, f"/{n}/bin_edges", [i.bin_edges for i in v])
                storage.dump_overwrite(output, f"/{n}/count", [i.count for i in v])

            for n, v in zip(["S_mean", "A_mean", "ell_mean"], [S_mean, A_mean, ell_mean]):
                storage.dump_overwrite(
                    output,
                    f"/{n}/first",
                    np.array([[int(m.first) for m in i] for i in v], dtype=np.float64),
                )
                storage.dump_overwrite(
                    output,
                    f"/{n}/second",
                    np.array([[int(m.second) for m in i] for i in v], dtype=np.float64),
                )
                storage.dump_overwrite(
                    output,
                    f"/{n}/norm",
                    np.array([[int(m.norm) for m in i] for i in v], dtype=np.float64),
                )

            for n, v in zip(["fractal_A", "fractal_ell"], [fractal_A, fractal_ell]):
                for key in ["S", n.split("_")[1]]:
                    storage.dump_overwrite(output, f"/{n}/{key}/first", [i[key].first for i in v])
                    storage.dump_overwrite(output, f"/{n}/{key}/second", [i[key].second for i in v])
                    storage.dump_overwrite(output, f"/{n}/{key}/norm", [i[key].norm for i in v])

            output.flush()
