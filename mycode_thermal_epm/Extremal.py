import argparse
import inspect
import logging
import pathlib
import textwrap

import enstat
import GooseEPM as epm
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import Preparation
from . import storage
from . import Thermal
from . import tools
from ._version import version

f_info = "EnsembleInfo.h5"
m_name = "Extremal"
m_exclude = ["AQS", "ExtremalAvalanche", "Thermal"]


def _upgrade_data(
    filename: pathlib.Path, temp_dir: pathlib.Path, insert: pathlib.Path
) -> pathlib.Path | None:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :param insert: ExtremalAvalanche to insert.
    :return: New file in temporary directory if the data is upgraded, ``None`` otherwise.
    """
    temp_file = Thermal._upgrade_data(filename, temp_dir, m_name)

    if temp_file is None:
        return None

    with h5py.File(insert) as src, h5py.File(temp_file, "a") as dst:
        g5.copy(src, dst, "/meta/ExtremalAvalanche/BranchExtremal")
        g5.copy(src, dst, "/meta/ExtremalAvalanche/Run")
        t = src["/meta/ExtremalAvalanche/BranchExtremal"].attrs["t"]
        i = src["/meta/ExtremalAvalanche/BranchExtremal"].attrs["snapshot"]
        n = src["ExtremalAvalanche"]["idx"].size
        index_snapshot = -1

        group = dst[m_name]["snapshots"]
        if np.isclose(group["t"][i], t) and src["restart"]["t"][...] > group["t"][-1]:
            # presume that this snapshot was at the end of the registered avalanche
            system = Thermal.allocate_System(dst, -1, m_name)
            system = Preparation.load_snapshot(None, src["restart"], system)
            index_snapshot = group["S"].size
            Thermal.dump_snapshot(index_snapshot, group, system, n, 0)
        else:
            logging.warning(f"Restart of avalanche not possible: {filename}")

        # only one avalanche was registered
        group = dst[m_name].create_group("avalanches")
        with g5.ExtendableSlice(group, "xmin", dtype=np.float64, shape=[n]) as dset:
            dset[0, :] = src["ExtremalAvalanche"]["xmin"][...]
        with g5.ExtendableSlice(group, "idx", dtype=np.uint64, shape=[n]) as dset:
            dset[0, :] = src["ExtremalAvalanche"]["idx"][...]
        with g5.ExtendableList(group, "t0", dtype=np.float64) as dset:
            dset[0] = t
        with g5.ExtendableList(group, "S", dtype=np.uint64) as dset:
            dset[0] = n
        with g5.ExtendableList(group, "index_snapshot", dtype=np.int64) as dset:
            dset[0] = index_snapshot

    return temp_file


def UpgradeData(cli_args: list = None) -> None:
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, m_name, _upgrade_data, combine=True)


def BranchPreparation(cli_args: list = None) -> None:
    return Thermal.BranchPreparation(cli_args, m_name)


def Run(cli_args: list = None) -> None:
    return Thermal.Run(cli_args, m_name)


def EnsembleInfo(cli_args: list = None) -> None:
    return Thermal.EnsembleInfo(cli_args, m_name)


def EnsembleStructure(cli_args: list = None) -> None:
    return Thermal.EnsembleStructure(cli_args, m_name)


def Plot(cli_args: list = None) -> None:
    return Thermal.Plot(cli_args, m_name)


def EnsembleAvalanches(cli_args: list = None, myname=m_name):
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
    parser.add_argument("--nbins", type=int, help="Number of bins P(S), P(A), P(ell)", default=60)
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
        tools.create_check_meta(output, tools.path_meta(myname, funcname), dev=args.develop)

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
                res = file[myname]["avalanches"]
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
                res = file[myname]["avalanches"]
                for iava in range(res["xmin"].shape[0]):
                    idx = res["idx"][iava, ...]
                    xmin = res["xmin"][iava, ...]
                    x_hist += xmin
                    for i0, x0 in enumerate(tqdm.tqdm(x0_list)):
                        S, A, _ = epm.segment_avalanche(x0 >= xmin, idx, first=False, last=False)
                        ell = np.sqrt(A)
                        A_hist[i0] += A
                        S_hist[i0] += S
                        ell_hist[i0] += ell
                        fractal_A[i0].add_sample(A, S)
                        fractal_ell[i0].add_sample(ell, S)
                        S = S.astype(int).astype("object")  # to avoid overflow (ell=float)
                        A = A.astype(int).astype("object")
                        for p in range(args.means):
                            S_mean[i0][p] += S ** (p + 1)
                            A_mean[i0][p] += A ** (p + 1)
                            ell_mean[i0][p] += ell ** (p + 1)

            for name, value in zip(["x_hist"], [x_hist]):
                value = dict(value)
                for key in ["bin_edges", "count"]:
                    storage.dump_overwrite(output, f"/{name}/{key}", value[key])

            for name, value in zip(["S_hist", "A_hist", "ell_hist"], [S_hist, A_hist, ell_hist]):
                value = [dict(i0) for i0 in value]
                for key in ["bin_edges", "count"]:
                    storage.dump_overwrite(output, f"/{name}/{key}", [i0[key] for i0 in value])

            for name, value in zip(["S_mean", "A_mean", "ell_mean"], [S_mean, A_mean, ell_mean]):
                value = [[dict(p) for p in i0] for i0 in value]
                for key in ["first", "second", "norm"]:
                    storage.dump_overwrite(
                        output, f"/{name}/{key}", [[float(p[key]) for p in i0] for i0 in value]
                    )

            for name, value in zip(["fractal_A", "fractal_ell"], [fractal_A, fractal_ell]):
                for key in ["S", name.split("_")[1]]:
                    storage.dump_overwrite(
                        output, f"/{name}/{key}/first", [i0[key].first for i0 in value]
                    )
                    storage.dump_overwrite(
                        output, f"/{name}/{key}/second", [i0[key].second for i0 in value]
                    )
                    storage.dump_overwrite(
                        output, f"/{name}/{key}/norm", [i0[key].norm for i0 in value]
                    )

            output.flush()
