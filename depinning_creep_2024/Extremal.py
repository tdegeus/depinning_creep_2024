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
from . import Thermal
from . import storage
from . import tools
from ._version import version
from .tools import MyFmt

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
    temp_file = Thermal._upgrade_data(filename, temp_dir, m_name, upgrade_meta=False)

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
        kwargs = dict(shape=[n], chunks=tools.default_chunks([n]))
        with g5.ExtendableSlice(group, "xmin", dtype=np.float64, **kwargs) as dset:
            dset[0, :] = src["ExtremalAvalanche"]["xmin"][...]
        with g5.ExtendableSlice(group, "idx", dtype=np.uint64, **kwargs) as dset:
            dset[0, :] = src["ExtremalAvalanche"]["idx"][...]
        with g5.ExtendableList(group, "t0", dtype=np.float64, chunks=(16,)) as dset:
            dset[0] = t
        with g5.ExtendableList(group, "S", dtype=np.uint64, chunks=(16,)) as dset:
            dset[0] = n
        with g5.ExtendableList(group, "index_snapshot", dtype=np.int64, chunks=(16,)) as dset:
            dset[0] = index_snapshot

    return Preparation._upgrade_metadata(temp_file, temp_dir)


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


def EnsembleAvalanches_x0(cli_args: list = None, myname=m_name):
    """
    Calculate properties of avalanches.
    -   Avalanches are segmented by using an arbitrary "x0".
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output file",
        default="EnsembleAvalanches_x0.h5",
    )
    parser.add_argument("--xc", type=float, help="Value of x_c")
    parser.add_argument("--ndx", type=int, help="Number of x_c - x_0 to sample", default=100)
    parser.add_argument("--bins", type=int, help="Number of bins P(S), P(A), P(ell)", default=60)
    parser.add_argument(
        "--means", type=int, default=4, help="Compute <S, A, ell>**(i + 1) for i in range(means)"
    )
    parser.add_argument("info", type=pathlib.Path, help="EnsembleInfo: read files")

    args = tools._parse(parser, cli_args)
    assert args.info.exists()
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.info) as file:
        root = args.info.parent
        files = [root / f for f in file["files"].asstr()[...]]

    files = [f for f in files if f.exists()]
    assert len(files) > 0

    # allocate statistics
    for ifile, f in enumerate(tqdm.tqdm(files)):
        with h5py.File(f) as file:
            if ifile == 0:
                L = np.max(file["param"]["shape"][...])
                N = np.prod(file["param"]["shape"][...])
                alpha = file["param"]["alpha"][...]
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

    if args.xc is None:
        n = args.ndx // 2
        x0_list = np.linspace(xmin, xmax, n)[:-2]
        x0_list = np.concatenate((x0_list, np.linspace(x0_list[-1], xmax, n + 3)[1:]))
    else:
        x0_list = args.xc - np.logspace(-4, np.log10(args.xc), args.ndx)
        x0_list = np.sort(np.concatenate(([args.xc], x0_list)))
        xc_xmin_hist = enstat.histogram(
            bin_edges=enstat.histogram.from_data([1e-6, 1e0], mode="log", bins=500).bin_edges,
            bound_error="ignore",
        )
        if np.isclose(alpha, 1.5):
            limit = [1e-9, 1e0]
        else:
            limit = [(1e-6) ** alpha, 1e0]
        Ec_Emin_hist = enstat.histogram(
            bin_edges=enstat.histogram.from_data(limit, mode="log", bins=500).bin_edges,
            bound_error="ignore",
        )

    opts = dict(bins=args.bins, mode="log", integer=True)
    measurement = Thermal.MeasureAvalanches(
        n=len(x0_list),
        S_bin_edges=enstat.histogram.from_data(np.array([1, smax]), **opts).bin_edges,
        A_bin_edges=enstat.histogram.from_data(np.array([1, N]), **opts).bin_edges,
        ell_bin_edges=enstat.histogram.from_data(np.array([1, L]), **opts).bin_edges,
        n_moments=args.means,
    )
    xmin_hist = enstat.histogram(
        bin_edges=enstat.histogram.from_data(np.array([xmin, xmax]), bins=500).bin_edges,
        bound_error="ignore",
    )

    # collect statistics
    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(myname, funcname), dev=args.develop)
        with h5py.File(files[0]) as file:
            g5.copy(file, output, ["/param"])

        output["/settings/files"] = sorted([f.name for f in files])
        output["/settings/x0"] = x0_list
        if args.xc is not None:
            output["/settings/xc"] = args.xc

        for ifile, f in enumerate(tqdm.tqdm(files)):
            with h5py.File(f) as file:
                res = file[myname]["avalanches"]
                for iava in range(res["xmin"].shape[0]):
                    idx = res["idx"][iava, ...]
                    xmin = res["xmin"][iava, ...]
                    xmin_hist += xmin
                    if args.xc is not None:
                        xc_xmin_hist += args.xc - xmin
                        Ec_Emin_hist += args.xc**alpha - (np.abs(xmin) ** alpha * np.sign(xmin))

                    for i0, x0 in enumerate(tqdm.tqdm(x0_list)):
                        S, A, _ = epm.segment_avalanche(x0 >= xmin, idx, first=False, last=False)
                        ell = np.sqrt(A)
                        measurement.add_sample(i0, S, ell, A)

            names = ["xmin_hist"]
            values = [xmin_hist]
            if args.xc is not None:
                names.append("xc_xmin_hist")
                values.append(xc_xmin_hist)
                names.append("Ec_Emin_hist")
                values.append(Ec_Emin_hist)

            for name, value in zip(names, values):
                value = dict(value)
                for key in value:
                    storage.dump_overwrite(output, f"/data/{name}/{key}", value[key])

            measurement.store(file=output, root="/data")
            output.flush()
