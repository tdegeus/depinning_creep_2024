import argparse
import inspect
import logging
import pathlib
import sys
import textwrap
import time

import enstat
import GooseEPM as epm
import GooseEYE as eye
import GooseHDF5 as g5
import GooseSLURM as slurm
import h5py
import numpy as np
import tqdm

from . import Preparation
from . import storage
from . import tag
from . import tools
from ._version import version
from .Preparation import data_version

f_info = "EnsembleInfo.h5"
f_structure = "EnsembleStructure.h5"
f_avalanches = "EnsembleAvalanches.h5"
m_name = "Thermal"
m_exclude = ["AQS", "Extremal", "Preparation", "Thermal"]


def _upgrade_data(
    filename: pathlib.Path, temp_dir: pathlib.Path, myname: str = m_name
) -> pathlib.Path | None:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :param myname: Module name.
    :return: New file in temporary directory if the data is upgraded, ``None`` otherwise.
    """
    assert myname in ["Thermal", "Extremal"]

    with h5py.File(filename) as src:
        assert not any(n in src for n in [i for i in m_exclude if i != myname])
        ver = Preparation.get_data_version(src)

    if tag.greater_equal(ver, "3.0"):
        return None

    assert tag.equal(ver, "2.0")

    temp_file = temp_dir / "new_file.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        g5.copy(src, dst, ["/meta", "/param"])
        dst["/param/seed"] = src["/restart/state"].attrs["seed"]
        dst["/param/data_version"][...] = data_version
        N = np.prod(dst["/param/shape"][...])

        # copy snapshots and avalanches
        rename = []
        if myname == m_name:
            rename.append([f"/{myname}/T", f"/{myname}/avalanches/t"])
            rename.append([f"/{myname}/idx", f"/{myname}/avalanches/idx"])
        for key in ["epsp", "sigma", "sigmay", "state", "t"]:
            rename.append([f"/{myname}/{key}", f"/{myname}/snapshots/{key}"])
        for entry in rename:
            g5.copy(src, dst, *entry)
        if f"/{myname}/idx_ignore" in src:
            g5.copy(src, dst, f"/{myname}/idx_ignore", f"/{myname}/avalanches/idx_ignore")

        # add new datasets with minimal data
        n = dst[f"/{myname}/snapshots/t"].size
        with g5.ExtendableList(dst[f"/{myname}/snapshots"], "S", np.uint64) as dset:
            dset.append(np.zeros(n, dtype=np.uint64))

        with g5.ExtendableList(dst[f"/{myname}/snapshots"], "index_avalanche", np.int64) as dset:
            dset.append(-1 * np.ones(n, dtype=np.int64))

        if myname == m_name:
            group = dst[myname]["avalanches"]
            n = group["idx"].shape[0]
            with g5.ExtendableList(group, "index_snapshot", np.int64) as dset:
                dset.append(-1 * np.ones(n, dtype=np.int64))
            with g5.ExtendableList(group, "t0", np.float64) as dset:
                dset.append(np.zeros(n, dtype=np.float64))
            with g5.ExtendableList(group, "S", np.uint64) as dset:
                dset.append(n * np.ones(n, dtype=np.uint64))

        group = dst[myname]["snapshots"]
        group.attrs["preparation"] = 100 * N
        group.attrs["interval"] = 100 * N

        if "restart" not in src:
            dst[myname].create_group("lock")
            logging.warning(f"No restart found: {filename}")
        elif np.all(src["restart"]["epsp"][...] > group["epsp"][-1, ...]):
            system = allocate_System(dst, -1, myname)
            system = Preparation.load_snapshot(None, src["restart"], system)
            dump_snapshot(group["S"].size, group, system, 0, -1)
        elif np.all(src["restart"]["epsp"][...] == group["epsp"][-1, ...]):
            pass
        else:
            dst[myname].create_group("lock")
            logging.warning(f"Restart not possible: {filename}")

        Preparation.check_copy(src, dst, rename=rename, allow={"->": ["/restart"]})

    return temp_file


def UpgradeData(cli_args: list = None) -> None:
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, m_name, _upgrade_data)


def allocate_System(file: h5py.File, index: int, myname: str = m_name) -> epm.SystemClass:
    """
    Allocate the system, and restore snapshot.

    :param param: Opened file.
    :param index: Index of the snapshot to load.
    :param myname: Name of the module.
    :return: System.
    """
    system = Preparation.allocate_System(
        group=file["param"], random_stress=False, thermal="temperature" in file["param"]
    )
    Preparation.load_snapshot(index=index, group=file[myname]["snapshots"], system=system)
    system.sigmabar = file["param"]["sigmabar"][...]
    if myname == m_name:
        system.temperature = file["param"]["temperature"][...]
    return system


def BranchPreparation(cli_args: list = None, myname: str = m_name) -> None:
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

    parser.add_argument("--sigmabar", type=float, default=0.3, help="Stress")
    if myname == m_name:
        parser.add_argument("--temperature", type=float, required=True, help="Temperature")

    parser.add_argument(
        "--interval-preparation", type=int, help="#steps before first snapshot (default ``100 N``)"
    )
    parser.add_argument(
        "--interval-snapshot", type=int, help="#steps between avalanches (default ``100 N``)"
    )
    parser.add_argument(
        "--interval-avalanche", type=int, help="#steps inside avalanches (default ``20 N``)"
    )

    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(n in src for n in [i for i in m_exclude if i != "Preparation"])
        g5.copy(src, dest, ["/meta", "/param"])
        tools.create_check_meta(dest, tools.path_meta(myname, funcname), dev=args.develop)
        dest["param"]["sigmabar"] = args.sigmabar
        if myname == m_name:
            dest["param"]["temperature"] = args.temperature

        n = np.prod(dest["param"]["shape"][...])
        if args.interval_preparation is None:
            args.interval_preparation = 100 * n
        if args.interval_snapshot is None:
            args.interval_snapshot = 100 * n
        if args.interval_avalanche is None:
            args.interval_avalanche = 20 * n

        g5.copy(src, dest, f"/{Preparation.m_name}/snapshots", f"/{myname}/snapshots")
        group = dest[myname]["snapshots"]
        group.attrs["preparation"] = args.interval_preparation
        group.attrs["interval"] = args.interval_snapshot
        g5.ExtendableList(group, "S", np.uint64).setitem(index=0, data=0).flush()
        g5.ExtendableList(
            file=group,
            key="index_avalanche",
            dtype=np.int64,
            desc=">= 0 if snapshot corresponds to start of avalanche",
        ).setitem(index=0, data=-1).flush()

        group = dest[myname].create_group("avalanches")
        g5.ExtendableList(group, "S", np.uint64).setitem(index=0, data=0).flush()
        g5.ExtendableList(group, "t0", np.float64).flush()
        g5.ExtendableList(
            file=group,
            key="index_snapshot",
            dtype=np.int64,
            desc=">= 0 if snapshot at the last event is stored",
        ).flush()
        g5.ExtendableSlice(group, "idx", dtype=np.uint64, shape=[args.interval_avalanche])
        if myname == m_name:
            g5.ExtendableSlice(group, "t", dtype=np.float64, shape=[args.interval_avalanche])
        else:
            g5.ExtendableSlice(group, "xmin", dtype=np.float64, shape=[args.interval_avalanche])


def dump_snapshot(
    index: int, group: h5py.Group, system: epm.SystemClass, S: int, index_avalanche: int
) -> None:
    """
    Add/overwrite snapshot of the current state (fully recoverable).

    :param index: Index of the snapshot to overwrite.
    :param group: Group to store the snapshot in.
    :param system: System.
    :param S: #failures since the last snapshot.
    :param index_avalanche:
        Index of the avalanche after which the snapshot was taken
        (``-1`` if not taken directly after an avalanche).
    """
    Preparation.dump_snapshot(index, group, system)

    with g5.ExtendableList(group, "S") as dset:
        dset[index] = S

    with g5.ExtendableList(group, "index_avalanche") as dset:
        dset[index] = index_avalanche


def new_avalanche(index: int, group: h5py.Group, t0: float, index_snapshot: int):
    """
    Open new avalanche.

    :param t0: Time before starting to measure the avalanche.
    :param index_snapshot:
        Index of the snapshot at the start of the avalanche
        (``-1`` if no snapshot is stored).
    """
    assert group["t0"].size == index
    assert group["index_snapshot"].size == index

    with g5.ExtendableList(group, "t0") as dset:
        dset[index] = t0

    with g5.ExtendableList(group, "index_snapshot") as dset:
        dset[index] = index_snapshot


def dump_avalanche(
    index: int, start_column: int, group: h5py.Group, avalanche: epm.Avalanche, myname: str
) -> None:
    """
    Add (part of) avalanche measurement.

    :param index: Index of the snapshot to overwrite ("row").
    :param start_column: Start index of the items in the avalanche sequence ("column").
    :param group: Group to store the snapshot in.
    :param avalanche: Measurement.
    :param myname: Module name.
    """
    assert group["t0"].size == index + 1
    assert group["index_snapshot"].size == index + 1

    stop_column = start_column + avalanche.idx.size

    with g5.ExtendableList(group, "S") as dset:
        dset[index] = stop_column

    with g5.ExtendableSlice(group, "idx") as dset:
        dset[index, start_column:stop_column] = avalanche.idx

    if myname == m_name:
        with g5.ExtendableSlice(group, "t") as dset:
            dset[index, start_column:stop_column] = avalanche.t
    else:
        with g5.ExtendableSlice(group, "xmin") as dset:
            dset[index, start_column:stop_column] = avalanche.x


def Run(cli_args: list = None, myname: str = m_name) -> None:
    """
    Run simulation at fixed stress.

    0.  Run ``file["/Thermal/snapshots"].attrs["preparation"]`` events and take snapshot.
    1.  Run ``file["/Thermal/avalanches/idx"].shape[1]`` events, and record:
        a.  Sequence of failing blocks:
            -   time ``t``
            -   flat index of failing block ``idx``
        b.  Snapshot.
    2.  Run ``file["/Thermal/snapshots"].attrs["interval"]`` events and take snapshot.
    3.  Repeat from 1.
    """
    tic = time.time()

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
        "--buffer",
        type=slurm.duration.asSeconds,
        default=30 * 60,
        help="Write interval to write partial (restartable) data",
    )
    parser.add_argument(
        "--walltime",
        type=slurm.duration.asSeconds,
        default=sys.maxsize,
        help="Walltime at which to stop",
    )
    parser.add_argument(
        "--save-duration",
        type=slurm.duration.asSeconds,
        default=0,
        help="Duration to reserve for saving data",
    )
    parser.add_argument(
        "-n", "--snapshots", type=int, default=100, help="Total #snapshots to sample"
    )
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    args.walltime -= args.save_duration
    assert args.file.exists()

    with h5py.File(args.file) as file:
        assert Preparation.get_data_version(file) == data_version
        assert "lock" not in file[myname], "File locked: restart not possible"

    class Method:
        def __init__(self, file: h5py.File):
            grp_snap = file[myname]["snapshots"]
            grp_ava = file[myname]["avalanches"]
            self.interval_preparation = int(grp_snap.attrs["preparation"])
            self.interval_snapshot = int(grp_snap.attrs["interval"])
            self.interval_avalanche = grp_ava["idx"].shape[1]
            self.index_snapshot = grp_snap["S"].size - 1
            self.index_avalanche = int(grp_snap["index_avalanche"][self.index_snapshot])
            assert self.interval_avalanche > 0 or self.interval_snapshot > 0
            if self.index_snapshot == 0:
                assert self.index_avalanche == -1
                self.target = self.interval_preparation
            elif self.index_avalanche == -1:
                self.target = self.interval_snapshot
            else:
                self.target = self.interval_avalanche

        def next_snapshot(self):
            if self.interval_snapshot == 0:
                return self.next_avalanche()
            self.target = self.interval_snapshot
            self.index_snapshot = file[myname]["snapshots"]["S"].size
            self.index_avalanche = -1
            return 0

        def next_avalanche(self):
            if self.interval_avalanche == 0:
                return self.next_snapshot()
            self.target = self.interval_avalanche
            self.index_snapshot = file[myname]["snapshots"]["S"].size
            self.index_avalanche = file[myname]["avalanches"]["idx"].shape[0]
            return 0

        def increment(self, s: int) -> int:
            if s < self.target:
                return s
            if self.index_avalanche == -1:
                return self.next_avalanche()
            return self.next_snapshot()

    duration_snapshot = enstat.scalar()
    duration_avalanche = enstat.scalar()

    with h5py.File(args.file, "a") as file:
        tools.create_check_meta(file, tools.path_meta(myname, funcname), dev=args.develop)
        grp_snap = file[myname]["snapshots"]
        grp_ava = file[myname]["avalanches"]
        i = grp_snap["S"].size - 1
        s = int(grp_snap["S"][i])
        system = allocate_System(file, i, myname)
        method = Method(file)
        ava = epm.Avalanche()

        if myname == m_name:
            my_steps_system = system.makeThermalFailureSteps_chrono
            my_steps_avalanche = ava.makeThermalFailureSteps_chrono
            my_steps_options = dict(elapsed=args.buffer)
        else:
            my_steps_system = system.makeWeakestFailureSteps_chrono
            my_steps_avalanche = ava.makeWeakestFailureSteps_chrono
            my_steps_options = dict(elapsed=args.buffer, allow_stable=True)

        for _ in tqdm.tqdm(range(args.snapshots - method.index_snapshot), desc=str(args.file)):
            s = method.increment(s)
            i = method.index_snapshot
            if i == args.snapshots:
                return
            while s < method.target:
                tici = time.time()
                ds = method.target - s
                if method.index_avalanche == -1:
                    if duration_snapshot.mean() > args.walltime - (time.time() - tic):
                        return
                    s += my_steps_system(n=ds, **my_steps_options)
                    if time.time() - tic >= args.walltime:
                        return
                    dump_snapshot(i, grp_snap, system, s, method.index_avalanche)
                    duration_snapshot += time.time() - tici
                    duration_snapshot.mean()
                else:
                    if duration_avalanche.mean() > args.walltime - (time.time() - tic):
                        return
                    if s == 0:
                        new_avalanche(method.index_avalanche, grp_ava, system.t, i - 1)
                    my_steps_avalanche(system, n=ds, **my_steps_options)
                    if time.time() - tic >= args.walltime:
                        return
                    dump_avalanche(method.index_avalanche, s, grp_ava, ava, myname)
                    dump_snapshot(i, grp_snap, system, s + ava.idx.size, method.index_avalanche)
                    s += ava.idx.size
                    duration_avalanche += time.time() - tici
                    duration_avalanche.mean()


def _index_avalanches(group: h5py.Group) -> list[int]:
    """
    Get list of indices of avalanches.

    :param group: Root of results.
    :return: slice
    """
    if "lock" in group:
        if "idx_ignore" in group:
            return np.setdiff1d(
                np.arange(group["avalanches"]["idx"].shape[0]), group["idx_ignore"][...]
            )
    return np.arange(group["avalanches"]["idx"].shape[0])


def _index_snapshots(group: h5py.Group) -> list[int]:
    """
    Get list of indices of snapshots.

    :param group: Root of results.
    :return: slice
    """
    if "lock" in group:
        return np.arange(group["snapshots"]["S"].size)
    return np.arange(group["snapshots"]["S"].size)


def EnsembleInfo(cli_args: list = None, myname: str = m_name) -> None:
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
    parser.add_argument("--nbin-x", type=int, default=2000, help="#bins for P(x)")
    parser.add_argument("--nbin-t", type=int, default=10000, help="#bins for tau")
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    if myname == m_name:
        tmin = np.inf
        tmax = 0
        nsim = 0
        for f in tqdm.tqdm(args.files):
            with h5py.File(f) as file:
                assert Preparation.get_data_version(file) == data_version
                assert not any(n in file for n in [i for i in m_exclude if i != myname])
                if f"/{myname}/avalanches/t" not in file:
                    continue
                nsim += 1
                avalanches = file[myname]["avalanches"]
                for i in range(avalanches["t"].shape[0]):
                    tmin = min(tmin, avalanches["t"][i, 0] - avalanches["t0"][i])
                    tmax = max(tmax, avalanches["t"][i, -1] - avalanches["t0"][i])

        if nsim == 0:
            return

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(myname, funcname), dev=args.develop)
        output["files"] = sorted([f.name for f in args.files])
        output.create_group("hist_x")

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])

                # allocate output
                if ifile == 0:
                    N = np.prod(file["param"]["shape"][...])
                    pdfx = enstat.histogram(
                        bin_edges=np.linspace(0, 3, args.nbin_x), bound_error="ignore"
                    )

                if ifile == 0 and myname == m_name:
                    binned = enstat.binned(
                        bin_edges=enstat.histogram.from_data(
                            data=np.array([max(tmin, 1e-9), tmax]), bins=args.nbin_t, mode="log"
                        ).bin_edges,
                        names=["t", "S", "Ssq", "A", "Asq", "ell"],  # N.B.: ell^2 == A
                        bound_error="ignore",
                    )

                # collect from snapshots
                indices = _index_snapshots(file[myname])
                pdfx += Preparation.compute_x(
                    dynamics=Preparation.get_dynamics(file),
                    sigma=file[myname]["snapshots"]["sigma"][indices, ...],
                    sigmay=file[myname]["snapshots"]["sigmay"][indices, ...],
                ).ravel()

                if myname != m_name:
                    continue

                # collect from avalanches
                indices = _index_avalanches(file[myname])
                avalanches = file[myname]["avalanches"]
                for i in indices:
                    ti = avalanches["t"][i, ...] - avalanches["t0"][i]
                    ai = epm.cumsum_n_unique(avalanches["idx"][i, ...]) / N
                    si = np.arange(1, ti.size + 1) / N
                    if np.all(ti == 0):  # happens for very small temperatures
                        continue
                    binned.add_sample(ti, si, si**2, ai, ai**2, np.sqrt(ai))

            # update output file
            Preparation.store_histogram(output["hist_x"], pdfx)

            if myname == m_name:
                for name in binned.names:
                    for key, value in binned[name]:
                        storage.dump_overwrite(output, f"/restore/{name}/{key}", value)

            output.flush()

        if myname != m_name:
            return

        # compute relaxation time
        t = enstat.static.restore(
            first=output["/restore/t/first"][...],
            norm=output["/restore/t/norm"][...],
        )
        A = enstat.static.restore(
            first=output["/restore/A/first"][...],
            norm=output["/restore/A/norm"][...],
        )
        t.squash(4)  # todo: find something more intelligent
        A.squash(4)  # todo: find something more intelligent
        phi = 1 - A.mean()
        tau = t.mean()
        tau_alpha = tau[np.argmax(phi < 0.5)]
        output["tau_alpha"] = tau_alpha


def EnsembleStructure(cli_args: list = None, myname: str = m_name):
    """
    Extract the structure factor at snapshots.
    See:
    https://doi.org/10.1103/PhysRevB.74.140201
    https://doi.org/10.1103/PhysRevLett.118.147208
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
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="Output file", default=f_structure
    )
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(myname, funcname), dev=args.develop)
        output["files"] = sorted([f.name for f in args.files])

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    data = eye.Structure(shape=file["param"]["shape"][...])

                if f"/{myname}/snapshots/epsp" not in file:
                    continue

                snapshots = file[myname]["snapshots"]
                for i in _index_snapshots(file[myname]):
                    data += snapshots["epsp"][i, ...]

            for name, value in data:
                storage.dump_overwrite(output, f"/restore/{name}", value)

            output.flush()


def Plot(cli_args: list = None, myname: str = m_name) -> None:
    """
    Basic of the ensemble.
    """

    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file) as file:
        realisation = "hist_x" not in file

    if realisation:
        with h5py.File(args.file) as file:
            res = file[myname]["snapshots"]
            u = res["epsp"][-1, ...] + res["sigma"][-1, ...]

        fig, ax = gplt.subplots()
        ax.imshow(u)
        plt.show()

    else:
        with h5py.File(args.file) as file:
            pdfx = enstat.histogram.restore(
                bin_edges=file["hist_x"]["bin_edges"][...],
                count=file["hist_x"]["count"][...],
                count_left=file["hist_x"]["count_left"][...],
                count_right=file["hist_x"]["count_right"][...],
            )

        fig, ax = gplt.subplots()

        ax.plot(pdfx.x, pdfx.p)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$P(x)$")

        plt.show()
        plt.close(fig)


def EnsembleAvalanches(cli_args: list = None, myname=m_name):
    """
    Calculate properties of avalanches at different times compared to an arbitrary reference time.
    The interface is arbitrarily assumed flat.
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
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="Output file", default=f_avalanches
    )
    parser.add_argument(
        "--tau", type=float, nargs=3, default=[-5, 0, 51], help="logspace tau (units of tau_alpha)"
    )
    parser.add_argument("--bins", type=int, help="Number of bins for P(S, A, ell)", default=60)
    parser.add_argument(
        "--means", type=int, default=4, help="Compute <S, A, ell>**(i + 1) for i in range(means)"
    )
    parser.add_argument("info", type=pathlib.Path, help="EnsembleInfo: read tau_alpha")

    args = tools._parse(parser, cli_args)
    assert args.info.exists()
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.info) as file:
        root = args.info.parent
        files = [root / f for f in file["files"].asstr()[...]]
        shape = file["param"]["shape"][...]
        tau_alpha = file["tau_alpha"][...]
        t_measure = np.logspace(args.tau[0], args.tau[1], int(args.tau[2])) * tau_alpha

    files = [f for f in files if f.exists()]
    assert len(files) > 0

    # allocate
    init = False
    for f in files:
        with h5py.File(f) as file:
            if myname not in file:
                assert not any(n in file for n in [i for i in m_exclude if i != myname])
                continue

            avalanches = file[myname]["avalanches"]
            for iava in _index_avalanches(file[myname]):
                t = avalanches["t"][iava, ...] - avalanches["t0"][iava]
                idx = avalanches["idx"][iava, ...]

                structure = [eye.Structure(shape=shape) for _ in t_measure]
                mean_t = [enstat.scalar() for _ in t_measure]

                opts = dict(bins=args.bins, mode="log", integer=True)
                edges = enstat.histogram.from_data(np.array([1, idx.size]), **opts).bin_edges
                hist_S = [enstat.histogram(bin_edges=edges) for _ in t_measure]

                edges = enstat.histogram.from_data(np.array([1, np.prod(shape)]), **opts).bin_edges
                hist_A = [enstat.histogram(bin_edges=edges) for _ in t_measure]

                edges = enstat.histogram.from_data(np.array([1, np.max(shape)]), **opts).bin_edges
                hist_ell = [enstat.histogram(bin_edges=edges) for _ in t_measure]

                n = args.means
                mean_S = [[enstat.scalar(dtype=int) for _ in range(n)] for _ in t_measure]
                mean_A = [[enstat.scalar(dtype=int) for _ in range(n)] for _ in t_measure]
                mean_ell = [[enstat.scalar(dtype=int) for _ in range(n)] for _ in t_measure]

                init = True
                break

            if init:
                break

    assert init, "Did not find any data"

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(myname, funcname), dev=args.develop)
        output["/settings/tau"] = t_measure
        output["/settings/tau_alpha"] = tau_alpha
        with h5py.File(files[0]) as file:
            g5.copy(file, output, ["/param"])

        for ifile, f in enumerate(tqdm.tqdm(files)):
            with h5py.File(f) as file:
                if myname not in file:
                    continue
                avalanches = file[myname]["avalanches"]
                for iava in _index_avalanches(file[myname]):
                    t = avalanches["t"][iava, ...] - avalanches["t0"][iava]
                    idx = avalanches["idx"][iava, ...].astype(int)

                    while True:
                        if t[-1] < t_measure[-1]:
                            break
                        segmenter = epm.allocate_AvalancheSegmenter(shape=shape, idx=idx, t=t)

                        for i0, t0 in enumerate(t_measure):
                            segmenter.advance_to(t0, floor=True)
                            s = segmenter.s.astype(int)
                            structure[i0] += s
                            mean_t[i0] += segmenter.t

                            lab = segmenter.labels.astype(int).ravel()
                            a = np.bincount(lab)
                            s = np.bincount(lab, weights=s.ravel()).astype(int)
                            ell = Preparation.convert_A_to_ell(a, len(shape))
                            keep = a > 0  # merged labels are empty
                            keep[0] = False  # background label=0
                            a = a[keep]
                            s = s[keep]
                            ell = ell[keep]
                            hist_S[i0] += s
                            hist_A[i0] += a
                            hist_ell[i0] += ell
                            s = s.astype(int).astype("object")  # to avoid overflow (ell=float)
                            a = a.astype(int).astype("object")
                            for p in range(args.means):
                                mean_S[i0][p] += s ** (p + 1)
                                mean_A[i0][p] += a ** (p + 1)
                                mean_ell[i0][p] += ell ** (p + 1)

                        if t[-1] <= 2 * tau_alpha:
                            break
                        i = np.argmax(t > 2 * tau_alpha)
                        t = t[i:] - t[i]
                        idx = idx[i:]

            for name, value in zip(["tau"], [mean_t]):
                value = [dict(i0) for i0 in value]
                for key in ["first", "second", "norm"]:
                    storage.dump_overwrite(
                        output,
                        f"/data/{name}/{key}",
                        np.array([float(i0[key]) for i0 in value], dtype=np.float64),
                    )

            for name, value in zip(["structure"], [structure]):
                value = [dict(i0) for i0 in value]
                for key in ["first", "second", "norm"]:
                    storage.dump_overwrite(
                        output,
                        f"/data/{name}/{key}",
                        np.array([i0[key][:, 0] for i0 in value], dtype=np.float64),
                    )

            for name, value in zip(["S_hist", "A_hist", "ell_hist"], [hist_S, hist_A, hist_ell]):
                value = [dict(i0) for i0 in value]
                for key in ["bin_edges", "count"]:
                    storage.dump_overwrite(
                        output,
                        f"/data/{name}/{key}",
                        np.array([i0[key] for i0 in value], dtype=np.float64),
                    )

            for name, value in zip(["S_mean", "A_mean", "ell_mean"], [mean_S, mean_A, mean_ell]):
                value = [[dict(p) for p in i0] for i0 in value]
                for key in ["first", "second", "norm"]:
                    storage.dump_overwrite(
                        output,
                        f"/data/{name}/{key}",
                        np.array([[float(p[key]) for p in i0] for i0 in value], dtype=np.float64),
                    )

            output.flush()
