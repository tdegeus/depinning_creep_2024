import argparse
import inspect
import logging
import pathlib
import sys
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
from .tools import MyFmt

f_info = "EnsembleInfo.h5"
f_structure = "EnsembleStructure.h5"
m_name = "Thermal"
m_exclude = ["AQS", "Extremal", "Preparation", "Thermal"]


def _upgrade_data(
    filename: pathlib.Path, temp_dir: pathlib.Path, _module: str = m_name, upgrade_meta: bool = True
) -> pathlib.Path | None:
    """
    Upgrade data to the current version.

    :param filename: Input filename.
    :param temp_dir: Temporary directory in which any file may be created/overwritten.
    :param _module: Module name.
    :param upgrade_meta: Upgrade metadata.
    :return: New file in temporary directory if the data is upgraded, ``None`` otherwise.
    """
    assert _module in ["Thermal", "Extremal"]

    with h5py.File(filename) as src:
        assert not any(n in src for n in [i for i in m_exclude if i != _module])
        ver = Preparation.get_data_version(src)

    if tag.greater_equal(ver, "3.0"):
        return None

    assert tag.equal(ver, "2.0")

    with h5py.File(filename) as src:
        if _module not in src:
            return None

    temp_file = temp_dir / "new_file.h5"
    with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
        g5.copy(src, dst, ["/meta", "/param"])
        dst["/param/seed"] = src["/restart/state"].attrs["seed"]
        dst["/param/data_version"][...] = data_version
        shape = dst["/param/shape"][...]
        N = np.prod(shape)

        # copy snapshots
        rename = []
        group = dst.create_group(_module).create_group("snapshots")
        n = src[_module]["t"].size
        kwargs = dict(shape=shape, chunks=tools.default_chunks(shape))
        for name, dtype, islist in zip(
            ["epsp", "sigma", "sigmay", "t", "state"],
            [np.float64, np.float64, np.float64, np.float64, np.uint64],
            [False, False, False, True, True],
        ):
            rename.append([f"/{_module}/{name}", f"/{_module}/snapshots/{name}"])
            if islist:
                with g5.ExtendableList(group, name, dtype, chunks=(16,)) as dset:
                    dset[...] = src[f"/{_module}/{name}"][...]
            else:
                with g5.ExtendableSlice(file=group, name=name, dtype=dtype, **kwargs) as dset:
                    for i in range(n):
                        dset[i, ...] = src[f"/{_module}/{name}"][i, ...]

        # copy avalanches
        if "T" in src[_module]:
            group = dst[_module].create_group("avalanches")
            if len(src[_module]["T"].shape) == 2:
                n = src[_module]["T"].shape[0]
                m = src[_module]["T"].shape[1]
            else:
                n = 1
                m = src[_module]["T"].size

            kwargs = dict(shape=[m], chunks=tools.default_chunks([m]))
            for oldname, name, dtype, islist in zip(
                ["T", "idx", "idx_ignore"],
                ["t", "idx", "idx_ignore"],
                [np.float64, np.uint64, np.uint64],
                [False, False, True],
            ):
                if oldname not in src[_module]:
                    continue
                rename.append([f"/{_module}/{oldname}", f"/{_module}/avalanches/{name}"])
                if islist:
                    with g5.ExtendableList(group, name, dtype, chunks=(16,)) as dset:
                        dset[...] = src[f"/{_module}/{oldname}"][...]
                else:
                    with g5.ExtendableSlice(file=group, name=name, dtype=dtype, **kwargs) as dset:
                        for i in range(n):
                            dset[i, ...] = src[f"/{_module}/{oldname}"][i, ...]

        # add new datasets with minimal data
        n = dst[f"/{_module}/snapshots/t"].size
        g = dst[f"/{_module}/snapshots"]
        with g5.ExtendableList(g, "S", np.uint64, chunks=(16,)) as dset:
            dset.append(np.zeros(n, dtype=np.uint64))

        with g5.ExtendableList(g, "index_avalanche", np.int64, chunks=(16,)) as dset:
            dset.append(-1 * np.ones(n, dtype=np.int64))

        if _module == m_name and "T" in src[_module]:
            group = dst[_module]["avalanches"]
            n = group["idx"].shape[0]
            with g5.ExtendableList(group, "index_snapshot", np.int64, chunks=(16,)) as dset:
                dset.append(-1 * np.ones(n, dtype=np.int64))
            with g5.ExtendableList(group, "t0", np.float64, chunks=(16,)) as dset:
                dset.append(np.zeros(n, dtype=np.float64))
            with g5.ExtendableList(group, "S", np.uint64, chunks=(16,)) as dset:
                dset.append(n * np.ones(n, dtype=np.uint64))

        group = dst[_module]["snapshots"]
        group.attrs["preparation"] = 100 * N
        group.attrs["interval"] = 100 * N

        if "restart" not in src:
            dst[_module].create_group("lock")
            logging.warning(f"No restart found: {filename}")
        elif np.all(src["restart"]["t"][...] > group["t"][-1]):
            system = allocate_System(dst, -1, _module)
            system = Preparation.load_snapshot(None, src["restart"], system)
            dump_snapshot(group["S"].size, group, system, 0, -1)
        elif np.all(src["restart"]["t"][...] == group["t"][-1]):
            pass
        else:
            dst[_module].create_group("lock")
            logging.warning(f"Restart not possible: {filename}")

        Preparation.check_copy(
            src, dst, rename=rename, allow={"->": ["/restart", f"/{_module}/mean_epsp"]}
        )

    if upgrade_meta:
        return Preparation._upgrade_metadata(temp_file, temp_dir)
    else:
        return temp_file


@tools.docstring_append_cli()
def UpgradeData(cli_args: list = None, _return_parser: bool = False):
    """
    Upgrade data to the current version.
    """
    return Preparation.UpgradeData(
        cli_args=cli_args,
        _return_parser=_return_parser,
        _module=m_name,
        _upgrade_function=_upgrade_data,
    )


def allocate_System(file: h5py.File, index: int, _module: str = m_name) -> epm.SystemClass:
    """
    Allocate the system, and restore snapshot.

    :param param: Opened file.
    :param index: Index of the snapshot to load.
    :param _module: Name of the module calling this function.
    :return: System.
    """
    system = Preparation.allocate_System(
        group=file["param"], random_stress=False, thermal="temperature" in file["param"]
    )
    Preparation.load_snapshot(index=index, group=file[_module]["snapshots"], system=system)
    system.sigmabar = file["param"]["sigmabar"][...]
    if _module == m_name:
        system.temperature = file["param"]["temperature"][...]
    return system


def BranchPreparation(
    cli_args: list = None, _return_parser: bool = False, _module: str = m_name
) -> None:
    """
    Branch from prepared stress state and add parameters.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--sigmabar", type=float, default=0.3, help="Stress")
    if _module == m_name:
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

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(n in src for n in [i for i in m_exclude if i != "Preparation"])
        g5.copy(src, dest, ["/meta", "/param"])
        tools.create_check_meta(dest, tools.path_meta(_module, funcname), dev=args.develop)
        dest["param"]["sigmabar"] = args.sigmabar
        if _module == m_name:
            dest["param"]["temperature"] = args.temperature

        n = np.prod(dest["param"]["shape"][...])
        if args.interval_preparation is None:
            args.interval_preparation = 100 * n
        if args.interval_snapshot is None:
            args.interval_snapshot = 100 * n
        if args.interval_avalanche is None:
            args.interval_avalanche = 20 * n

        g5.copy(src, dest, f"/{Preparation.m_name}/snapshots", f"/{_module}/snapshots")
        group = dest[_module]["snapshots"]
        group.attrs["preparation"] = args.interval_preparation
        group.attrs["interval"] = args.interval_snapshot
        g5.ExtendableList(group, "S", np.uint64, chunks=(16,)).setitem(index=0, data=0).flush()
        g5.ExtendableList(
            file=group,
            name="index_avalanche",
            dtype=np.int64,
            chunks=(16,),
            attrs={"desc": ">= 0 if snapshot corresponds to start of avalanche"},
        ).setitem(index=0, data=-1).flush()

        if args.interval_avalanche > 0:
            group = dest[_module].create_group("avalanches")
            g5.ExtendableList(group, "S", np.uint64, chunks=(16,)).setitem(index=0, data=0).flush()
            g5.ExtendableList(group, "t0", np.float64, chunks=(16,)).flush()
            g5.ExtendableList(
                file=group,
                name="index_snapshot",
                dtype=np.int64,
                chunks=(16,),
                attrs={"desc": ">= 0 if snapshot at the last event is stored"},
            ).flush()
            s = [args.interval_avalanche]
            kwargs = dict(shape=s, chunks=tools.default_chunks(s))
            g5.ExtendableSlice(group, "idx", dtype=np.uint64, **kwargs)
            if _module == m_name:
                g5.ExtendableSlice(group, "t", dtype=np.float64, **kwargs)
            else:
                g5.ExtendableSlice(group, "xmin", dtype=np.float64, **kwargs)


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
    index: int, start_column: int, group: h5py.Group, avalanche: epm.Avalanche, _module: str
) -> None:
    """
    Add (part of) avalanche measurement.

    :param index: Index of the snapshot to overwrite ("row").
    :param start_column: Start index of the items in the avalanche sequence ("column").
    :param group: Group to store the snapshot in.
    :param avalanche: Measurement.
    :param _module: Module name.
    """
    assert group["t0"].size == index + 1
    assert group["index_snapshot"].size == index + 1

    stop_column = start_column + avalanche.idx.size

    with g5.ExtendableSlice(group, "idx") as dset:
        dset[index, start_column:stop_column] = avalanche.idx

    if _module == m_name:
        with g5.ExtendableSlice(group, "t") as dset:
            dset[index, start_column:stop_column] = avalanche.t
    else:
        with g5.ExtendableSlice(group, "xmin") as dset:
            dset[index, start_column:stop_column] = avalanche.x

    with g5.ExtendableList(group, "S") as dset:
        dset[index] = stop_column


def Run(cli_args: list = None, _return_parser: bool = False, _module: str = m_name) -> None:
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
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
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
        "-n",
        "--snapshots",
        type=int,
        default=112,
        help="Total #snapshots to sample (save storage by using a multiple of 16)",
    )
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    args.walltime -= args.save_duration
    assert args.file.exists()

    with h5py.File(args.file) as file:
        assert Preparation.get_data_version(file) == data_version
        assert "lock" not in file[_module], "File locked: restart not possible"

    class Method:
        def __init__(self, file: h5py.File):
            grp_snap = file[_module]["snapshots"]
            self.interval_preparation = int(grp_snap.attrs["preparation"])
            self.interval_snapshot = int(grp_snap.attrs["interval"])
            self.interval_avalanche = (
                file[_module]["avalanches"]["idx"].shape[1] if "avalanches" in file[_module] else 0
            )
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
            self.index_snapshot = file[_module]["snapshots"]["S"].size
            self.index_avalanche = -1
            return 0

        def next_avalanche(self):
            if self.interval_avalanche == 0:
                return self.next_snapshot()
            self.target = self.interval_avalanche
            self.index_snapshot = file[_module]["snapshots"]["S"].size
            self.index_avalanche = file[_module]["avalanches"]["t0"].size
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
        tools.create_check_meta(file, tools.path_meta(_module, funcname), dev=args.develop)
        grp_snap = file[_module]["snapshots"]
        grp_ava = file[_module]["avalanches"] if "avalanches" in file[_module] else None
        i = grp_snap["S"].size - 1
        s = int(grp_snap["S"][i])
        system = allocate_System(file, i, _module)
        method = Method(file)
        ava = epm.Avalanche()

        if _module == m_name:
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
                    assert not np.isnan(system.t)
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
                    assert not np.isnan(system.t)
                    assert not np.isnan(ava.t[0])
                    if time.time() - tic >= args.walltime:
                        return
                    dump_avalanche(method.index_avalanche, s, grp_ava, ava, _module)
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
    if "idx_ignore" in group["avalanches"]:
        return np.setdiff1d(
            np.arange(group["avalanches"]["idx"].shape[0]), group["avalanches"]["idx_ignore"][...]
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


def EnsembleInfo(
    cli_args: list = None, _return_parser: bool = False, _module: str = m_name
) -> None:
    """
    Basic interpretation of the ensemble.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_info)
    parser.add_argument("--nbin-x", type=int, default=2000, help="#bins for P(x)")
    parser.add_argument("--nbin-t", type=int, default=10000, help="#bins for tau")
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    assert len(args.files) > 0
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.files[0]) as ref_file:
        param_paths = sorted(g5.getdatapaths(ref_file, root="/param"))
        param_paths.remove("/param/seed")
        N = np.prod(ref_file["param"]["shape"][...])
        pdfx = enstat.histogram(bin_edges=np.linspace(0, 3, args.nbin_x), bound_error="ignore")

        tmin = np.inf
        tmax = 0
        nsim = 0
        seeds = []
        for f in tqdm.tqdm(args.files):
            with h5py.File(f) as file:
                assert Preparation.get_data_version(file) == data_version, f"Incompatible data: {f}"

                excl = [i for i in m_exclude if i != _module]
                assert not any(n in file for n in excl), f"Wrong module: {f}"

                eq = sorted(g5.compare(file, ref_file, param_paths)["=="])
                assert eq == param_paths, f"Wrong parameters: {f}"

                seeds.append(file["param"]["seed"][...])

                if _module != m_name:
                    nsim += 1
                    continue

                if f"/{_module}/avalanches/t" not in file:
                    continue

                nsim += 1
                avalanches = file[_module]["avalanches"]
                for i in range(avalanches["t"].shape[0]):
                    tmin = min(tmin, avalanches["t"][i, 0] - avalanches["t0"][i])
                    tmax = max(tmax, avalanches["t"][i, -1] - avalanches["t0"][i])

        assert np.unique(seeds).size == len(seeds), "Duplicate seeds"
        if nsim == 0:
            return

        if _module == m_name:
            binned = enstat.binned(
                bin_edges=enstat.histogram.from_data(
                    data=np.array([max(tmin, 1e-9), tmax]), bins=args.nbin_t, mode="log"
                ).bin_edges,
                names=["t", "S", "Ssq", "A", "Asq", "ell"],  # N.B.: ell^2 == A
                bound_error="ignore",
            )

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(_module, funcname), dev=args.develop)
        output["files"] = sorted([f.name for f in args.files])
        output.create_group("hist_x")
        with h5py.File(args.files[0]) as file:
            g5.copy(file, output, ["/param"])

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                # collect from snapshots
                indices = _index_snapshots(file[_module])
                pdfx += Preparation.compute_x(
                    dynamics=Preparation.get_dynamics(file),
                    sigma=file[_module]["snapshots"]["sigma"][indices, ...],
                    sigmay=file[_module]["snapshots"]["sigmay"][indices, ...],
                ).ravel()

                # collect from avalanches
                if _module == m_name:
                    indices = _index_avalanches(file[_module])
                    avalanches = file[_module]["avalanches"]
                    for i in indices:
                        ti = avalanches["t"][i, ...] - avalanches["t0"][i]
                        ai = epm.cumsum_n_unique(avalanches["idx"][i, ...]) / N
                        si = np.arange(1, ti.size + 1) / N
                        if not np.all(ti == 0):  # happens for very small temperatures
                            binned.add_sample(ti, si, si**2, ai, ai**2, np.sqrt(ai))

            # update output file
            Preparation.store_histogram(output["hist_x"], pdfx)

            if _module == m_name:
                for name in binned.names:
                    for key, value in binned[name]:
                        storage.dump_overwrite(output, f"/restore/{name}/{key}", value)

            output.flush()

        # compute relaxation time
        if _module == m_name:
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


def EnsembleStructure(cli_args: list = None, _return_parser: bool = False, _module: str = m_name):
    """
    Extract the structure factor at snapshots.
    See:
    https://doi.org/10.1103/PhysRevB.74.140201
    https://doi.org/10.1103/PhysRevLett.118.147208
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="Output file", default=f_structure
    )
    parser.add_argument("info", type=pathlib.Path, help="EnsembleInfo: read files")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.info.exists()
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.info) as file:
        root = args.info.parent
        files = [root / f for f in file["files"].asstr()[...]]

    files = [f for f in files if f.exists()]
    assert len(files) > 0

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(_module, funcname), dev=args.develop)
        output["/settings/files"] = sorted([f.name for f in files])

        for ifile, f in enumerate(tqdm.tqdm(files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    data = eye.Structure(shape=file["param"]["shape"][...])

                if f"/{_module}/snapshots/epsp" not in file:
                    continue

                snapshots = file[_module]["snapshots"]
                for i in _index_snapshots(file[_module]):
                    data += snapshots["epsp"][i, ...]

            for name, value in data:
                storage.dump_overwrite(output, f"/restore/{name}", value)

            output.flush()


def Plot(cli_args: list = None, _return_parser: bool = False, _module: str = m_name) -> None:
    """
    Basic of the ensemble.
    """
    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="Simulation file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file) as file:
        realisation = "hist_x" not in file

    if realisation:
        with h5py.File(args.file) as file:
            res = file[_module]["snapshots"]
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


class MeasureAvalanches:
    def __init__(
        self,
        n: int,
        S_bin_edges: np.array,
        A_bin_edges: np.array,
        ell_bin_edges: np.array,
        n_moments: int,
    ):
        self.line = False
        self.n_moments = n_moments

        self.S_hist = [enstat.histogram(bin_edges=S_bin_edges) for _ in range(n)]
        self.A_hist = [enstat.histogram(bin_edges=A_bin_edges) for _ in range(n)]
        self.ell_hist = [enstat.histogram(bin_edges=ell_bin_edges) for _ in range(n)]

        self.S_mean = [[enstat.scalar(dtype=int) for _ in range(n_moments)] for _ in range(n)]
        self.A_mean = [[enstat.scalar(dtype=int) for _ in range(n_moments)] for _ in range(n)]
        self.ell_mean = [[enstat.scalar(dtype=int) for _ in range(n_moments)] for _ in range(n)]

        self.A_fractal = [enstat.binned(bin_edges=A_bin_edges, names=["A", "S"]) for _ in range(n)]
        self.ell_fractal = [
            enstat.binned(bin_edges=ell_bin_edges, names=["ell", "S"]) for _ in range(n)
        ]

    def add_sample(self, index: int, S: np.array, ell: np.array, A: np.array = None):
        if A is None:
            return self.add_sample_1d(index, S, ell)

        assert np.issubdtype(S.dtype, np.integer)
        assert np.issubdtype(A.dtype, np.integer)
        assert not np.issubdtype(ell.dtype, np.integer)

        if len(S) == 0:
            return

        self.S_hist[index] += S
        self.A_hist[index] += A
        self.ell_hist[index] += ell
        self.A_fractal[index].add_sample(A, S)
        self.ell_fractal[index].add_sample(ell, S)

        # to avoid overflow: assume that "ell" is float
        S = S.astype(int).astype("object")
        A = A.astype(int).astype("object")

        for p in range(self.n_moments):
            self.S_mean[index][p] += S ** (p + 1)
            self.A_mean[index][p] += A ** (p + 1)
            self.ell_mean[index][p] += ell ** (p + 1)

    def add_sample_1d(self, index: int, S: np.array, ell: np.array):
        assert np.issubdtype(S.dtype, np.integer)
        assert np.issubdtype(ell.dtype, np.integer)
        self.line = True

        if len(S) == 0:
            return

        self.S_hist[index] += S
        self.ell_hist[index] += ell
        self.ell_fractal[index].add_sample(ell, S)

        # to avoid overflow
        S = S.astype(int).astype("object")
        ell = ell.astype(int).astype("object")

        for p in range(self.n_moments):
            self.S_mean[index][p] += S ** (p + 1)
            self.ell_mean[index][p] += ell ** (p + 1)

    def store(self, file: h5py.File, root: str = ""):
        names = ["S_hist", "ell_hist", "A_hist"]
        values = [self.S_hist, self.ell_hist, self.A_hist]

        if self.line:
            names = names[:-1]
            values = values[:-1]

        for name, value in zip(names, values):
            vdict = [dict(i0) for i0 in value]
            for field in ["bin_edges", "count"]:
                path = g5.join(root, name, field, root=True)
                storage.dump_overwrite(file, path, [i0[field] for i0 in vdict])

        names = ["S_mean", "ell_mean", "A_mean"]
        values = [self.S_mean, self.ell_mean, self.A_mean]

        if self.line:
            names = names[:-1]
            values = values[:-1]

        for name, value in zip(names, values):
            vdict = [[dict(p) for p in i0] for i0 in value]
            for field in ["first", "second", "norm"]:
                path = g5.join(root, name, field, root=True)
                storage.dump_overwrite(file, path, [[float(p[field]) for p in i0] for i0 in vdict])

        names = ["fractal_ell", "fractal_A"]
        values = [self.ell_fractal, self.A_fractal]

        if self.line:
            names = names[:-1]
            values = values[:-1]

        for name, value in zip(names, values):
            for key in ["S", name.split("_")[1]]:
                vdict = [dict(i0[key]) for i0 in value]
                for field in ["first", "second", "norm"]:
                    path = g5.join(root, name, key, field, root=True)
                    storage.dump_overwrite(file, path, [i0[field] for i0 in vdict])


class MySegmenterBasic:
    """
    Add points to the system and measure avalanches.
    The base class only measures :py:attr:`S`, the amount of times each block has failed.
    """

    def __init__(self, shape):
        self.shape = shape
        self.N = np.prod(shape)
        self.S = np.zeros(shape, dtype=int)

    def reset(self):
        """
        Reset to initial state: a completely flat interface.
        """
        self.S *= 0

    def add_points(self, idx):
        if len(idx) == 0:
            return
        self.S += np.bincount(idx, minlength=self.N).reshape(self.shape)


class MySegmenterClusters(MySegmenterBasic):
    """
    Spatially segment avalanches identifying connected clusters.
    """

    def __init__(self, shape):
        super().__init__(shape)
        self.segmenter = eye.ClusterLabeller(shape=shape, periodic=True)
        self._init_empty()

    def _init_empty(self):
        self.s = np.array([], dtype=int)
        self.a = np.array([], dtype=int)
        self.ell = np.array([], dtype=float)

    def avalanches(self):
        return dict(S=self.s, A=self.a, ell=self.ell)

    def reset(self):
        super().reset()
        self.segmenter.reset()

    def add_points(self, idx):
        super().add_points(idx)
        if np.sum(self.S) == 0:
            self._init_empty()
            return

        self.segmenter.add_points(np.copy(idx))
        self.segmenter.prune()
        labels = self.segmenter.labels.astype(int).ravel()
        keep = labels > 0
        assert np.sum(keep) > 0
        labels = labels[keep]
        self.a = np.bincount(labels)[1:]
        self.s = np.bincount(labels, weights=self.S.ravel()[keep]).astype(int)[1:]
        self.ell = Preparation.convert_A_to_ell(self.a, len(self.shape))


class MySegmenterChord(MySegmenterBasic):
    def __init__(self, shape):
        super().__init__(shape)
        self.nchord = max(int(0.1 * np.min(shape)), 1)
        self._init_empty()

    def _init_empty(self):
        self.s = np.array([], dtype=int)
        self.ell = np.array([], dtype=int)

    def avalanches(self):
        return dict(S=self.s, ell=self.ell)

    def reset(self):
        super().reset()

    def add_points(self, idx):
        super().add_points(idx)
        if np.sum(self.S) == 0:
            self._init_empty()
            return

        rows = np.random.choice(np.arange(self.shape[0]), size=self.nchord, replace=False)
        label_offset = 0
        labels = []
        sizes = []

        rows = np.random.choice(np.arange(self.shape[0]), size=self.nchord, replace=False)
        cols = np.random.choice(np.arange(self.shape[1]), size=self.nchord, replace=False)
        indices = [(row, None) for row in rows] + [(None, col) for col in cols]
        for row, col in indices:
            srow = np.copy(self.S[row, col].ravel())
            lrow = eye.clusters(srow, periodic=True)
            keep = lrow > 0
            if np.sum(keep) == 0:
                continue
            lrow = lrow[keep] + label_offset
            srow = srow[keep]
            label_offset += np.max(lrow)
            labels += list(lrow.astype(int))
            sizes += list(srow.astype(int))

        # may happen if all chords are outside events
        if len(labels) == 0:
            self._init_empty()
            return

        labels = eye.labels_prune(labels)
        self.ell = np.bincount(labels)[1:]
        self.s = np.bincount(labels, weights=sizes).astype(int)[1:]


def EnsembleAvalanches_base(
    cli_args: list, _return_parser: bool, _module: str, mymode: str, funcname, doc: str
) -> None:
    """
    Calculate properties of avalanches.

    .. warning::

        This function assumes separately stored "avalanches" sequences as independent.
    """
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing file")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output file",
        default=f"EnsembleAvalanches_{mymode}.h5",
    )
    parser.add_argument(
        "--xc", type=float, required=True, help=r"Gap in P(x) -> \tau_\alpha = \exp(x_c^\alpha / T)"
    )
    parser.add_argument(
        "--tau", type=float, nargs=3, default=[-5, 0, 51], help="logspace tau (units of tau_alpha)"
    )
    parser.add_argument(
        "--skip",
        type=float,
        default=2,
        help="Consider independent measurement after t = `skip` tau_alpha",
    )
    parser.add_argument("--bins", type=int, help="Number of bins P(S), P(A), P(ell)", default=60)
    parser.add_argument(
        "--means", type=int, default=4, help="Compute <S, A, ell>**(i + 1) for i in range(means)"
    )
    parser.add_argument("info", type=pathlib.Path, help="EnsembleInfo: read files")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert args.info.exists()
    tools._check_overwrite_file(args.output, args.force)
    assert args.tau[1] < np.log10(args.skip)  # both in units of tau_alpha

    # load basic data
    with h5py.File(args.info) as file:
        root = args.info.parent
        files = [root / f for f in file["files"].asstr()[...]]
        shape = file["param"]["shape"][...]
        alpha = file["param"]["alpha"][...]
        temperature = file["param"]["temperature"][...]
        tau_alpha = np.exp(args.xc**alpha / temperature)

    files = [f for f in files if f.exists()]
    assert len(files) > 0

    # size estimate
    max_s = 0
    navalanches = 0
    for ifile, f in enumerate(files):
        with h5py.File(f) as file:
            avalanches = file[_module]["avalanches"]
            max_s = max(max_s, avalanches["idx"].shape[1])
            navalanches += len(_index_avalanches(file[_module]))

    # time points to measure: correct for smallest times available, to avoid useless measurements
    t_measure = np.logspace(args.tau[0], args.tau[1], args.tau[2])  # units of tau_alpha

    # allocate statistics
    if mymode in ["chord", "clusters"]:
        L = np.max(shape)
        N = np.prod(shape)
        opts = dict(bins=args.bins, mode="log", integer=True)
        measurement = MeasureAvalanches(
            n=len(t_measure),
            S_bin_edges=enstat.histogram.from_data(np.array([1, max_s]), **opts).bin_edges,
            A_bin_edges=enstat.histogram.from_data(np.array([1, N]), **opts).bin_edges,
            ell_bin_edges=enstat.histogram.from_data(np.array([1, L]), **opts).bin_edges,
            n_moments=args.means,
        )
    elif mymode == "structure":
        structure = [eye.Structure(shape=shape) for _ in t_measure]
    else:
        raise ValueError(f"Unknown mode {mymode}")

    # collect statistics
    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, tools.path_meta(_module, funcname), dev=args.develop)
        output["/settings/files"] = sorted([f.name for f in files])
        output["/settings/tau"] = t_measure * tau_alpha
        output["/settings/tau"].attrs["units"] = "physical"
        output["/settings/xc"] = args.xc
        output["/settings/tau_alpha"] = tau_alpha
        output["/settings/tau_alpha"].attrs["units"] = "physical"
        with h5py.File(files[0]) as file:
            g5.copy(file, output, ["/param"])

        if mymode == "chord":
            mysegmenter = MySegmenterChord(shape)
            output["/settings/nchord"] = mysegmenter.nchord
        elif mymode == "clusters":
            mysegmenter = MySegmenterClusters(shape)
        else:
            mysegmenter = MySegmenterBasic(shape)

        # measure
        pbar = tqdm.tqdm(total=navalanches)
        for f in tqdm.tqdm(files):
            with h5py.File(f) as file:
                avalanches = file[_module]["avalanches"]
                indices = _index_avalanches(file[_module])
                for iava in indices:
                    pbar.n += 1
                    pbar.set_description(f"{f.name}({iava})")
                    pbar.refresh()

                    # extract sequence of failures
                    # (normalisation of time to avoid overflow)
                    t0 = avalanches["t0"][iava] / tau_alpha
                    t_load = avalanches["t"][iava, ...] / tau_alpha - t0
                    idx_load = avalanches["idx"][iava, ...].astype(int)

                    # throw away / ignore incomplete data
                    i = np.argmax(t_load < 0)
                    if i > 0:
                        if i < t_load.size / 2:
                            continue
                        t_load = t_load[:i]
                        idx_load = idx_load[:i]

                    # split data into independent measurements
                    t_split = []
                    idx_split = []
                    while True:
                        i = np.argmax(t_load > args.skip)
                        if i == 0:
                            t_split.append(np.copy(t_load))
                            idx_split.append(np.copy(idx_load))
                            break
                        t_split.append(np.copy(t_load[:i]))
                        idx_split.append(np.copy(idx_load[:i]))
                        t_load = t_load[i:] - t_load[i - 1]
                        idx_load = idx_load[i:]

                    # measure avalanches
                    for t, idx in zip(t_split, idx_split):
                        mysegmenter.reset()
                        for i0, t0 in enumerate(t_measure):
                            i = np.argmax(t > t0)
                            if i > 0:
                                mysegmenter.add_points(idx[:i])
                                idx = np.copy(idx[i:])
                                t = np.copy(t[i:])
                            else:
                                mysegmenter.add_points([])

                            if mymode == "structure":
                                structure[i0] += mysegmenter.S
                            else:
                                measurement.add_sample(i0, **mysegmenter.avalanches())

            if mymode == "structure":
                for name, value in zip(["structure"], [structure]):
                    value = [dict(i0) for i0 in value]
                    for key in ["first", "second", "norm"]:
                        storage.dump_overwrite(
                            output,
                            f"/data/{name}/{key}",
                            np.array([i0[key][:, 0] for i0 in value], dtype=np.float64),
                        )
            else:
                measurement.store(file=output, root="/data")

            output.flush()


@tools.docstring_append_cli()
def EnsembleAvalanches_clusters(
    cli_args: list = None, _return_parser: bool = False, _module: str = m_name
):
    """
    Calculate properties of avalanches.
    -   Measure at different times compared to an arbitrary reference time.
    -   Avalanches are segmented by spatial clustering.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    return EnsembleAvalanches_base(
        cli_args=cli_args,
        _return_parser=_return_parser,
        _module=_module,
        mymode="clusters",
        funcname=funcname,
        doc=doc,
    )


@tools.docstring_append_cli()
def EnsembleAvalanches_chord(
    cli_args: list = None, _return_parser: bool = False, _module: str = m_name
):
    """
    Calculate properties of avalanches.
    -   Measure at different times compared to an arbitrary reference time.
    -   Avalanches are measured along ranmdomly chosen lines.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    return EnsembleAvalanches_base(
        cli_args=cli_args,
        _return_parser=_return_parser,
        _module=_module,
        mymode="chord",
        funcname=funcname,
        doc=doc,
    )


@tools.docstring_append_cli()
def EnsembleAvalanches_structure(
    cli_args: list = None, _return_parser: bool = False, _module: str = m_name
):
    """
    Measure the structure factor of snapshots.
    -   Measure at different times compared to an arbitrary reference time.
    -   The interface is arbitrarily assumed flat at the reference time.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = tools._fmt_doc(inspect.getdoc(globals()[funcname]))
    return EnsembleAvalanches_base(
        cli_args=cli_args,
        _return_parser=_return_parser,
        _module=_module,
        mymode="structure",
        funcname=funcname,
        doc=doc,
    )


# <autodoc> generated by docs/conf.py


def _BranchPreparation_parser() -> argparse.ArgumentParser:
    return BranchPreparation(_return_parser=True)


def _EnsembleAvalanches_chord_parser() -> argparse.ArgumentParser:
    return EnsembleAvalanches_chord(_return_parser=True)


def _EnsembleAvalanches_clusters_parser() -> argparse.ArgumentParser:
    return EnsembleAvalanches_clusters(_return_parser=True)


def _EnsembleAvalanches_structure_parser() -> argparse.ArgumentParser:
    return EnsembleAvalanches_structure(_return_parser=True)


def _EnsembleInfo_parser() -> argparse.ArgumentParser:
    return EnsembleInfo(_return_parser=True)


def _EnsembleStructure_parser() -> argparse.ArgumentParser:
    return EnsembleStructure(_return_parser=True)


def _Plot_parser() -> argparse.ArgumentParser:
    return Plot(_return_parser=True)


def _Run_parser() -> argparse.ArgumentParser:
    return Run(_return_parser=True)


def _UpgradeData_parser() -> argparse.ArgumentParser:
    return UpgradeData(_return_parser=True)


# </autodoc>
