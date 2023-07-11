import argparse
import inspect
import logging
import pathlib
import textwrap

import enstat
import GooseEPM as epm
import GooseEYE as eye
import GooseHDF5 as g5
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
f_height = "EnsembleHeightHeight.h5"
m_name = "Thermal"
m_avalanche = "Avalanche"
m_exclude = ["AQS", "Extremal", "ExtremalAvalanche"]
m_dummy_int = int(12e12)


class SystemThermalStressControl(epm.SystemThermalStressControl):
    def __init__(self, file: h5py.File):
        param = file["param"]
        restart = file["restart"]

        epm.SystemThermalStressControl.__init__(
            self,
            *Preparation.propagator(param),
            sigmay_mean=np.ones(param["shape"][...]) * param["sigmay"][0],
            sigmay_std=np.ones(param["shape"][...]) * param["sigmay"][1],
            seed=restart["state"].attrs["seed"],
            alpha=param["alpha"][...],
            random_stress=False,
        )

        self.epsp = restart["epsp"][...]
        self.sigma = restart["sigma"][...]
        self.sigmay = restart["sigmay"][...]
        self.t = restart["t"][...]
        self.state = restart["state"][...]
        self.sigmabar = param["sigmabar"][...]
        self.temperature = param["temperature"][...]


def _upgrade_data_v1_to_v2(src: h5py.File, dst: h5py.File):
    """
    -   Convert ``A`` to ``idx``, add ``idx_ignore`` for those data.
    """
    g5.copy(src, dst, ["/param", "/restart"])
    Preparation._copy_metadata_pre_v2(src, dst)

    if m_name not in src:
        return

    A = src[f"/{m_name}/A"][...]
    if "idx" in src[m_name]:
        idx = src[f"/{m_name}/idx"][...]
        n = A.shape[0] - idx.shape[0]
    else:
        idx = None
        n = 0
    cp = g5.getdatapaths(src, root=f"/{m_name}")
    cp.remove(f"/{m_name}/A")

    if n == 0:
        g5.copy(src, dst, cp)
        root = dst[f"/{m_name}"]
        assert root["t"].maxshape[0] is None
        if idx is None:
            g5.copy(src, dst, f"/{m_name}/A", f"/{m_name}/idx")
            n = A.shape[0]
            dset = root.create_dataset("idx_ignore", (n,), maxshape=(None,), dtype=np.uint64)
            dset[:] = np.arange(n, dtype=np.uint64)
        return

    cp.remove(f"/{m_name}/idx")
    g5.copy(src, dst, cp)
    root = dst[f"/{m_name}"]
    assert root["t"].maxshape[0] is None

    dset = root.create_dataset("idx", A.shape, maxshape=(None, A.shape[1]), dtype=idx.dtype)
    dset[:n, :] = A[:n, :]
    dset[n:, :] = idx
    if A.shape[1] > 0:
        assert np.all(np.equal(dset[0, :], epm.cumsum_n_unique(A[0, :])))
    if idx.shape[1] > 0:
        assert np.all(np.equal(dset[-1, :], idx[-1, :]))

    dset = root.create_dataset("idx_ignore", (n,), maxshape=(None,), dtype=np.uint64)
    dset[:] = np.arange(n, dtype=np.uint64)


def _upgrade_data_to_v1(src: h5py.File, dst: h5py.File):
    paths = g5.getdatapaths(src)
    paths.remove(f"/{m_name}/S")
    g5.copy(src, dst, paths)

    S = src[f"/{m_name}/S"][...]
    expect = np.arange(S.shape[1]).reshape((1, -1))
    assert np.all(np.equal(S, expect)), "S must arange"


def _write_completed(src: h5py.File, myname: str = m_name, error: bool = True):
    """
    Check basic integrity of data.

    :return: (bool, int) with:
        bool: True if data is complete
        int: number of snapshots
    """
    if myname not in src:
        return 0

    if np.any(np.diff(src[f"/{myname}/t"][...]) < 0):
        raise ValueError(f"{src.filename} has unrecoverable corrupted data (t).")

    if np.any(np.diff(src[f"/{myname}/epsp"][:, 0, 0]) < 0):
        raise ValueError(f"{src.filename} has unrecoverable corrupted data (epsp).")

    paths = g5.getdatapaths(src, root=f"/{myname}")
    if f"/{myname}/mean_epsp" in paths:
        paths.remove(f"/{myname}/mean_epsp")
    if f"/{myname}/idx_ignore" in paths:
        paths.remove(f"/{myname}/idx_ignore")

    n = np.unique([src[path].shape[0] for path in paths])
    if n.size > 1:
        msg = f"{src.filename} has inconsistent data length. Run UpdateData."
        if error:
            raise ValueError(msg)
        else:
            logging.warning(msg)

    return np.min(n)


def _cleanup(src: h5py.File, dst: h5py.File):
    """
    -   Correct corrupted storage
    -   Remove unused data
    """

    ret = None
    paths = g5.getdatapaths(src, root=f"/{m_name}")

    if f"/{m_name}/mean_epsp" in paths:
        paths.remove(f"/{m_name}/mean_epsp")
        ret = dst.filename

    n = np.sort(np.unique([src[path].shape[0] for path in paths]))
    if n.size == 1 and ret is None:
        return ret
    elif n.size == 1:
        g5.copy(src, dst, paths + ["/meta", "/param", "/restart"])
    else:
        n = n[0]
        ret = dst.filename
        g5.copy(src, dst, ["/meta", "/param"])
        for path in paths:
            shape = list(src[path].shape)
            shape[0] = n
            data = dst.create_dataset(
                path, shape, maxshape=[None] + shape[1:], dtype=src[path].dtype
            )
            data[...] = src[path][:n, ...]
        for key in src["restart"]:
            if src[f"/restart/{key}"].ndim == 0:
                dst[f"/restart/{key}"] = src[f"/restart/{key}"][...]
            else:
                dst[f"/restart/{key}"] = src[f"/{m_name}/{key}"][-1, ...]

    assert np.all(np.equal(np.argsort(dst[m_name]["t"][...]), np.arange(n)))
    return ret


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

    if tag.equal(ver, "2.0"):
        temp_file = temp_dir / "cleanup.h5"
        with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
            return _cleanup(src, dst)

    if tag.less(ver, "1.0"):
        temp_file = temp_dir / "from_1_0.h5"
        with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
            _upgrade_data_to_v1(src, dst)
        filename = temp_file
        ver = "1.0"

    if tag.less(ver, "2.0"):
        temp_file = temp_dir / "from_2_0.h5"
        with h5py.File(filename) as src, h5py.File(temp_file, "w") as dst:
            _upgrade_data_v1_to_v2(src, dst)
        filename = temp_file
        ver = "2.0"

    with h5py.File(temp_file, "a") as dst:
        storage.dump_overwrite(dst, "/param/data_version", data_version)

    return temp_file


def UpgradeData(cli_args=None):
    r"""
    Upgrade data to the current version.
    """
    Preparation.UpgradeData(cli_args, _upgrade_data)


def BranchPreparation(cli_args=None):
    r"""
    Branch from prepared stress state and add parameters.

    1.  Copy ``\param``.
        Add ``\param\sigmabar``, ``\param\temperature``, ``\param\sigmay``.

    2.  Copy ``\init`` to ``\restart``.
        Add ``\restart\epsp``, ``\restart\t``.
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
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument(
        "--sigmay", type=float, nargs=2, default=[1.0, 0.3], help="Mean and std of sigmay"
    )
    parser.add_argument("input", type=pathlib.Path, help="Input file (read-only)")
    parser.add_argument("output", type=pathlib.Path, help="Output file (overwritten)")

    args = tools._parse(parser, cli_args)
    assert args.input.exists()
    assert not args.output.exists()

    with h5py.File(args.input) as src, h5py.File(args.output, "w") as dest:
        assert not any(m in src for m in m_exclude), "Wrong file type"
        g5.copy(src, dest, ["/meta", "/param"])
        g5.copy(src, dest, "/init", "/restart")
        dest["restart"]["epsp"] = np.zeros(src["param"]["shape"][...], dtype=np.float64)
        dest["restart"]["t"] = 0.0
        dest["param"]["sigmay"] = args.sigmay
        dest["param"]["sigmabar"] = args.sigmabar
        dest["param"]["temperature"] = args.temperature
        tools.create_check_meta(dest, f"/meta/{m_name}/{funcname}", dev=args.develop)


def Run(cli_args=None):
    """
    Run simulation at fixed stress.
    Measure the state every ``--interval`` events.
    ``--interval`` is the number of times that all blocks have to have failed between measurements.
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
    parser.add_argument("--interval", type=int, default=100, help="Measure every #events")
    parser.add_argument("-n", "--measurements", type=int, default=100, help="Total #measurements")
    parser.add_argument("--ninc", type=int, help="#increments to measure (default: ``20 N``)")
    parser.add_argument("file", type=pathlib.Path, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert args.file.exists()

    with h5py.File(args.file, "r") as file:
        _write_completed(file)

    with h5py.File(args.file, "a") as file:
        assert Preparation.get_data_version(file) == data_version
        tools.create_check_meta(file, f"/meta/{m_name}/{funcname}", dev=args.develop)
        system = SystemThermalStressControl(file)
        global_restart = file["restart"]
        if args.ninc is None:
            args.ninc = 20 * system.size
        else:
            assert args.ninc > 0

        if m_name not in file:
            assert not any(m in file for m in m_exclude), "Wrong file type"
            res = file.create_group(m_name)
        else:
            res = file[m_name]

        for _ in tqdm.tqdm(range(args.measurements), desc=str(args.file)):
            nfails = system.nfails.copy()
            system.makeThermalFailureSteps(args.interval * system.size)
            while True:
                if np.all(system.nfails - nfails >= args.interval):
                    break
                system.makeThermalFailureSteps(system.size)

            with g5.ExtendableSlice(res, "epsp", system.shape, np.float64) as dset:
                dset += system.epsp
            with g5.ExtendableSlice(res, "sigma", system.shape, np.float64) as dset:
                dset += system.sigma
            with g5.ExtendableSlice(res, "sigmay", system.shape, np.float64) as dset:
                dset += system.sigmay
            with g5.ExtendableList(res, "state", np.uint64) as dset:
                dset.append(system.state)
            with g5.ExtendableList(res, "t", np.float64) as dset:
                dset.append(system.t)

            if args.ninc > 0:
                t0 = float(system.t)
                measurement = epm.Avalanche()
                measurement.makeThermalFailureSteps(system, args.ninc)
                with g5.ExtendableSlice(res, "idx", [args.ninc], np.uint64) as dset:
                    dset += measurement.idx
                with g5.ExtendableSlice(res, "T", [args.ninc], np.float64) as dset:
                    dset += measurement.t - t0

            global_restart["epsp"][...] = system.epsp
            global_restart["sigma"][...] = system.sigma
            global_restart["sigmay"][...] = system.sigmay
            global_restart["t"][...] = system.t
            global_restart["state"][...] = system.state
            file.flush()


def EnsembleInfo(cli_args=None):
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

    tmin = np.inf
    tmax = 0
    nsim = 0
    for f in tqdm.tqdm(args.files):
        with h5py.File(f) as file:
            assert Preparation.get_data_version(file) == data_version
            assert not any(m in file for m in m_exclude), "Wrong file type"
            if f"/{m_name}/T" not in file:
                continue
            nsim += 1
            res = file[m_name]
            for i in range(res["T"].shape[0]):
                tmin = min(tmin, res["T"][i, 0])
                tmax = max(tmax, res["T"][i, -1])

    if nsim == 0:
        return

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, f"/meta/{m_name}/{funcname}", dev=args.develop)
        output["files"] = sorted([f.name for f in args.files])

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    g5.copy(file, output, ["/param"])
                    alpha = file["param"]["alpha"][...]
                    dE = enstat.scalar()
                    pdfx = enstat.histogram(
                        bin_edges=np.linspace(0, 3, args.nbin_x), bound_error="ignore"
                    )
                    binned = enstat.binned(
                        bin_edges=enstat.histogram.from_data(
                            data=np.array([max(tmin, 1e-9), tmax]), bins=args.nbin_t, mode="log"
                        ).bin_edges,
                        names=["t", "S", "Ssq", "A", "Asq", "ell"],  # N.B.: ell^2 == A
                        bound_error="ignore",
                    )
                    N = np.prod(file["param"]["shape"][...])

                if f"/{m_name}/T" not in file:
                    continue

                res = file[m_name]
                x = (res["sigmay"][...] - np.abs(res["sigma"][...])).ravel()
                sorter = np.argsort(x)
                dE += x[sorter[1]] ** alpha - x[sorter[0]] ** alpha
                pdfx += x
                n = _write_completed(file, error=False)
                for i in range(n):
                    ti = res["T"][i, ...]
                    ai = epm.cumsum_n_unique(res["idx"][i, ...]) / N
                    si = np.arange(1, ti.size + 1) / N
                    if np.all(ti == 0):  # happens for very small temperatures
                        continue
                    binned.add_sample(ti, si, si**2, ai, ai**2, np.sqrt(ai))

            storage.dump_overwrite(output, "/dE/first", dE.first)
            storage.dump_overwrite(output, "/dE/second", dE.second)
            storage.dump_overwrite(output, "/dE/norm", dE.norm)
            storage.dump_overwrite(output, "/hist_x/bin_edges", pdfx.bin_edges)
            storage.dump_overwrite(output, "/hist_x/count", pdfx.count)
            storage.dump_overwrite(
                output, "chi4_S", N * (binned["Ssq"].mean() - binned["S"].mean() ** 2)
            )
            storage.dump_overwrite(
                output, "chi4_A", N * (binned["Asq"].mean() - binned["A"].mean() ** 2)
            )
            storage.dump_overwrite(
                output, "chi4_ell", N * (binned["A"].mean() - binned["ell"].mean() ** 2)
            )
            storage.dump_overwrite(output, "t", binned["t"].mean())

            for name in binned.names:
                for key, value in binned[name]:
                    storage.dump_overwrite(output, f"/restore/{name}/{key}", value)

            output.flush()


def EnsembleHeightHeight(cli_args=None, myname: str = m_name):
    """
    Get the height-height correlation function.
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
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output file", default=f_height)
    parser.add_argument("files", nargs="*", type=pathlib.Path, help="Simulation files")

    args = tools._parse(parser, cli_args)
    assert all([f.exists() for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        tools.create_check_meta(output, f"/meta/{m_name}/{funcname}", dev=args.develop)
        output["files"] = sorted([f.name for f in args.files])

        for ifile, f in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(f) as file:
                if ifile == 0:
                    w = int(file["param"]["shape"][0] // 2)
                    h = int(file["param"]["shape"][1] // 2)
                    data = {
                        key: {
                            "horizontal": eye.Ensemble([2 * w + 1], variance=True, periodic=True),
                            "vertical": eye.Ensemble([2 * h + 1], variance=True, periodic=True),
                        }
                        for key in ["eps", "epsp", "epse"]
                    }

                if f"/{myname}/epsp" not in file:
                    continue

                res = file[myname]
                n = _write_completed(file, myname, error=False)
                for i in range(n):
                    entry = {"epsp": res["epsp"][i, ...], "epse": res["sigma"][i, ...]}
                    entry["eps"] = entry["epsp"] + entry["epse"]
                    for key in data:
                        e = entry[key]
                        data[key]["horizontal"].heightheight(e[0, :])
                        data[key]["vertical"].heightheight(e[:, 0])

            for key in data:
                for d in data[key]:
                    datum = data[key][d]
                    r = datum.distance(0).astype(int)
                    keep = r >= 0
                    storage.dump_overwrite(output, f"/{key}/{d}/mean", datum.result()[keep])
                    storage.dump_overwrite(output, f"/{key}/{d}/variance", datum.variance()[keep])
                    storage.dump_overwrite(output, f"/{key}/{d}/distance", r[keep])
            output.flush()


def Plot(cli_args=None):
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
        x = file["hist_x"]["x"][...]
        p = file["hist_x"]["p"][...]
        xc = file["param"]["sigmay"][0] - file["param"]["sigmabar"][...]

        chi4_S = file["chi4_S"][...]
        chi4_A = file["chi4_A"][...]
        t = file["t"][...]

    fig, axes = gplt.subplots(ncols=3)

    ax = axes[0]
    ax.plot(x, p)
    ax.axvline(xc, color="r", ls="-", label=r"$\langle \sigma_y \rangle - \bar{\sigma}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    ax.legend()

    ax = axes[1]
    ax.plot(t, chi4_S, label=r"$\chi_4^S$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\chi_4^S$")

    ax = axes[2]
    ax.plot(t, chi4_A, label=r"$\chi_4^A$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\chi_4^A$")

    plt.show()
    plt.close(fig)
