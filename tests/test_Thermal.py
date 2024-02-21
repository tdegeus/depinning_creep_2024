import itertools
import os
import pathlib
import tempfile

import h5py
import numpy as np
import pytest
import shelephant

from depinning_creep_2024 import Preparation
from depinning_creep_2024 import Thermal


@pytest.fixture(scope="module")
def mydata():
    """
    *   Generate a temporary directory and change working directory to it.
    *   Generate a prepared realization in "Preparation".
    *   If all tests are finished: change working directory back and remove temporary directory.
    """
    origin = pathlib.Path().absolute()
    tmpDir = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmpDir.name)
    os.chdir(tmp_dir)

    Preparation.Generate(["--dev", "-n", 1, "--shape", 10, 10, "--dynamics", "default", "."])
    assert pathlib.Path("id=0000.h5").exists()

    yield {"workdir": tmp_dir}

    os.chdir(origin)
    tmpDir.cleanup()


def test_basic(mydata):
    """
    Run workflow, not a unit test.
    """
    options = {
        "--temperature": 0.1,
        "--interval-preparation": 10 * 10 * 30,
        "--interval-snapshot": 10 * 10 * 20,
        "--interval-avalanche": 10 * 10 * 20,
    }
    args = list(itertools.chain(*[(key, value) for key, value in options.items()]))
    Thermal.BranchPreparation(["--dev", "id=0000.h5", "sim.h5"] + args)
    Thermal.Run(["--dev", "-n", 200, "sim.h5"])
    Thermal.EnsembleInfo(["--dev", "-o", "info.h5", "sim.h5"])
    Thermal.EnsembleStructure(["--dev", "info.h5"])
    Thermal.EnsembleAvalanches_clusters(["--dev", "--xc", 0.3, "info.h5"])
    Thermal.EnsembleAvalanches_chord(["--dev", "--xc", 0.3, "info.h5"])
    Thermal.EnsembleAvalanches_structure(["--dev", "--xc", 0.3, "info.h5"])

    with h5py.File("sim.h5") as file:
        assert np.allclose(
            file["Thermal"]["snapshots"]["t"][::2], file["Thermal"]["avalanches"]["t0"][...]
        )

    with h5py.File("info.h5") as file:
        assert "t" in file["restore"]
        assert "S" in file["restore"]
        assert "A" in file["restore"]
        assert "ell" in file["restore"]


def test_restart(mydata):
    """
    Test that the simulation can be correctly restarted if it is interrupted.
    """
    with shelephant.path.tempdir():
        configs = [
            [3, 2, 2],
            [3, 2, 1],
            [3, 2, 0],
            [3, 1, 2],
            [3, 0, 2],
            [0, 1, 2],
            [0, 0, 2],
            [0, 2, 0],
        ]
        for p, s, a in configs:
            options = {
                "--temperature": 0.1,
                "--interval-preparation": 10 * 10 * p,
                "--interval-snapshot": 10 * 10 * s,
                "--interval-avalanche": 10 * 10 * a,
            }
            args = list(itertools.chain(*[(key, value) for key, value in options.items()]))
            Thermal.BranchPreparation(["--dev", mydata["workdir"] / "id=0000.h5", "sim.h5"] + args)
            Thermal.BranchPreparation(["--dev", mydata["workdir"] / "id=0000.h5", "res.h5"] + args)
            Thermal.Run(["--dev", "-n", 200, "sim.h5"])

            while True:
                Thermal.Run(["--dev", "--walltime", 0.05, "-n", 200, "res.h5"])
                with h5py.File("res.h5") as file:
                    interval = file["Thermal"]["snapshots"].attrs["interval"]
                    dset = file["Thermal"]["snapshots"]["S"]
                    if dset.size >= 200 and dset[-2] >= interval:
                        break

            with h5py.File("sim.h5") as a, h5py.File("res.h5") as b:
                for variable in ["sigma", "sigmay", "epsp", "state", "t"]:
                    key = f"/Thermal/snapshots/{variable}"
                    assert np.allclose(a[key][...], b[key][...])

                if "avalanches" in a["Thermal"]:
                    for variable in ["idx", "t", "t0"]:
                        key = f"/Thermal/avalanches/{variable}"
                        assert np.allclose(a[key][...], b[key][...])

                index = a["Thermal"]["snapshots"]["index_avalanche"][...]
                index_snapshot = np.arange(index.size)[index > 0] - 1
                index_avalanche = index[index > 0]
                index_avalanche = index_avalanche[index_snapshot >= 0]
                index_snapshot = index_snapshot[index_snapshot >= 0]
                if "avalanches" in a["Thermal"]:
                    assert np.allclose(
                        a["Thermal"]["avalanches"]["t0"][index_avalanche],
                        a["Thermal"]["snapshots"]["t"][index_snapshot],
                    )

            os.remove("sim.h5")
            os.remove("res.h5")
