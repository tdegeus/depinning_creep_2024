import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

import h5py
import numpy as np
import shelephant
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.parent.absolute()
if (root / "mycode_thermal_epm" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_thermal_epm import AQS  # noqa: E402
from mycode_thermal_epm import Extremal  # noqa: E402
from mycode_thermal_epm import Thermal  # noqa: E402
from mycode_thermal_epm import Preparation  # noqa: E402


class MyThermal(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        myfile = pathlib.Path(__file__)
        path = myfile.parent / myfile.name.replace(".py", "").replace("test_", "output_")
        if path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.workdir = path
        self.origin = pathlib.Path().absolute()
        os.chdir(path)

        for path in (root / "tests" / "data_version=2.0").rglob("*h5"):
            shutil.copy2(path, ".")

    @classmethod
    def tearDownClass(self):
        os.chdir(self.origin)

    def test_upgrade(self):
        Preparation.UpgradeData(["--dev", "Preparation.h5"])
        AQS.UpgradeData(["--dev", "AQS.h5"])
        Extremal.UpgradeData(["--dev", "Extremal.h5", "--insert", "ExtremalAvalanche.h5"])
        Thermal.UpgradeData(["--dev", "Thermal.h5"])

        # check new run
        with shelephant.path.tempdir():
            AQS.BranchPreparation(["--dev", self.workdir / "Preparation.h5", "sim.h5"])
            AQS.Run(["--dev", "sim.h5", "-n", 200])
            with h5py.File(self.workdir / "AQS.h5") as old, h5py.File("sim.h5") as new:
                a = old["AQS"]["data"]
                b = new["AQS"]["data"]
                assert np.allclose(a["T"][...], b["T"][...])
                assert np.allclose(a["S"][...], b["S"][...])

                a = old["AQS"]["snapshots"]
                b = new["AQS"]["snapshots"]
                assert np.allclose(a["step"][...], b["step"][1:-1])
                assert np.allclose(a["uframe"][...], b["uframe"][1:-1])
                for ai, step in enumerate(a["step"][...]):
                    bi = np.argwhere(b["step"][...] == step).ravel()
                    assert np.allclose(a["epsp"][ai, ...], b["epsp"][bi, ...])
                    assert np.allclose(a["state"][ai], b["state"][bi])

        # check new run
        with shelephant.path.tempdir():
            Extremal.BranchPreparation(
                [
                    "--dev",
                    self.workdir / "Preparation.h5",
                    "sim.h5",
                    "--interval-preparation",
                    6 * 100 * 10 * 10,
                    "--interval-avalanche",
                    100 * 10 * 10,
                ]
            )
            Extremal.Run(["--dev", "sim.h5", "-n", 2])
            with h5py.File(self.workdir / "Extremal.h5") as old, h5py.File("sim.h5") as new:
                a = old["Extremal"]["avalanches"]
                b = new["Extremal"]["avalanches"]
                assert np.allclose(a["xmin"][...], b["xmin"][...])
                assert np.allclose(a["idx"][...], b["idx"][...])

                a = old["Extremal"]["snapshots"]
                b = new["Extremal"]["snapshots"]
                assert np.allclose(a["epsp"][-1, ...], b["epsp"][-1, ...])
                assert np.allclose(a["state"][-1], b["state"][-1])
                assert np.allclose(a["t"][-1], b["t"][-1])

        # check new run
        with shelephant.path.tempdir():
            Thermal.BranchPreparation(
                ["--dev", self.workdir / "Preparation.h5", "sim.h5", "--temperature", 0.1]
            )
            Thermal.Run(["--dev", "sim.h5", "-n", 12])
            with h5py.File(self.workdir / "Thermal.h5") as old, h5py.File("sim.h5") as new:
                a = old["Thermal"]["avalanches"]
                b = new["Thermal"]["avalanches"]
                assert np.allclose(a["t"][...], b["t"][...] - b["t0"][...].reshape(-1, 1))
                assert np.allclose(a["idx"][...], b["idx"][...])
                a = old["Thermal"]["snapshots"]
                b = new["Thermal"]["snapshots"]
                index = np.argwhere(b["index_avalanche"][...] < 0).ravel()
                assert np.allclose(a["epsp"][:-1, ...], b["epsp"][index, ...])
                assert np.allclose(a["state"][:-1], b["state"][index])


if __name__ == "__main__":
    unittest.main(verbosity=2)
