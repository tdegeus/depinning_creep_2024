import itertools
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

        Preparation.Generate(["--dev", "-n", 1, "--shape", 10, 10, "--dynamics", "default", "."])

    @classmethod
    def tearDownClass(self):
        os.chdir(self.origin)

    def test_basic(self):
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
        Thermal.EnsembleStructure(["--dev", "sim.h5"])
        Thermal.EnsembleAvalanches(["--dev", "info.h5"])

        with h5py.File("sim.h5") as file:
            assert np.allclose(
                file["Thermal"]["snapshots"]["t"][::2], file["Thermal"]["avalanches"]["t0"][...]
            )

        with h5py.File("info.h5") as file:
            assert "t" in file["restore"]
            assert "S" in file["restore"]
            assert "A" in file["restore"]
            assert "ell" in file["restore"]

    def test_restart(self):
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
                Thermal.BranchPreparation(["--dev", self.workdir / "id=0000.h5", "sim.h5"] + args)
                Thermal.BranchPreparation(["--dev", self.workdir / "id=0000.h5", "res.h5"] + args)
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
                        self.assertTrue(np.allclose(a[key][...], b[key][...]))

                    if "avalanches" in a["Thermal"]:
                        for variable in ["idx", "t", "t0"]:
                            key = f"/Thermal/avalanches/{variable}"
                            self.assertTrue(np.allclose(a[key][...], b[key][...]))

                    index = a["Thermal"]["snapshots"]["index_avalanche"][...]
                    index_snapshot = np.arange(index.size)[index > 0] - 1
                    index_avalanche = index[index > 0]
                    index_avalanche = index_avalanche[index_snapshot >= 0]
                    index_snapshot = index_snapshot[index_snapshot >= 0]
                    if "avalanches" in a["Thermal"]:
                        self.assertTrue(
                            np.allclose(
                                a["Thermal"]["avalanches"]["t0"][index_avalanche],
                                a["Thermal"]["snapshots"]["t"][index_snapshot],
                            )
                        )

                os.remove("sim.h5")
                os.remove("res.h5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
