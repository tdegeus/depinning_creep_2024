import os
import pathlib
import shutil
import sys
import time
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
from mycode_thermal_epm import Preparation  # noqa: E402


class MyTests(unittest.TestCase):
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
        AQS.BranchPreparation(["--dev", "id=0000.h5", "sim.h5"])
        AQS.Run(["--dev", "-n", 200, "sim.h5"])
        AQS.EnsembleInfo(["--dev", "sim.h5"])

    def test_restart(self):
        with shelephant.path.tempdir():
            AQS.BranchPreparation(["--dev", self.workdir / "id=0000.h5", "sim.h5"])
            AQS.BranchPreparation(["--dev", self.workdir / "id=0000.h5", "res.h5"])
            tic = time.time()
            AQS.Run(["--dev", "-n", 200, "sim.h5"])
            dt = time.time() - tic

            while True:
                AQS.Run(
                    ["--dev", "-n", 200, "res.h5", "--walltime", 0.2 * dt, "--buffer", 0.1 * dt]
                )
                with h5py.File("res.h5") as file:
                    if file["AQS"]["data"]["S"].size >= 200:
                        break

            with h5py.File("sim.h5") as a, h5py.File("res.h5") as b:
                aa = a["AQS"]["data"]
                bb = b["AQS"]["data"]
                for key in ["uframe", "sigma", "S", "A", "T"]:
                    self.assertTrue(np.allclose(aa[key][...], bb[key][...]))

                aa = a["AQS"]["snapshots"]
                bb = b["AQS"]["snapshots"]
                asel = np.argwhere(aa["systemspanning"]).ravel()
                bsel = np.argwhere(bb["systemspanning"]).ravel()
                for key in ["epsp", "sigma", "sigmay", "uframe", "state", "step"]:
                    self.assertTrue(np.allclose(aa[key][asel, ...], bb[key][bsel, ...]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
