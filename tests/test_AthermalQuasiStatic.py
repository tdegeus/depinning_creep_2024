import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from functools import partialmethod

import h5py
import numpy as np
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.parent.absolute()
if (root / "mycode_thermal_epm" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_thermal_epm import AthermalQuasiStatic  # noqa: E402
from mycode_thermal_epm import AthermalPreparation  # noqa: E402


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.origin = pathlib.Path().absolute()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    @classmethod
    def tearDownClass(self):
        os.chdir(self.origin)
        shutil.rmtree(self.tempdir)

    def test_basic(self):
        AthermalPreparation.Generate(
            ["--dev", "-n", 1, "--interactions", "monotonic-shortrange", "--shape", 10, 10, "."]
        )
        AthermalPreparation.Run(["--dev", "id=0000.h5"])
        AthermalQuasiStatic.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_qs.h5", "--sigmay", 1.0, 0.3]
        )
        AthermalQuasiStatic.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_res.h5", "--sigmay", 1.0, 0.3]
        )
        AthermalQuasiStatic.Run(["--dev", "-n", 60, "id=0000_qs.h5"])
        for _ in range(6):
            AthermalQuasiStatic.Run(["--dev", "-n", 10, "id=0000_res.h5"])

        with h5py.File("id=0000_qs.h5") as a, h5py.File("id=0000_res.h5") as b:
            aa = a["AthermalQuasiStatic"]
            bb = b["AthermalQuasiStatic"]
            for key in ["uframe", "sigma", "S", "A", "T"]:
                self.assertTrue(np.allclose(aa[key][...], bb[key][...]))

            aa = a["AthermalQuasiStatic/restore"]
            bb = b["AthermalQuasiStatic/restore"]
            for key in ["epsp", "sigma", "sigmay", "uframe", "state", "step"]:
                self.assertTrue(np.allclose(aa[key][...], bb[key][...]))

        AthermalQuasiStatic.EnsembleInfo(["id=0000_qs.h5"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
