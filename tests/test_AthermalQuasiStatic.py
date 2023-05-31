import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from functools import partialmethod

from tqdm import tqdm
import h5py
import numpy as np

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
        self.orgin = pathlib.Path().absolute()

        self.origin = pathlib.Path().absolute()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        os.chdir(self.origin)
        shutil.rmtree(self.tempdir)

    def test_basic(self):
        AthermalPreparation.Generate(
            ["--dev", "-n", 1, "--interactions", "monotonic-shortrange", "--shape", 10, 10, "."]
        )
        AthermalPreparation.Run(["--dev", "id=0000.h5"])
        AthermalQuasiStatic.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_qs.h5", "--sigmay", 1.0, 0.1]
        )
        AthermalQuasiStatic.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_res.h5", "--sigmay", 1.0, 0.1]
        )
        AthermalQuasiStatic.Run(["--dev", "-n", 50, "id=0000_qs.h5"])
        for _ in range(5):
            AthermalQuasiStatic.Run(["--dev", "-n", 10, "id=0000_res.h5"])

        with h5py.File("id=0000_qs.h5") as a, h5py.File("id=0000_res.h5") as b:
            aa = a["AthermalQuasiStatic"]
            bb = b["AthermalQuasiStatic"]
            self.assertTrue(np.allclose(aa["uframe"][...], bb["uframe"][...]))
            self.assertTrue(np.allclose(aa["sigma"][...], bb["sigma"][...]))
            self.assertTrue(np.allclose(aa["S"][...], bb["S"][...]))
            self.assertTrue(np.allclose(aa["A"][...], bb["A"][...]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
