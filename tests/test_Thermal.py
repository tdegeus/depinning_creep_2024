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

from mycode_thermal_epm import Thermal  # noqa: E402
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
        Thermal.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_qs.h5", "--sigmay", 1.0, 0.1, "--temperature", 0.1]
        )
        Thermal.BranchPreparation(
            ["--dev", "id=0000.h5", "id=0000_res.h5", "--sigmay", 1.0, 0.1, "--temperature", 0.1]
        )
        Thermal.Run(["--dev", "-n", 6, "id=0000_qs.h5"])
        for _ in range(3):
            Thermal.Run(["--dev", "-n", 2, "id=0000_res.h5"])

        with h5py.File("id=0000_qs.h5") as a, h5py.File("id=0000_res.h5") as b:
            self.assertTrue(np.allclose(a["/Thermal/x/0"][...], b["/Thermal/x/0"][...]))
            self.assertTrue(np.allclose(a["/Thermal/x/1"][...], b["/Thermal/x/1"][...]))
            self.assertTrue(np.allclose(a["/Thermal/x/2"][...], b["/Thermal/x/2"][...]))
            self.assertTrue(np.allclose(a["/Thermal/x/3"][...], b["/Thermal/x/3"][...]))
            self.assertTrue(np.allclose(a["/Thermal/x/4"][...], b["/Thermal/x/4"][...]))
            self.assertTrue(np.allclose(a["/Thermal/x/5"][...], b["/Thermal/x/5"][...]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
