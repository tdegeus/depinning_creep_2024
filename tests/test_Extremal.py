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

from mycode_thermal_epm import Extremal  # noqa: E402
from mycode_thermal_epm import Preparation  # noqa: E402


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
        Preparation.Generate(
            ["--dev", "-n", 1, "--interactions", "monotonic-shortrange", "--shape", 10, 10, "."]
        )
        Preparation.Run(["--dev", "id=0000.h5"])
        Extremal.BranchPreparation(["--dev", "id=0000.h5", "id=0000_sim.h5", "--sigmay", 1.0, 0.3])
        Extremal.BranchPreparation(["--dev", "id=0000.h5", "id=0000_res.h5", "--sigmay", 1.0, 0.3])
        Extremal.Run(["--dev", "-n", 6, "id=0000_sim.h5"])
        for _ in range(3):
            Extremal.Run(["--dev", "-n", 2, "id=0000_res.h5"])

        with h5py.File("id=0000_sim.h5") as a, h5py.File("id=0000_res.h5") as b:
            key = "/Extremal/sigma"
            self.assertTrue(np.allclose(a[key][...], b[key][...]))

        Extremal.EnsembleInfo(["--dev", "id=0000_sim.h5"])
        Extremal.EnsembleHeightHeight(["--dev", "id=0000_sim.h5"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
