import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.parent.absolute()
if (root / "mycode_thermal_epm" / "_version.py").exists():
    sys.path.insert(0, str(root))

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

    @classmethod
    def tearDownClass(self):
        os.chdir(self.origin)

    def test_basic(self):
        Preparation.Generate(
            ["--dev", "-n", 1, "--shape", 10, 10, "--dynamics", "default", "Preparation", "--all"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
