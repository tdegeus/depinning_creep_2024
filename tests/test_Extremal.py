import itertools
import os
import pathlib
import tempfile

import pytest

from depinning_creep_2024 import Extremal
from depinning_creep_2024 import Preparation


@pytest.fixture(scope="module")
def mydata():
    """
    *   Generate a temporary directory and change working directory to it.
    *   Generate a prepared realization.
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
        "--interval-preparation": 10 * 10 * 30,
        "--interval-snapshot": 10 * 10 * 20,
        "--interval-avalanche": 10 * 10 * 20,
    }
    args = list(itertools.chain(*[(key, value) for key, value in options.items()]))
    Extremal.BranchPreparation(["--dev", "id=0000.h5", "sim.h5"] + args)
    Extremal.Run(["--dev", "-n", 200, "sim.h5"])
    Extremal.EnsembleInfo(["--dev", "-o", "info.h5", "sim.h5"])
    Extremal.EnsembleStructure(["--dev", "info.h5"])
    Extremal.EnsembleAvalanches_x0(["--dev", "info.h5"])
    Extremal.EnsembleAvalanches_x0(["--dev", "info.h5", "--xc", 0.7337, "-o", "xc.h5"])
