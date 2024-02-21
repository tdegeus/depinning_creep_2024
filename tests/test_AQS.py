import os
import pathlib
import tempfile
import time

import h5py
import numpy as np
import pytest
import shelephant

from depinning_creep_2024 import AQS
from depinning_creep_2024 import Preparation


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
    AQS.BranchPreparation(["--dev", "id=0000.h5", "sim.h5"])
    AQS.Run(["--dev", "-n", 200, "sim.h5"])
    AQS.EnsembleInfo(["--dev", "sim.h5"])


def test_restart(mydata):
    """
    Test that the simulation can be correctly restarted if it is interrupted.
    """
    with shelephant.path.tempdir():
        AQS.BranchPreparation(["--dev", mydata["workdir"] / "id=0000.h5", "sim.h5"])
        AQS.BranchPreparation(["--dev", mydata["workdir"] / "id=0000.h5", "res.h5"])

        # run simulation without interruption, log the time that it took
        tic = time.time()
        AQS.Run(["--dev", "-n", 200, "sim.h5"])
        dt = time.time() - tic

        # run identical simulation from the start, but interrupt it several times
        while True:
            AQS.Run(["--dev", "-n", 200, "res.h5", "--walltime", 0.2 * dt, "--buffer", 0.1 * dt])
            with h5py.File("res.h5") as file:
                if file["AQS"]["data"]["S"].size >= 200:
                    break

        # check that both runs are identical
        with h5py.File("sim.h5") as a, h5py.File("res.h5") as b:
            aa = a["AQS"]["data"]
            bb = b["AQS"]["data"]
            for key in ["uframe", "sigma", "S", "A", "T"]:
                assert np.allclose(aa[key][...], bb[key][...])

            aa = a["AQS"]["snapshots"]
            bb = b["AQS"]["snapshots"]
            asel = np.argwhere(aa["systemspanning"]).ravel()
            bsel = np.argwhere(bb["systemspanning"]).ravel()
            for key in ["epsp", "sigma", "sigmay", "uframe", "state", "step"]:
                assert np.allclose(aa[key][asel, ...], bb[key][bsel, ...])
