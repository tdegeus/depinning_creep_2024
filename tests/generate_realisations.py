import os
import pathlib
import sys
from functools import partialmethod

import shelephant
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.parent.absolute()
if (root / "mycode_thermal_epm" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_thermal_epm import Thermal  # noqa: E402
from mycode_thermal_epm import Preparation  # noqa: E402
from mycode_thermal_epm import Extremal  # noqa: E402
from mycode_thermal_epm import ExtremalAvalanche  # noqa: E402


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent / f"data_version={Thermal.data_version}"
    root.mkdir(parents=True, exist_ok=True)

    with shelephant.path.tempdir():
        Preparation.Generate(
            [
                "--dev",
                "-n",
                1,
                "--interactions",
                "monotonic-shortrange",
                "--dynamics",
                "default",
                "--shape",
                10,
                10,
                ".",
            ]
        )
        os.rename("id=0000.h5", "Preparation.h5")
        Preparation.Run(["--dev", "Preparation.h5"])

        Thermal.BranchPreparation(
            ["--dev", "Preparation.h5", "Thermal.h5", "--sigmay", 1.0, 0.3, "--temperature", 0.1]
        )
        Thermal.Run(["--dev", "-n", 6, "--force-interval", "Thermal.h5"])
        Thermal.EnsembleInfo(["--dev", "-o", "Thermal_info.h5", "Thermal.h5"])
        Thermal.EnsembleStructure(["--dev", "-o", "Thermal_structure.h5", "Thermal.h5"])
        Thermal.EnsembleDynamicAvalanches(
            ["--dev", "-o", "Thermal_avalanches.h5", "Thermal_info.h5"]
        )

        Extremal.BranchPreparation(["--dev", "Preparation.h5", "Extremal.h5", "--sigmay", 1.0, 0.3])
        Extremal.Run(["--dev", "-n", 6, "--force-interval", "Extremal.h5"])
        Extremal.EnsembleInfo(["--dev", "-o", "Extremal_info.h5", "Extremal.h5"])
        Extremal.EnsembleStructure(["--dev", "-o", "Extremal_structure.h5", "Extremal.h5"])

        ExtremalAvalanche.BranchExtremal(["--dev", "Extremal.h5", "ExtremalAvalanche.h5"])
        ExtremalAvalanche.Run(["--dev", "ExtremalAvalanche.h5"])
        ExtremalAvalanche.EnsembleInfo(
            ["--dev", "-o", "ExtremalAvalanche_info.h5", "ExtremalAvalanche.h5", "--xc", 0.4]
        )

        for res in [
            "Preparation.h5",
            "Thermal.h5",
            "Thermal_info.h5",
            "Thermal_structure.h5",
            "Thermal_avalanches.h5",
            "Extremal.h5",
            "Extremal_info.h5",
            "Extremal_structure.h5",
        ]:
            os.rename(res, root / res)
