import os
import pathlib

import shelephant

from depinning_creep_2024 import AQS
from depinning_creep_2024 import Extremal
from depinning_creep_2024 import Preparation
from depinning_creep_2024 import Thermal

assert Thermal.data_version == "3.0"
path = pathlib.Path(__file__).parent / "tests" / f"data_version={Thermal.data_version}"
path.mkdir(parents=True, exist_ok=True)

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

    AQS.BranchPreparation(["--dev", "Preparation.h5", "AQS.h5"])
    AQS.Run(["--dev", "-n", 200, "AQS.h5"])
    AQS.EnsembleInfo(["--dev", "-o", "AQS_info.h5", "AQS.h5"])

    Thermal.BranchPreparation(["--dev", "Preparation.h5", "Thermal.h5", "--temperature", 0.1])
    Thermal.Run(["--dev", "-n", 6, "Thermal.h5"])
    Thermal.EnsembleInfo(["--dev", "-o", "Thermal_info.h5", "Thermal.h5"])
    Thermal.EnsembleStructure(["--dev", "-o", "Thermal_structure.h5", "Thermal_info.h5"])
    Thermal.EnsembleAvalanches_clusters(
        ["--dev", "-o", "Thermal_avalanches.h5", "--xc", 0.3, "Thermal_info.h5"]
    )

    Extremal.BranchPreparation(["--dev", "Preparation.h5", "Extremal.h5"])
    Extremal.Run(["--dev", "-n", 6, "Extremal.h5"])
    Extremal.EnsembleInfo(["--dev", "-o", "Extremal_info.h5", "Extremal.h5"])
    Extremal.EnsembleStructure(["--dev", "-o", "Extremal_structure.h5", "Extremal_info.h5"])

    for res in [
        "Preparation.h5",
        "AQS.h5",
        "AQS_info.h5",
        "Thermal.h5",
        "Thermal_info.h5",
        "Thermal_structure.h5",
        "Thermal_avalanches.h5",
        "Extremal.h5",
        "Extremal_info.h5",
        "Extremal_structure.h5",
    ]:
        os.rename(res, path / res)
