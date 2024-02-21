from mycode_thermal_epm import Preparation


def test_basic(tmp_path):
    """
    Run workflow, not a unit test.
    """
    outdir = tmp_path / "Preparation"
    Preparation.Generate(
        ["--dev", "-n", 1, "--shape", 10, 10, "--dynamics", "default", "--all", str(outdir)]
    )
    assert (outdir / "id=0000.h5").exists()
