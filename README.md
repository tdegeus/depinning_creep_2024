[![ci](https://github.com/tdegeus/depinning_creep_2024/workflows/CI/badge.svg)](https://github.com/tdegeus/depinning_creep_2024/actions)
[![Documentation Status](https://readthedocs.org/projects/depinning-creep-2024/badge/?version=latest)](https://depinning-creep-2024.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://github.com/tdegeus/depinning_creep_2024/workflows/pre-commit/badge.svg)](https://github.com/tdegeus/depinning_creep_2024/actions)

**Documentation: [depinning-creep-2024.readthedocs.io](https://depinning-creep-2024.readthedocs.io)**

# depinning_creep_2024

Interface between data and [GooseEPM](https://github.com/tdegeus/GooseEPM).

Getting started:

1.  Install dependencies.
    For example using Conda:

    ```bash
    mamba env update --file environment.yaml
    ```

    Tip: use a new environment.
    Note: some of these dependences are needed, some are merely a convenience for running the simulations.

2.  Install this package:

    ```bash
    python -m pip install . -v --no-build-isolation --no-deps
    ```

3.  Build the docs:

    ```bash
    cd docs
    make html
    ```

    Open `docs/_build/html/index.html` in a web browser.

**Warning:** This is a research code. The API and data structure may be subjected to changes.
