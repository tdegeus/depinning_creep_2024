import os
import pathlib
import sys
import textwrap
import tomllib
from collections import defaultdict

sys.path.insert(0, os.path.abspath(".."))

project = "depinning_creep_2024"
copyright = "2024, Tom de Geus"
author = "Tom de Geus"
html_theme = "furo"
autodoc_type_aliases = {"Iterable": "Iterable", "ArrayLike": "ArrayLike"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinxarg.ext",
]
templates_path = ["_templates"]

# autogenerate modules.rst

modules = list((pathlib.Path(__file__).parent / ".." / project).glob("*.py"))
modules = [m.stem for m in modules]
for name in ["__init__", "_version"]:
    if name in modules:
        modules.remove(name)
modules = sorted(modules)

ret = [".. This file is generated by conf.py"]
header = "Python module"
ret += [
    "\n".join(
        [
            "*" * len(header),
            header,
            "*" * len(header),
        ]
    )
]

ret += [
    "",
    f".. currentmodule:: {project}",
    "",
    ".. autosummary::",
    "    :toctree: generated",
    "",
]

for module in modules:
    ret.append(f"    {module}")

(pathlib.Path(__file__).parent / "module.rst").write_text("\n".join(ret) + "\n")

# autogenerate cli.rst

data = tomllib.loads((pathlib.Path(__file__).parent / ".." / "pyproject.toml").read_text())
scripts = data["project"]["scripts"]
generated = []
(pathlib.Path(__file__).parent / "cli").mkdir(exist_ok=True)

functions = defaultdict(list)

for name, funcname in scripts.items():
    modname, funcname = funcname.split(":")
    libname, modname = modname.split(".")
    parser = f"_{funcname}_parser"
    progname = f"{modname}_{funcname}"
    functions[modname].append(funcname)

    generated.append(progname)
    (pathlib.Path(__file__).parent / "cli" / f"{progname}.rst").write_text(
        "\n".join(
            [
                f".. _{modname}_{funcname}:",
                "",
                progname,
                "-" * len(progname),
                "",
                ".. argparse::",
                f"    :module: {libname}.{modname}",
                f"    :func: {parser}",
                f"    :prog: {progname}",
            ]
        )
    )

ret = [".. This file is generated by conf.py"]
header = "Command-line tools"
ret += [
    "\n".join(
        [
            "*" * len(header),
            header,
            "*" * len(header),
        ]
    )
]

ret += [
    "",
    ".. toctree::",
    "    :maxdepth: 1",
    "",
]
ret += [textwrap.indent(i, " " * 4) for i in generated]

(pathlib.Path(__file__).parent / "cli" / "index.rst").write_text("\n".join(ret) + "\n")


#  autogenerate parser code

changes = False
root = pathlib.Path(__file__).parent / ".." / project
fmt = """
def _{funcname:s}_parser() -> argparse.ArgumentParser:
    return {funcname:s}(_return_parser=True)
"""

for modname in functions:
    add = []
    for funcname in functions[modname]:
        add.append(fmt.format(funcname=funcname))

    text = (root / f"{modname}.py").read_text()
    if "# <autodoc>" not in text:
        ret = (
            text.rstrip()
            + "\n\n\n# <autodoc> generated by docs/conf.py\n\n"
            + "\n".join(add)
            + "\n\n# </autodoc>\n"
        )
    else:
        start, end = text.split("# <autodoc>")
        end = end.split("# </autodoc>")[1]
        ret = (
            start.rstrip()
            + "\n\n\n# <autodoc> generated by docs/conf.py\n\n"
            + "\n".join(add)
            + "\n\n# </autodoc>\n"
            + end
        )

    ret = ret.strip() + "\n"
    if ret != text:
        (root / f"{modname}.py").write_text(ret)
        changes = True

if changes:
    raise ValueError("Changes made to source files.")
