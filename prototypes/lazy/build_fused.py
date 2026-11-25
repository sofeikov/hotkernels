"""Builds the lazy layer's fused kernels, and the module that calls them, into an extension beside this file.

    uv run python -m prototypes.lazy.build_fused

Every directory of `prototypes/lazy/kernels/` is one fused kernel, laid out
as the kernels of `kernels/` are: `spec.py`, `naive.c`, and `kernel.s` under
the architecture's name where one was written. This command assembles the
`kernel.s` of the machine's architecture, or compiles `naive.c` where there
is none, generates the table `_fused.c` reads them from, and links the lot
into `_fused`. The layer runs without the module, computing every node with
NumPy's own call, and `fuse.enable()` takes it when it is there.

It needs clang and NumPy's headers.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import numpy
import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from judge.build import ARCH, C_FLAGS
from judge.spec import In, Kernel, Out, Scalar, Size, load_spec

log = logging.getLogger(__name__)

# The compiler. The same one `judge/build.py` calls.
CC = "clang"
# The directory of the layer, where the source is and the module goes.
LAYER = Path(__file__).resolve().parent
# The fused kernels, one directory each.
KERNELS = LAYER / "kernels"
# The source of the module that calls them, and the name it is imported by.
SOURCE, MODULE = LAYER / "_fused.c", "_fused"
# Where the generated table and the kernels' objects go. Not committed: the name is in `.gitignore`.
BUILD = LAYER / "build"
# The table `_fused.c` includes, and the template it is written from.
TABLE, TABLE_TEMPLATE = "_fused_kernels.h", "fused_kernels.h.j2"
# Where the layer keeps its templates.
TEMPLATES = LAYER / "templates"
# Flags for the module: optimised, position-independent, a shared library, with the warnings on.
FLAGS = ("-O2", "-fPIC", "-shared", "-Wall")
# On macOS an extension leaves Python's own symbols to be found when it is loaded.
MACOS_FLAGS = ("-undefined", "dynamic_lookup")
# The headers a `naive.c` counts on, which the judge supplies with the prototype it generates.
NAIVE_HEADERS = ("stddef.h", "stdint.h", "stdbool.h")
# The element type every argument of a fused kernel has.
ELEMENT_TYPE = "f64"
# What a reduction returns and what any other fused kernel returns, as a spec names them.
RETURNS_NUMBER, RETURNS_NOTHING = "f64", "void"

app = typer.Typer(add_completion=False)


class NoCompiler(RuntimeError):
    """Raised when the compiler is not installed."""


class BadSpec(ValueError):
    """Raised for a kernel whose arguments are not in the order a fused kernel's must be."""


@dataclass(frozen=True)
class Fused:
    """One fused kernel as the build and the layer need it."""

    # The kernel's name: its directory's, and its C function's.
    name: str
    # Its directory.
    directory: Path
    # Its spec.
    spec: Kernel
    # How many arrays it reads and how many numbers it takes.
    vectors: int
    scalars: int
    # True for a kernel that returns one number, False for one that fills an output array.
    reduces: bool

    @property
    def source(self) -> Path:
        """The file that is built: the assembly for this machine's architecture, or the C where there is none."""
        assembly = self.directory / ARCH / "kernel.s"
        return assembly if assembly.exists() else self.directory / "naive.c"


def read(directory: Path) -> Fused:
    """Read one kernel's spec and check its arguments are in a fused kernel's order.

    `_fused.c` calls every kernel one way: its input arrays first, then the
    output array unless it returns a number, then the length, then its
    numbers. All of them are float64.

    Args:
        directory: The kernel's directory.

    Returns:
        The kernel.

    Raises:
        BadSpec: When the arguments are in another order or of another type.
    """
    spec = load_spec(directory)
    kinds = [type(arg) for arg in spec.args]
    vectors, scalars = kinds.count(In), kinds.count(Scalar)
    reduces = spec.returns == RETURNS_NUMBER
    expected = [In] * vectors + ([] if reduces else [Out]) + [Size] + [Scalar] * scalars
    typed = all(arg.dtype == ELEMENT_TYPE for arg in spec.args if not isinstance(arg, Size))
    if kinds != expected or not typed or spec.returns not in (RETURNS_NUMBER, RETURNS_NOTHING) or vectors < 1:
        raise BadSpec(f"{spec.name}: a fused kernel takes float64 arrays, then its output, its length and its numbers")
    return Fused(spec.name, directory, spec, vectors, scalars, reduces)


def kernels() -> list[Fused]:
    """Return the fused kernels, in the order of their names: the order of the table."""
    return [read(spec.parent) for spec in sorted(KERNELS.glob("*/spec.py"))]


def target() -> Path:
    """Return where the built module goes: beside the source, with this Python's suffix for extensions."""
    return LAYER / f"{MODULE}{sysconfig.get_config_var('EXT_SUFFIX')}"


def _is_fresh(built: Path, sources: list[Path]) -> bool:
    """Say whether a built file is newer than everything it is built from.

    Args:
        built: The built file.
        sources: What it is built from.

    Returns:
        False when it does not exist or a source is newer.
    """
    return built.exists() and all(built.stat().st_mtime >= source.stat().st_mtime for source in sources)


def build(force: bool = False) -> Path:
    """Build the module if it is not built, or is older than a source.

    Args:
        force: Whether to build even when the module is newer than its sources.

    Returns:
        The path of the built module.

    Raises:
        NoCompiler: When clang is not on the path.
        BadSpec: For a kernel `_fused.c` could not call.
        subprocess.CalledProcessError: When the compiler refuses a source; its output is in the error.
    """
    fused = kernels()
    built = target()
    sources = [SOURCE, Path(__file__), TEMPLATES / TABLE_TEMPLATE, *(kernel.source for kernel in fused),
               *(kernel.directory / "spec.py" for kernel in fused)]
    if not force and _is_fresh(built, sources):
        return built
    if shutil.which(CC) is None:
        raise NoCompiler(f"{CC} is not installed")
    BUILD.mkdir(exist_ok=True)
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True)
    (BUILD / TABLE).write_text(environment.get_template(TABLE_TEMPLATE).render(kernels=fused))
    objects = []
    for kernel in fused:
        compiled = BUILD / f"{kernel.name}.o"
        flags = [] if kernel.source.suffix == ".s" else [*C_FLAGS, *(part for header in NAIVE_HEADERS for part in ("-include", header))]
        subprocess.run([CC, "-c", *flags, str(kernel.source), "-o", str(compiled)], check=True, capture_output=True, text=True)
        objects.append(str(compiled))
    flags = [*FLAGS, *(MACOS_FLAGS if platform.system() == "Darwin" else ())]
    includes = [f"-I{sysconfig.get_paths()['include']}", f"-I{numpy.get_include()}", f"-I{BUILD}"]
    subprocess.run([CC, *flags, *includes, str(SOURCE), *objects, "-o", str(built)], check=True, capture_output=True, text=True)
    return built


@app.command()
def main(force: bool = typer.Option(False, help="Build even when the module is newer than its sources.")) -> None:
    """Build the fused kernels and the module that calls them, and say where it is.

    Args:
        force: Whether to build even when the module is newer than its sources.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        log.info("built %s", build(force))
    except subprocess.CalledProcessError as error:
        log.error("%s", error.stderr)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
