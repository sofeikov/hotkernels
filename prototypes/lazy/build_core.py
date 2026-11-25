"""Builds the layer's C core, `_core.c`, into an extension module beside it.

    uv run python -m prototypes.lazy.build_core

The layer runs without the core, and takes it when it is there: `array.py`
imports `_core` if it can. The core is the quick road for writing one node
of the graph; see the comment at the top of `_core.c`. It needs clang, the
compiler the judge uses.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path

import typer

log = logging.getLogger(__name__)

# The compiler. The same one `judge/build.py` calls.
CC = "clang"
# The directory of the layer, where the source is and the module goes.
LAYER = Path(__file__).resolve().parent
# The source, and the name the module is imported by.
SOURCE, MODULE = LAYER / "_core.c", "_core"
# Flags for every platform: optimised, position-independent, a shared library, with the warnings on.
FLAGS = ("-O2", "-fPIC", "-shared", "-Wall")
# On macOS an extension leaves Python's own symbols to be found when it is loaded.
MACOS_FLAGS = ("-undefined", "dynamic_lookup")

app = typer.Typer(add_completion=False)


class NoCompiler(RuntimeError):
    """Raised when the compiler is not installed."""


def target() -> Path:
    """Return where the built module goes: beside the source, with this Python's suffix for extensions."""
    return LAYER / f"{MODULE}{sysconfig.get_config_var('EXT_SUFFIX')}"


def build(force: bool = False) -> Path:
    """Compile the core if it is not built, or is older than its source.

    Args:
        force: Whether to compile even when the module is newer than the source.

    Returns:
        The path of the built module.

    Raises:
        NoCompiler: When clang is not on the path.
        subprocess.CalledProcessError: When the compiler refuses the source; its output is in the error.
    """
    built = target()
    if not force and built.exists() and built.stat().st_mtime >= SOURCE.stat().st_mtime:
        return built
    if shutil.which(CC) is None:
        raise NoCompiler(f"{CC} is not installed")
    flags = [*FLAGS, *(MACOS_FLAGS if platform.system() == "Darwin" else ())]
    command = [CC, *flags, f"-I{sysconfig.get_paths()['include']}", str(SOURCE), "-o", str(built)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return built


@app.command()
def main(force: bool = typer.Option(False, help="Compile even when the module is newer than its source.")) -> None:
    """Build the layer's C core and say where it is.

    Args:
        force: Whether to compile even when the module is newer than its source.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        log.info("built %s", build(force))
    except subprocess.CalledProcessError as error:
        log.error("%s", error.stderr)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
