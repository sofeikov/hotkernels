"""Shared set-up of the tests: the lazy layer's tests run twice, in Python alone and with its C core.

The core (`prototypes/lazy/_core.c`) is a quick road for what the Python
does, so every test of the layer holds for it too. Each test of a
`test_lazy_*` module is run once with the core put away and once with it in
use; the second is skipped where the core cannot be built.
"""
from __future__ import annotations

import importlib
import subprocess
from typing import Any, Iterator

import pytest

# The two ways a test of the lazy layer is run.
PYTHON, CORE = "python", "c"
# The modules whose tests are run both ways, by the start of their names.
LAZY_TESTS = "test_lazy"
# The name the built core is imported by.
CORE_MODULE = "prototypes.lazy._core"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run each test of the lazy layer once per way of writing nodes.

    Args:
        metafunc: The test function being collected.
    """
    if metafunc.module.__name__.startswith(LAZY_TESTS):
        metafunc.parametrize("lazy_core", [PYTHON, CORE], indirect=True)


@pytest.fixture(scope="session")
def built_core() -> Any:
    """Build the C core and import it, once for the whole run.

    Returns:
        The module, or the reason it could not be built, as a string.
    """
    from prototypes.lazy import build_core
    try:
        build_core.build()
    except build_core.NoCompiler as error:
        return str(error)
    except subprocess.CalledProcessError as error:
        return f"the core does not compile here: {error.stderr}"
    return importlib.import_module(CORE_MODULE)


@pytest.fixture(autouse=True)
def lazy_core(request: pytest.FixtureRequest) -> Iterator[str | None]:
    """Put the layer on the road a test asks for, and back afterwards.

    Args:
        request: The test's request; its parameter is PYTHON or CORE for a
            test of the lazy layer, and absent for any other test.

    Returns:
        The way nodes are written during the test, or None for a test that is not the layer's.
    """
    way = getattr(request, "param", None)
    if way is None:
        yield None
        return
    from prototypes.lazy import array, graph
    # The layer logs each line of code once per process, and this process runs each test's lines twice.
    graph._logged.clear()
    before = array._core
    if way == CORE:
        core = request.getfixturevalue("built_core")
        if isinstance(core, str):
            pytest.skip(core)
        array.use_core(core)
        assert array.core_in_use()
    else:
        array.use_core(None)
    try:
        yield way
    finally:
        array.use_core(before)
