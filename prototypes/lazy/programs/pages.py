"""Writes the page of recorded graphs for the example programs.

    uv run python -m prototypes.lazy.programs.pages --out lazy-graphs.html

The three programs run in one recorded block: the regression, which leaves
one graph of a loop; the transformer block, whose graph has the calls of its
functions; and the regression that reports its loss, which leaves a graph for
each report.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy
import typer

from prototypes.lazy import gather, get_lazy, record_graphs
from prototypes.lazy.programs import attention, monitor, regression

log = logging.getLogger(__name__)

# Seed of the random data, so the page is the same from run to run.
SEED = 0
# Examples and features of the two regressions.
EXAMPLES, FEATURES = 20_000, 30
# Tokens, width and heads of the transformer block. Its feed-forward layer is WIDENING times as wide.
TOKENS, WIDTH, HEADS, WIDENING = 64, 96, 4, 4

app = typer.Typer(add_completion=False)


def block_parameters(rng: numpy.random.Generator) -> dict[str, numpy.ndarray]:
    """Draw float32 weights for one transformer block.

    Args:
        rng: The generator to draw from.

    Returns:
        Parameter name -> array, in the shapes `attention.block` expects.
    """
    shapes = {"gain_1": (WIDTH,), "bias_1": (WIDTH,), "gain_2": (WIDTH,), "bias_2": (WIDTH,),
              "w_qkv": (WIDTH, 3 * WIDTH), "b_qkv": (3 * WIDTH,), "w_out": (WIDTH, WIDTH), "b_out": (WIDTH,),
              "w_up": (WIDTH, WIDENING * WIDTH), "b_up": (WIDENING * WIDTH,),
              "w_down": (WIDENING * WIDTH, WIDTH), "b_down": (WIDTH,)}
    return {name: rng.standard_normal(shape).astype(numpy.float32) for name, shape in shapes.items()}


@app.command()
def main(out: Path = typer.Option(Path("lazy-graphs.html"), help="Where to write the page."),
         steps: int = typer.Option(50, help="Gradient steps of the two regressions."),
         show: bool = typer.Option(False, help="Open the page in the browser when it is written.")) -> None:
    """Run the example programs under a recording and write the page of their graphs.

    Args:
        out: Where to write the page.
        steps: Gradient steps of the two regressions.
        show: Whether to open the page in the browser.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = numpy.random.default_rng(SEED)
    features = rng.standard_normal((EXAMPLES, FEATURES))
    targets = features @ rng.standard_normal(FEATURES) + 0.1 * rng.standard_normal(EXAMPLES)
    tokens = rng.standard_normal((TOKENS, WIDTH)).astype(numpy.float32)
    parameters = block_parameters(rng)
    lazy = get_lazy()
    with record_graphs() as recording:
        weights, intercept, losses = regression.fit(lazy, features, targets, steps)
        gather(weights, intercept, *losses)
        attention.block(lazy.asarray(tokens), parameters, HEADS).gather()
        monitor.fit(lazy, features, targets, steps)
    written = recording.save(out)
    log.info("wrote %s: %d graphs", written, len(recording.graphs))
    if show:
        recording.show()


if __name__ == "__main__":
    app()
