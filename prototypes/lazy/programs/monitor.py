"""A regression that looks at its loss as it goes: a program that makes the layer compute many times.

It is here for the recording mode. `regression.fit` never needs a value, so
it leaves one graph. This one gathers its loss every few steps, and each
gather is a graph of its own, alike from one to the next; and it ends with
`np.nanargmax`, a call outside the deferred set, which is made at once and so
cuts the graph a second way.
"""
from __future__ import annotations

from typing import Any

# Step size of the descent.
LEARNING_RATE = 0.05
# The loss is gathered once in this many steps.
REPORT_EVERY = 5


def fit(np: Any, features: Any, targets: Any, steps: int) -> tuple[Any, list[float], int]:
    """Fit a linear regression by gradient descent, reporting the loss as it goes.

    Args:
        np: The lazy stand-in for the `numpy` module. The function gathers, so it does not run with the real one.
        features: The examples, one row each.
        targets: One target per example.
        steps: How many gradient steps to take.

    Returns:
        (the coefficients as an `ndarray`, the loss at every REPORT_EVERY-th
        step, the index of the example with the largest residual at the last
        step).
    """
    X, y = np.asarray(features), np.asarray(targets)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    w = np.zeros(X.shape[1])
    history = []
    for step in range(steps):
        residual = X @ w - y
        loss = (residual ** 2).mean()
        w = w - LEARNING_RATE * (2 / len(y)) * (X.T @ residual)
        if step % REPORT_EVERY == 0:
            history.append(float(loss.gather()))
    worst = np.nanargmax(np.abs(residual))
    return w.gather(), history, int(worst.gather())
