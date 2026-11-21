"""A linear regression by gradient descent, written as a function of the module it computes with."""
from __future__ import annotations

from typing import Any

# How far each step moves along the gradient.
LEARNING_RATE = 0.05
# Weight of the L2 penalty on the coefficients.
PENALTY = 1e-3


def fit(np: Any, features: Any, targets: Any, steps: int) -> tuple[Any, Any, list]:
    """Fit a linear model by batch gradient descent on standardised features.

    Args:
        np: The `numpy` module, or the lazy stand-in for it.
        features: One example per row.
        targets: One number per example.
        steps: Gradient steps to take.

    Returns:
        (coefficients, intercept, the loss before each step). With the lazy
        stand-in nothing in the function needs a value, so all of them come
        back pending.
    """
    X, y = np.asarray(features), np.asarray(targets)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    count, width = X.shape
    w, b = np.zeros(width), 0.0
    losses = []
    for _ in range(steps):
        residual = X @ w + b - y
        losses.append((residual ** 2).mean() + PENALTY * (w ** 2).sum())
        gradient = 2.0 / count * (X.T @ residual) + 2.0 * PENALTY * w
        w = w - LEARNING_RATE * gradient
        b = b - LEARNING_RATE * 2.0 * residual.mean()
    return w, b, losses
