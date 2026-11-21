"""A transformer block's forward pass, written the way hand-written NumPy transformers write it.

Every function calls the module-level `np`, which is the real NumPy: the
block becomes lazy when its input is, because NumPy passes each call that
meets a lazy array to it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Added to the variance before the square root, against division by zero.
EPSILON = 1e-5
# Added to the scores of positions a token may not look at.
MASKED = -1e10
# The two constants of the tanh approximation of GELU.
GELU_SCALE, GELU_CUBIC = np.sqrt(2 / np.pi), 0.044715


def softmax(x: Any) -> Any:
    """Turn scores into probabilities along the last axis.

    Args:
        x: Scores.

    Returns:
        Probabilities of the same shape, each row summing to one.
    """
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layer_norm(x: Any, gain: Any, bias: Any) -> Any:
    """Normalise each row to zero mean and unit variance, then scale and shift.

    Args:
        x: Activations, one token per row.
        gain: Scale per feature.
        bias: Shift per feature.

    Returns:
        The normalised activations.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(variance + EPSILON) + bias


def gelu(x: Any) -> Any:
    """Apply the tanh approximation of GELU, element by element.

    Args:
        x: Activations.

    Returns:
        The activations after the non-linearity.
    """
    return 0.5 * x * (1 + np.tanh(GELU_SCALE * (x + GELU_CUBIC * x ** 3)))


def attention(q: Any, k: Any, v: Any, mask: Any) -> Any:
    """Scaled dot-product attention.

    Args:
        q: Queries, (heads, tokens, width).
        k: Keys, the same shape.
        v: Values, the same shape.
        mask: Added to the scores: zero where a token may look, `MASKED` where not.

    Returns:
        The attended values, (heads, tokens, width).
    """
    return softmax(q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1]) + mask) @ v


def block(x: Any, params: dict, heads: int) -> Any:
    """Run one pre-norm transformer block: causal self-attention, then a feed-forward layer.

    Args:
        x: Activations, (tokens, width).
        params: The block's weights, by name.
        heads: Attention heads; divides the width.

    Returns:
        Activations of the same shape.
    """
    tokens, width = x.shape
    h = layer_norm(x, params["gain_1"], params["bias_1"])
    qkv = h @ params["w_qkv"] + params["b_qkv"]
    q, k, v = (qkv[:, part * width:(part + 1) * width].reshape(tokens, heads, -1).swapaxes(0, 1) for part in range(3))
    mask = (1 - np.tri(tokens, dtype=params["w_qkv"].dtype)) * MASKED
    attended = attention(q, k, v, mask).swapaxes(0, 1).reshape(tokens, width)
    x = x + attended @ params["w_out"] + params["b_out"]
    h = layer_norm(x, params["gain_2"], params["bias_2"])
    return x + gelu(h @ params["w_up"] + params["b_up"]) @ params["w_down"] + params["b_down"]
