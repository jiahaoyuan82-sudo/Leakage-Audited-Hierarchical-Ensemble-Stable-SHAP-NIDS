"""Metrics that respect the saved, explicitly named class-probability order."""

import numpy as np


def named_log_loss(y_true, probabilities, class_order):
    """Mean negative log probability of the true class, with float64 clipping.

    `class_order` must list columns in the same order as `probabilities`.
    This avoids silently pairing an alphabetically sorted label list with a
    different probability-column order.
    """
    classes = list(class_order)
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true)
    if p.ndim != 2 or p.shape != (len(y), len(classes)):
        raise ValueError("Probability shape does not match rows and class order")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    lookup = {name: index for index, name in enumerate(classes)}
    try:
        indices = np.fromiter((lookup[label] for label in y), dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown true class: {exc.args[0]}") from exc
    sums = p.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError("At least one probability row has zero mass")
    true_probability = p[np.arange(len(y)), indices] / sums
    return float(-np.log(np.maximum(true_probability, np.finfo(np.float64).eps)).mean())
