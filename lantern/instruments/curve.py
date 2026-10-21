"""A key-rate zero curve: linear interpolation between factor nodes, flat outside.

Yields are treated as continuously compounded zero rates (the Yahoo indices are
par yields; the difference is second order for risk). Because the curve is linear in
the node yields, a key-rate shock moves the zero rate at time t by W(t) @ dy, where
W(t) are the interpolation weights: this is what makes key-rate durations exact.
"""
from __future__ import annotations

import numpy as np


class Curve:
    def __init__(self, tenors, yields, ids):
        self.tenors = np.asarray(tenors, float)
        self.yields = np.asarray(yields, float)
        self.ids = list(ids)

    def weights(self, t) -> np.ndarray:
        """(len(t), n_nodes) linear-interpolation weights."""
        t = np.clip(np.atleast_1d(np.asarray(t, float)), self.tenors[0], self.tenors[-1])
        W = np.zeros((t.size, self.tenors.size))
        j = np.clip(np.searchsorted(self.tenors, t, side="right") - 1, 0, self.tenors.size - 2)
        lo, hi = self.tenors[j], self.tenors[j + 1]
        a = (t - lo) / (hi - lo)
        W[np.arange(t.size), j] = 1 - a
        W[np.arange(t.size), j + 1] = a
        return W

    def zero(self, t, dy=None) -> np.ndarray:
        y = self.yields if dy is None else self.yields + dy
        return self.weights(t) @ np.asarray(y).T

    def discount(self, t) -> np.ndarray:
        t = np.atleast_1d(t)
        return np.exp(-self.zero(t) * t)
