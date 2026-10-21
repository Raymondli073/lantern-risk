"""Common (systemic) explosiveness: test the first principal component of a panel.

Ex-post   : PC1 from the full-sample SVD of the standardized panel, factor = X w1 / N.
Real-time : PC1 tracked online with Candid Covariance-free Incremental PCA (CCIPCA,
            Weng, Zhang & Hwang 2003); each observation is projected on the loading
            available when it arrives, so the factor series has no look-ahead.
The factor is then fed to the BSADF recursion (see explosive.py). This mirrors the
IPCA + recursive-least-squares design of the thesis' common-bubble chapter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explosive import current_state


def _standardize(X: np.ndarray, warmup: int):
    """Scale by the first `warmup` observations only (real-time feasible)."""
    mu = np.nanmean(X[:warmup], 0)
    sd = np.nanstd(X[:warmup], 0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def pc1_expost(panel: pd.DataFrame) -> pd.Series:
    X = panel.dropna().values
    X = (X - X.mean(0)) / X.std(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    w = vt[0] * np.sign(vt[0].sum())
    return pd.Series(X @ w / X.shape[1], index=panel.dropna().index, name="PC1 (ex-post)")


def pc1_realtime(panel: pd.DataFrame, warmup: int = 36, amnesic: float = 0.0) -> pd.Series:
    P = panel.dropna()
    X = _standardize(P.values, warmup)
    N = X.shape[1]
    v = np.ones(N) / np.sqrt(N)
    out = np.empty(len(X))
    for t in range(len(X)):
        x = X[t]
        n = t + 1
        if n > 1:
            l = min(amnesic, n - 2)
            v = (n - 1 - l) / n * v + (1 + l) / n * x * (x @ v) / np.linalg.norm(v)
        u = v / np.linalg.norm(v)
        u = u * np.sign(u.sum()) if u.sum() != 0 else u
        out[t] = x @ u / N
    return pd.Series(out, index=P.index, name="PC1 (real-time)")


def systemic_explosiveness(panel: pd.DataFrame, swin: int = 36, realtime: bool = True,
                           level: float = 0.95) -> dict:
    """BSADF of the panel's first principal component (levels)."""
    pc = pc1_realtime(np.log(panel)) if realtime else pc1_expost(np.log(panel))
    level_series = pc - pc.min() + 1.0                       # positive "price" level
    st = current_state(level_series, swin=swin, level=level)
    st["pc1"] = pc
    return st
