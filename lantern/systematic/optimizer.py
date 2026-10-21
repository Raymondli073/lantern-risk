"""Portfolio construction on the factor risk model.

The asset covariance is structured: Sigma = B S B' + diag(u^2), with B the assets'
factor exposures, S the factor covariance and u the specific vols.

  factor_targeting  "factor workbench": hit target factor exposures with the least
                    tracking error to a reference portfolio (long-only, bounded)
  min_variance, risk_parity (equal risk contribution), max_diversification
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def asset_covariance(ctx, symbols, cov: pd.DataFrame):
    B = np.array([ctx.fit(s).beta.values for s in symbols])
    u = np.array([ctx.fit(s).specific_vol for s in symbols])
    S = np.nan_to_num(cov.loc[ctx.factor_ids, ctx.factor_ids].values)
    return B @ S @ B.T + np.diag(u ** 2), B


def _solve(obj, n, lo, hi, x0=None, extra=()):
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(extra)
    res = minimize(obj, x0, method="SLSQP", bounds=[(lo, hi)] * n, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, lo, hi)
    return w / w.sum()


def min_variance(Sigma, lo=0.0, hi=1.0):
    return _solve(lambda w: w @ Sigma @ w * 1e4, len(Sigma), lo, hi)


def risk_parity(Sigma, lo=0.0, hi=1.0):
    n = len(Sigma)
    def f(y):
        return 0.5 * y @ Sigma @ y - np.log(y).sum() / n
    y0 = 1 / np.sqrt(np.diag(Sigma))
    res = minimize(f, y0, jac=lambda y: Sigma @ y - 1 / (n * y), method="L-BFGS-B",
                   bounds=[(1e-12, None)] * n)
    w = res.x / res.x.sum()
    if w.max() > hi or w.min() < lo:          # bounded ERC: minimize RC dispersion
        def g(w):
            rc = w * (Sigma @ w)
            return ((rc[:, None] - rc[None, :]) ** 2).sum() * 1e8
        w = _solve(g, n, lo, hi, np.clip(w, lo, hi))
    return w


def max_diversification(Sigma, lo=0.0, hi=1.0):
    sd = np.sqrt(np.diag(Sigma))
    return _solve(lambda w: -(w @ sd) / np.sqrt(w @ Sigma @ w), len(Sigma), lo, hi)


def factor_targeting(ctx, symbols, cov, targets: Dict[str, float],
                     reference: Optional[np.ndarray] = None, lo: float = 0.0, hi: float = 0.4,
                     penalty: float = 1e4) -> dict:
    """Weights whose portfolio exposure to each targeted factor equals the target
    (per $1 of NAV), minimizing tracking error to `reference` (default equal weight).

    The targets enter as a heavy quadratic penalty so the problem stays feasible; the
    achieved exposures are reported."""
    Sigma, B = asset_covariance(ctx, symbols, cov)
    n = len(symbols)
    ref = np.full(n, 1.0 / n) if reference is None else np.asarray(reference, float)
    k = [ctx.index(f) for f in targets]
    t = np.array(list(targets.values()))
    Bk = B[:, k]
    fvol = np.sqrt(np.clip(np.diag(np.nan_to_num(cov.values))[k], 1e-16, None))
    def obj(w):
        a = w - ref
        miss = (w @ Bk - t) * fvol                # in daily-vol units of each factor
        return 1e4 * (a @ Sigma @ a) + penalty * 1e4 * (miss @ miss)
    w = _solve(obj, n, lo, hi, ref.copy())
    a = w - ref
    return {"weights": pd.Series(w, index=symbols),
            "achieved": pd.Series(w @ Bk, index=list(targets)),
            "targets": pd.Series(t, index=list(targets)),
            "tracking_error_ann": float(np.sqrt(a @ Sigma @ a * 252)),
            "vol_ann": float(np.sqrt(w @ Sigma @ w * 252)),
            "exposures": pd.Series(w @ B, index=ctx.factor_ids)}


def construct(ctx, symbols, cov, method: str = "risk_parity", lo=0.0, hi=1.0) -> pd.Series:
    Sigma, _ = asset_covariance(ctx, symbols, cov)
    fn = {"min_variance": min_variance, "risk_parity": risk_parity,
          "max_diversification": max_diversification}[method]
    return pd.Series(fn(Sigma, lo, hi), index=symbols)


def risk_contributions(w, Sigma) -> np.ndarray:
    w = np.asarray(w)
    rc = w * (Sigma @ w)
    return rc / rc.sum()
