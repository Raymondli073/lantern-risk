"""Returns-based factor exposures with exponentially weighted (adaptive) LASSO.

For securities without a structural model (stocks, ETFs, funds, crypto), exposures
are estimated by regressing daily log returns on the factor moves. Many factors are
correlated (regions, styles, rates), so plain OLS over-fits; the adaptive LASSO
(Zou, 2006) selects a sparse set of factors with oracle properties, and the selected
set is refit by weighted least squares to remove the shrinkage bias.
The residual is the security's specific (idiosyncratic) return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExposureFit:
    beta: pd.Series                 # per factor, in log-return per unit factor move
    specific_vol: float             # daily vol of the residual
    r2: float
    n_obs: int
    method: str
    residuals: pd.Series = field(repr=False, default=None)


def _ridge(X, y, alpha):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)


def _lasso_bic(X, y, n_eff):
    """LASSO path by coordinate descent; the penalty is chosen by BIC with the noise
    variance taken from the full least-squares fit (as in LassoLarsIC). X and y are
    already weighted, so n_eff (the effective sample size) replaces n."""
    from sklearn.linear_model import lasso_path
    p = X.shape[1]
    amax = np.max(np.abs(X.T @ y)) / len(y)
    grid = amax * np.logspace(0, -4, 60)
    alphas, coefs, _ = lasso_path(X, y, alphas=grid)
    rss_full = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    sigma2 = rss_full / max(1.0 - p / n_eff, 0.05)
    best, best_bic = np.zeros(p), np.inf
    for j in range(coefs.shape[1]):
        b = coefs[:, j]
        bic = n_eff * np.sum((y - X @ b) ** 2) / sigma2 + np.log(n_eff) * np.count_nonzero(b)
        if bic < best_bic:
            best, best_bic = b, bic
    return best


def estimate_exposures(y: pd.Series, X: pd.DataFrame, method: str = "adaptive_lasso",
                       halflife: float = 252, lookback: int = 756, gamma: float = 1.0,
                       ridge_alpha: float = 1e-3, min_obs: int = 120,
                       max_missing: float = 0.2) -> ExposureFit:
    """Regress returns `y` on factor moves `X` (both daily, aligned by date)."""
    df = pd.concat([y.rename("_y"), X], axis=1).loc[y.first_valid_index():]
    df = df.dropna(subset=["_y"]).iloc[-lookback:]
    cols = [c for c in X.columns if df[c].isna().mean() <= max_missing]
    df = df[["_y"] + cols].dropna()
    if len(df) < min_obs:
        raise ValueError(f"only {len(df)} observations for {y.name}")
    yv, Xv = df["_y"].values, df[cols].values
    w = 0.5 ** (np.arange(len(df))[::-1] / halflife) if halflife else np.ones(len(df))
    w = w / w.sum()
    mx, my = w @ Xv, w @ yv
    sw = np.sqrt(w)[:, None]
    sd = np.sqrt(w @ (Xv - mx) ** 2)
    sd[sd == 0] = 1.0
    Z = (Xv - mx) / sd * sw
    t = (yv - my) * sw[:, 0]

    if method == "ols":
        bz = np.linalg.lstsq(Z, t, rcond=None)[0]
    elif method == "ridge":
        bz = _ridge(Z, t, ridge_alpha)
    elif method == "adaptive_lasso":
        b0 = _ridge(Z, t, ridge_alpha)
        scale = np.abs(b0) ** gamma
        bt = _lasso_bic(Z * scale, t, 1.0 / np.sum(w ** 2))
        sel = np.flatnonzero(bt)
        bz = np.zeros(len(cols))
        if sel.size:                         # post-selection weighted least squares
            bz[sel] = np.linalg.lstsq(Z[:, sel], t, rcond=None)[0]
    else:
        raise ValueError(method)

    beta = bz / sd
    fitted = Xv @ beta
    resid = yv - my - (fitted - mx @ beta)
    spec_var = w @ resid ** 2
    tot_var = w @ (yv - my) ** 2
    out = pd.Series(0.0, index=X.columns)
    out[cols] = beta
    return ExposureFit(beta=out, specific_vol=float(np.sqrt(spec_var)),
                       r2=float(1 - spec_var / tot_var) if tot_var > 0 else 0.0,
                       n_obs=len(df), method=method,
                       residuals=pd.Series(resid, index=df.index, name=y.name))
