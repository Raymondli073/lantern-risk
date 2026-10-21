"""Performance statistics, with Newey-West t-statistics for means and alphas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def newey_west_se(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    n = len(e)
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V))


def ols_nw(y: pd.Series, X: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    Xm = np.column_stack([np.ones(len(df)), df[X.columns].values])
    yv = df["_y"].values
    b = np.linalg.lstsq(Xm, yv, rcond=None)[0]
    se = newey_west_se(Xm, yv - Xm @ b, lags)
    return pd.DataFrame({"coef": b, "t": b / se}, index=["alpha"] + list(X.columns))


def stats(r: pd.Series, periods: int = 12, rf: pd.Series = None) -> dict:
    r = r.dropna()
    ex = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    mu, sd = ex.mean() * periods, ex.std() * np.sqrt(periods)
    tstat = ols_nw(ex, pd.DataFrame(index=ex.index))["t"].iloc[0] if len(ex) > 12 else np.nan
    return {"Ann. return %": 100 * mu, "Ann. vol %": 100 * sd,
            "Sharpe": mu / sd if sd > 0 else np.nan, "t(mean, NW)": tstat,
            "Max drawdown %": 100 * dd.min(), "Hit rate %": 100 * (r > 0).mean(),
            "Months": len(r)}


def stats_table(R: pd.DataFrame, periods: int = 12, factors: pd.DataFrame = None) -> pd.DataFrame:
    rows = {}
    for c in R:
        s = stats(R[c], periods)
        if factors is not None:
            reg = ols_nw(R[c], factors)
            s["Alpha %/mo"] = 100 * reg.loc["alpha", "coef"]
            s["t(alpha)"] = reg.loc["alpha", "t"]
        rows[c] = s
    return pd.DataFrame(rows).T


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1
