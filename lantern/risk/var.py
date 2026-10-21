"""Value-at-Risk and Expected Shortfall from P&L samples, with Euler contributions,
historical (hypothetical) P&L for the current portfolio, and VaR backtests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


def var_es(pnl: np.ndarray, alpha: float = 0.99):
    """VaR and ES as positive loss numbers at confidence alpha."""
    loss = -np.asarray(pnl)
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    return float(q), float(tail.mean()) if tail.size else float(q)


def component_es(pnl_by_position: pd.DataFrame, alpha: float = 0.99) -> pd.DataFrame:
    """Euler ES contributions: E[-pnl_i | portfolio loss >= VaR]. They sum to the ES."""
    tot = pnl_by_position.sum(axis=1).values
    q = np.quantile(-tot, alpha)
    tail = -tot >= q
    es_i = -pnl_by_position[tail].mean()
    out = pd.DataFrame({"ES_contribution": es_i, "stand_alone_ES": [
        var_es(pnl_by_position[c].values, alpha)[1] for c in pnl_by_position]})
    out["share_%"] = 100 * out["ES_contribution"] / out["ES_contribution"].sum()
    return out.sort_values("ES_contribution", ascending=False)


def residual_history(portfolio, ctx) -> pd.DataFrame:
    """Daily standardized specific shocks (residual / unit vol) for each key that has a
    return history (equity-like securities)."""
    keys = portfolio.specific_keys(ctx)
    out = {}
    for key, uvol in keys.items():
        if key in ctx._fits and uvol > 0:
            out[key] = ctx._fits[key].residuals / uvol
    return pd.DataFrame(out)


def historical_pnl(portfolio, ctx, days: int = 750, horizon: int = 1,
                   include_specific: bool = True) -> pd.DataFrame:
    """Hypothetical P&L of today's portfolio under each historical h-day window of
    factor moves (full revaluation). Missing factor moves are set to zero."""
    X = ctx.moves.iloc[-(days + horizon - 1):]
    if horizon > 1:
        X = X.rolling(horizon).sum().iloc[horizon - 1:]
    F = np.nan_to_num(X.values)
    Z = {}
    if include_specific:
        R = residual_history(portfolio, ctx)
        if len(R.columns):
            R = R.reindex(ctx.moves.index).iloc[-(days + horizon - 1):]
            if horizon > 1:
                R = R.rolling(horizon).sum().iloc[horizon - 1:]
            Z = {k: np.nan_to_num(R[k].values) for k in R.columns}
    pnl = portfolio.revalue(ctx, F, Z)
    pnl.index = X.index
    return pnl


# ----------------------------------------------------------------------------- backtests

@dataclass
class BacktestResult:
    alpha: float
    n: int
    exceptions: int
    expected: float
    kupiec_LR: float
    kupiec_p: float
    christoffersen_LR: float
    christoffersen_p: float
    traffic_light: str
    series: pd.DataFrame


def kupiec(n: int, x: int, alpha: float):
    p = 1 - alpha
    phat = x / n if n else 0
    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)
    LR = -2 * (ll(p) - ll(phat))
    return LR, 1 - chi2.cdf(LR, 1)


def christoffersen(hits: np.ndarray):
    """Independence test of the exception sequence (first-order Markov)."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    def lg(x, q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return x * np.log(q)
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = lg(n00 + n10, 1 - p) + lg(n01 + n11, p)
    l1 = lg(n00, 1 - p01) + lg(n01, p01) + lg(n10, 1 - p11) + lg(n11, p11)
    LR = -2 * (l0 - l1)
    return LR, 1 - chi2.cdf(LR, 1)


def basel_zone(exceptions_250: int) -> str:
    return "green" if exceptions_250 <= 4 else ("yellow" if exceptions_250 <= 9 else "red")


def backtest_var(pnl: pd.Series, alpha: float = 0.99, window: int = 500,
                 method: str = "historical", halflife: float = 30) -> BacktestResult:
    """Rolling one-day VaR forecasts from past P&L only, compared with realized P&L.

    method 'historical' : empirical quantile of the previous `window` days
           'ewma'       : normal VaR with an EWMA volatility (RiskMetrics-style)
    """
    from scipy.stats import norm
    x = pnl.dropna()
    if method == "historical":
        var = (-x).rolling(window).quantile(alpha).shift(1)
    else:
        lam = 0.5 ** (1 / halflife)
        vol = np.sqrt(x.pow(2).ewm(alpha=1 - lam).mean()).shift(1)
        var = norm.ppf(alpha) * vol
    df = pd.DataFrame({"pnl": x, "VaR": var}).dropna()
    hits = (-df["pnl"] > df["VaR"]).values
    n, k = len(df), int(hits.sum())
    LRk, pk = kupiec(n, k, alpha)
    LRc, pc = christoffersen(hits)
    last = hits[-250:].sum() if n >= 250 else int(round(k * 250 / max(n, 1)))
    df["exception"] = hits
    return BacktestResult(alpha, n, k, n * (1 - alpha), LRk, pk, LRc, pc,
                          basel_zone(int(last)), df)
