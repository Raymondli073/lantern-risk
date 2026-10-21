"""Factor-timing strategies: time-series factor momentum (TSFM, Ehsani & Linnainmaa 2022)
and time-series factor explosiveness (TSFE, Li & Urga).

Signals are observed at month-end t and earn the factor return of month t + 1.
Long ('L')        : equal weight across signalled factors (cash if none).
Long-short ('LS') : equal-weight long the signalled factors, short the rest.
Costs: `cost_bps` is a round-trip cost, charged as cost/2 per unit of one-way turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import explosive


def cumulative_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative value of $1 in each factor (the level series the test runs on)."""
    return (1 + returns.fillna(0)).cumprod().where(returns.notna().cummax())


def tsfm_signal(returns: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    past = np.log1p(returns).rolling(lookback, min_periods=lookback).sum()
    return (past > 0).astype(float).where(past.notna())


def tsfe_signal(returns: pd.DataFrame, swin: int = 36, level: float = 0.95):
    """Returns (I, E, CV): the indicator and the statistic/critical-value panels."""
    E, CV, I = explosive.panel(cumulative_index(returns), swin=swin, level=level)
    return I, E, CV


def weights(signal: pd.DataFrame, mode: str = "L") -> pd.DataFrame:
    s = signal.fillna(0)
    avail = signal.notna()
    nl = s.sum(axis=1)
    long = s.div(nl.where(nl > 0), axis=0).fillna(0)
    if mode == "L":
        return long
    short_mask = (avail & (s == 0)).astype(float)
    ns = short_mask.sum(axis=1)
    short = short_mask.div(ns.where(ns > 0), axis=0).fillna(0)
    return long - short


def strategy_returns(returns: pd.DataFrame, w: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """Weights at t earn returns at t + 1, net of costs from rebalancing at t."""
    gross = (w * returns.shift(-1)).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    net = gross - cost_bps / 1e4 / 2 * turnover
    return net.shift(1).dropna()            # label each return with its own month


def run_all(returns: pd.DataFrame, swin: int = 36, cost_bps: float = 10.0,
            start=None) -> pd.DataFrame:
    """Monthly returns of TSFE/TSFM (L and LS) and equal-weight buy & hold."""
    I, _, _ = tsfe_signal(returns, swin)
    M = tsfm_signal(returns)
    M = M.where(I.notna())                     # same sample for a fair comparison
    out = {}
    for name, sig in (("TSFE", I), ("TSFM", M)):
        for mode in ("L", "LS"):
            out[f"{name} ({mode})"] = strategy_returns(returns, weights(sig, mode), cost_bps)
    avail = I.notna().astype(float).where(I.notna())
    out["Buy & hold (EW)"] = strategy_returns(returns, weights(avail, "L"), 0.0)
    df = pd.DataFrame(out)
    first = I.dropna(how="all").index[0]
    df = df.loc[df.index > first]
    return df.loc[start:] if start else df
