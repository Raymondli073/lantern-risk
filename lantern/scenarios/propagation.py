"""Market-driven scenarios: shock a few variables, imply the rest.

Following the market-driven scenario approach described publicly by BlackRock, a
scenario fixes explicit shocks s on a subset k of risk variables and infers the
remaining variables u by their conditional expectation under the factor model:

    E[x_u | x_k = s]   = S_uk S_kk^-1 s
    Var[x_u | x_k = s] = S_uu - S_uk S_kk^-1 S_ku        (the scenario's prediction error)

Variables can be factors or individual securities (a security's return has covariance
S b with the factors and variance b'S b + u^2), so a "stock X falls 30%" scenario implies
the factor moves consistent with it. Plausibility is the Mahalanobis distance of the
explicit shocks, d^2 = s' S_kk^-1 s ~ chi2(k) under the model, turned into a probability
and a return period at the scenario horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..factors.build import to_display


@dataclass
class ScenarioResult:
    name: str
    explicit: Dict[str, float]
    factor_moves: pd.Series            # natural units, all factors
    pnl_by_position: pd.Series
    pnl: float
    pnl_pct: float
    pred_error_95: float               # $ half-width of the 95% band (linear approximation)
    mahalanobis: float
    probability: float                 # P(d^2 >= observed) per horizon window
    return_period_years: float
    horizon_days: int
    description: str = ""
    security_moves: Dict[str, float] = field(default_factory=dict)

    def table(self, specs) -> pd.DataFrame:
        rows = []
        for fid, v in self.factor_moves.items():
            k = specs[fid].kind
            rows.append({"factor": fid, "group": specs[fid].group,
                         "move": to_display(v, k), "unit": {"return": "%", "rate": "bp",
                         "spread": "bp", "vol": "pts"}[k],
                         "explicit": fid in self.explicit})
        return pd.DataFrame(rows).set_index("factor")


def conditional(cov: np.ndarray, k_idx, s: np.ndarray, ridge: float = 1e-10):
    """Conditional mean of all variables and conditional covariance given x_k = s."""
    S = np.nan_to_num(cov)
    Skk = S[np.ix_(k_idx, k_idx)]
    Skk = Skk + ridge * np.trace(Skk) / max(len(k_idx), 1) * np.eye(len(k_idx))
    Sak = S[:, k_idx]
    sol = np.linalg.solve(Skk, s)
    mu = Sak @ sol
    mu[k_idx] = s
    Ccond = S - Sak @ np.linalg.solve(Skk, Sak.T)
    Ccond[k_idx, :] = 0.0
    Ccond[:, k_idx] = 0.0
    d2 = float(s @ sol)
    return mu, Ccond, d2


def run_scenario(portfolio, ctx, cov: pd.DataFrame, shocks: Dict[str, float],
                 name: str = "Scenario", horizon_days: int = 21,
                 security_shocks: Optional[Dict[str, float]] = None,
                 description: str = "") -> ScenarioResult:
    """shocks: factor id -> move (natural units); security_shocks: symbol -> log return.
    The covariance is scaled to the horizon."""
    security_shocks = dict(security_shocks or {})
    fids = list(ctx.factor_ids)
    K = len(fids)
    S = np.nan_to_num(cov.loc[fids, fids].values) * horizon_days

    # augment with shocked securities
    syms = list(security_shocks)
    B = np.array([ctx.fit(s).beta.values for s in syms]).reshape(len(syms), K)
    u2 = np.array([ctx.fit(s).specific_vol ** 2 * horizon_days for s in syms])
    if syms:
        top = np.hstack([S, S @ B.T])
        bot = np.hstack([B @ S, B @ S @ B.T + np.diag(u2)])
        S = np.vstack([top, bot])
    k_idx = [fids.index(f) for f in shocks] + [K + j for j in range(len(syms))]
    s = np.array(list(shocks.values()) + [security_shocks[x] for x in syms], float)
    if not k_idx:
        raise ValueError("a scenario needs at least one explicit shock")
    mu, Ccond, d2 = conditional(S, k_idx, s)
    F = mu[:K][None, :]

    # specific shocks: shocked securities get exactly their move; others none
    Z = {}
    for j, sym in enumerate(syms):
        uvol = ctx.fit(sym).specific_vol
        Z[sym] = np.array([(security_shocks[sym] - B[j] @ mu[:K]) / uvol]) if uvol > 0 else np.zeros(1)
    pnl_pos = portfolio.revalue(ctx, F, Z).iloc[0]
    nav = portfolio.nav(ctx)

    # prediction error: conditional factor variance + specific variance of unshocked keys
    e = portfolio.exposures(ctx).values
    var = float(e @ Ccond[:K, :K] @ e)
    spec = portfolio.specific_terms(ctx)
    if len(spec):
        A = spec.groupby("key")["dollars"].sum()
        u = spec.groupby("key")["unit_vol"].first()
        keep = [k for k in A.index if k not in security_shocks]
        var += float(((A[keep] * u[keep]) ** 2).sum()) * horizon_days
    k = len(k_idx)
    prob = float(1 - chi2.cdf(d2, k))
    periods_per_year = 252.0 / horizon_days
    rp = np.inf if prob <= 0 else 1.0 / (prob * periods_per_year)
    return ScenarioResult(name=name, explicit=dict(shocks), factor_moves=pd.Series(mu[:K], index=fids),
                          pnl_by_position=pnl_pos, pnl=float(pnl_pos.sum()),
                          pnl_pct=100 * float(pnl_pos.sum()) / nav,
                          pred_error_95=1.96 * np.sqrt(max(var, 0.0)), mahalanobis=np.sqrt(d2),
                          probability=prob, return_period_years=rp, horizon_days=horizon_days,
                          description=description, security_moves=security_shocks)


def historical_plausibility(moves: pd.DataFrame, cov: pd.DataFrame, shocks: Dict[str, float],
                            horizon_days: int = 21) -> float:
    """Share of historical h-day windows whose Mahalanobis distance (on the same shocked
    variables) is at least that of the scenario: an empirical, fat-tail-aware check."""
    ids = list(shocks)
    X = moves[ids].rolling(horizon_days).sum().dropna()
    S = np.nan_to_num(cov.loc[ids, ids].values) * horizon_days
    Si = np.linalg.pinv(S)
    s = np.array(list(shocks.values()))
    d2 = s @ Si @ s
    hist = np.einsum("ij,jk,ik->i", X.values, Si, X.values)
    return float(np.mean(hist >= d2))
