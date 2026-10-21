"""Ex-ante (forward-looking) factor risk and its Euler decomposition.

With $ exposures e (K) and factor covariance S (K x K, daily), and specific terms
a_ik ($ per unit z of key k from position i) with unit vols u_k:

    factor variance    e' S e
    specific variance  sum_k (sum_i a_ik)^2 u_k^2
    total vol          sigma = sqrt(factor + specific)

Contributions are Euler allocations (they sum to sigma): factor k contributes
e_k (S e)_k / sigma; position i contributes e_i' S e / sigma + sum_k a_ik u_k^2 A_k / sigma.
Everything is reported annualized (x sqrt(252)) and as % of NAV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = np.sqrt(252.0)


@dataclass
class RiskReport:
    nav: float
    total_vol: float                 # annualized, fraction of NAV
    factor_vol: float
    specific_vol: float
    by_factor: pd.DataFrame          # exposure, factor vol, marginal, contribution, %
    by_group: pd.DataFrame
    by_position: pd.DataFrame
    label: str = "Total risk"

    def summary(self) -> dict:
        return {"NAV": self.nav, self.label + " (ann. %)": 100 * self.total_vol,
                "Factor (ann. %)": 100 * self.factor_vol,
                "Specific (ann. %)": 100 * self.specific_vol,
                "Factor share of variance (%)": 100 * (self.factor_vol / self.total_vol) ** 2
                if self.total_vol else 0.0}


def _clean(cov: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(cov.values)


def risk_decomposition(exposure_matrix: pd.DataFrame, specific: pd.DataFrame,
                       cov: pd.DataFrame, nav: float, specs: dict,
                       label: str = "Total risk") -> RiskReport:
    """exposure_matrix: positions x factors ($/unit); specific: rows (position, key,
    dollars, unit_vol); cov: daily factor covariance."""
    S = _clean(cov.loc[exposure_matrix.columns, exposure_matrix.columns])
    E = exposure_matrix.values
    e = E.sum(0)
    Se = S @ e
    fvar = float(e @ Se)

    pos_spec = pd.Series(0.0, index=exposure_matrix.index)
    svar = 0.0
    if len(specific):
        A = specific.groupby("key")["dollars"].sum()
        u = specific.groupby("key")["unit_vol"].first()
        svar = float(((A * u) ** 2).sum())
        for _, row in specific.iterrows():
            pos_spec[row.position] += row.dollars * row.unit_vol ** 2 * A[row.key]
    var = fvar + svar
    sigma = np.sqrt(max(var, 0.0))
    sd = sigma if sigma > 0 else 1.0

    fvol = np.sqrt(np.clip(np.diag(S), 0, None))
    by_factor = pd.DataFrame({
        "group": [specs[f].group for f in exposure_matrix.columns],
        "kind": [specs[f].kind for f in exposure_matrix.columns],
        "exposure_$": e,
        "factor_vol_ann": fvol * ANN,
        "stand_alone_%": 100 * np.abs(e) * fvol * ANN / nav,
        "marginal_%": 100 * Se / sd * ANN / nav,
        "contribution_%": 100 * e * Se / sd * ANN / nav,
    }, index=exposure_matrix.columns)
    by_factor["share_of_risk_%"] = 100 * e * Se / var if var > 0 else 0.0

    grp = by_factor.groupby("group")[["contribution_%", "share_of_risk_%"]].sum()
    spec_row = pd.DataFrame({"contribution_%": [100 * svar / sd * ANN / nav],
                             "share_of_risk_%": [100 * svar / var if var > 0 else 0.0]},
                            index=["Specific"])
    by_group = pd.concat([grp, spec_row]).sort_values("contribution_%", ascending=False)

    pos_f = E @ Se / sd
    by_position = pd.DataFrame({
        "factor_contribution_%": 100 * pos_f * ANN / nav,
        "specific_contribution_%": 100 * pos_spec.values / sd * ANN / nav,
    }, index=exposure_matrix.index)
    by_position["contribution_%"] = by_position.sum(axis=1)
    by_position["share_of_risk_%"] = 100 * by_position["contribution_%"] / (100 * sigma * ANN / nav) \
        if sigma > 0 else 0.0
    return RiskReport(nav=nav, total_vol=sigma * ANN / nav, factor_vol=np.sqrt(fvar) * ANN / nav,
                      specific_vol=np.sqrt(svar) * ANN / nav, by_factor=by_factor,
                      by_group=by_group, by_position=by_position, label=label)


def ex_ante_risk(portfolio, ctx, cov: pd.DataFrame) -> RiskReport:
    return risk_decomposition(portfolio.exposure_matrix(ctx), portfolio.specific_terms(ctx),
                              cov, portfolio.nav(ctx), ctx.specs)


def active_risk(portfolio, ctx, cov: pd.DataFrame, benchmark=None) -> RiskReport:
    """Ex-ante tracking error against a benchmark scaled to the portfolio's NAV."""
    bench = benchmark or portfolio.benchmark_portfolio(ctx)
    if bench is None:
        raise ValueError("portfolio has no benchmark")
    Ep, Eb = portfolio.exposure_matrix(ctx), bench.exposure_matrix(ctx)
    Eb.index = [f"BM:{i}" for i in Eb.index]
    E = pd.concat([Ep, -Eb])
    sp, sb = portfolio.specific_terms(ctx), bench.specific_terms(ctx).copy()
    sb["position"] = "BM:" + sb["position"]
    sb["dollars"] = -sb["dollars"]
    return risk_decomposition(E, pd.concat([sp, sb], ignore_index=True), cov,
                              portfolio.nav(ctx), ctx.specs, label="Tracking error")


def parametric_var(daily_vol_frac: float, nav: float, alpha: float = 0.99, horizon: int = 1,
                   dist: str = "normal", dof: float = 5.0):
    """(VaR, ES) in $ under a normal or a variance-matched Student-t P&L."""
    s = daily_vol_frac * nav * np.sqrt(horizon)
    if dist == "normal":
        z = norm.ppf(alpha)
        return s * z, s * norm.pdf(z) / (1 - alpha)
    from scipy.stats import t
    scale = s * np.sqrt((dof - 2) / dof)
    q = t.ppf(alpha, dof)
    es = scale * (t.pdf(q, dof) / (1 - alpha)) * (dof + q ** 2) / (dof - 1)
    return scale * q, es
