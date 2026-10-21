"""Lantern: a one-object facade over the data, risk, simulation and scenario engines."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .context import MarketContext
from .factors.covariance import ewma_covariance, fill_missing, sample_covariance
from .factors.universe import DEFAULT_UNIVERSE, universe_symbols
from .portfolio import Portfolio
from .risk import parametric, var as varmod
from .scenarios import bubble, library
from .scenarios.propagation import run_scenario
from .simulation import engine

STRESS_WINDOWS = (("2008-09-01", "2009-03-31"), ("2020-02-15", "2020-04-30"))


def portfolio_symbols(portfolio: Portfolio) -> list:
    out = []
    for p in portfolio.positions:
        for attr in ("symbol", "underlying"):
            v = getattr(p, attr, None)
            if v:
                out.append(v)
    out += list((portfolio.benchmark or {}).keys())
    return out


class Lantern:
    """Build once, then ask for risk, VaR, scenarios, bubble scans and optimizations.

    >>> lt = Lantern(Portfolio.from_yaml("examples/multi_asset.yaml"), synthetic=True)
    >>> lt.risk().summary()
    """

    def __init__(self, portfolio: Portfolio, market=None, synthetic: bool = False,
                 asof=None, universe=DEFAULT_UNIVERSE, exposure_method: str = "adaptive_lasso",
                 vol_halflife: float = 63, corr_halflife: float = 252, lookback: int = 2520,
                 start: str = "2004-01-01", refresh: bool = False):
        if market is None:
            if synthetic:
                from .data.synthetic import synthetic_market
                market = synthetic_market()
            else:
                from .data.market import load_market
                market = load_market(universe_symbols(universe) + portfolio_symbols(portfolio),
                                     start=start, refresh=refresh)
        self.market = market
        self.portfolio = portfolio
        self.ctx = MarketContext(market, list(universe), asof=asof, exposure_method=exposure_method)
        self.cov = ewma_covariance(self.ctx.moves, vol_halflife, corr_halflife, lookback)
        self.cov = fill_missing(self.cov, sample_covariance(self.ctx.moves))
        self._stress = None

    # ----- covariances ----------------------------------------------------------------
    @property
    def stress_cov(self) -> pd.DataFrame:
        if self._stress is None:
            X = pd.concat([self.ctx.moves.loc[a:b] for a, b in STRESS_WINDOWS])
            S = sample_covariance(X, min_obs=40) if len(X) > 40 else self.cov
            self._stress = fill_missing(S, self.cov)
        return self._stress

    def covariance(self, regime: str = "current") -> pd.DataFrame:
        return self.stress_cov if regime == "stress" else self.cov

    # ----- ex-ante risk ---------------------------------------------------------------
    def risk(self, regime: str = "current") -> parametric.RiskReport:
        return parametric.ex_ante_risk(self.portfolio, self.ctx, self.covariance(regime))

    def active_risk(self, regime: str = "current") -> parametric.RiskReport:
        return parametric.active_risk(self.portfolio, self.ctx, self.covariance(regime))

    def holdings(self) -> pd.DataFrame:
        return self.portfolio.holdings_table(self.ctx)

    # ----- simulation -----------------------------------------------------------------
    def simulator(self, method: str = "student_t", regime: str = "current", p_stress: float = 0.1,
                  dof: float = 5.0):
        keys = list(self.portfolio.specific_keys(self.ctx).index)
        cov = self.covariance(regime)
        if method == "gaussian":
            return engine.GaussianSimulator(cov, keys)
        if method == "student_t":
            return engine.StudentTSimulator(cov, dof, keys)
        if method == "fhs":
            return engine.FilteredHistoricalSimulator(
                self.ctx.moves, varmod.residual_history(self.portfolio, self.ctx), spec_keys=keys)
        if method == "regime":
            return engine.RegimeMixtureSimulator(self.cov, self.stress_cov, p_stress, keys)
        raise ValueError(method)

    def var(self, method: str = "student_t", alpha: float = 0.99, horizon: int = 1,
            n: int = 20000, regime: str = "current", seed: int = 7, **kw) -> dict:
        """Monte Carlo (full revaluation) or historical VaR/ES, with Euler contributions."""
        nav = self.portfolio.nav(self.ctx)
        if method == "historical":
            pnl_pos = varmod.historical_pnl(self.portfolio, self.ctx, days=kw.get("days", 2520),
                                            horizon=horizon)
            pnl = pnl_pos.sum(axis=1).values
        elif method == "parametric":
            rr = self.risk(regime)
            v, es = parametric.parametric_var(rr.total_vol / np.sqrt(252), nav, alpha, horizon)
            return {"method": method, "VaR": v, "ES": es, "VaR_%": 100 * v / nav,
                    "ES_%": 100 * es / nav, "pnl": None, "contributions": None}
        else:
            sim = self.simulator(method, regime, **kw)
            pnl, pnl_pos, _ = engine.simulate_portfolio(self.portfolio, self.ctx, sim, n, horizon, seed)
        v, es = varmod.var_es(pnl, alpha)
        return {"method": method, "VaR": v, "ES": es, "VaR_%": 100 * v / nav, "ES_%": 100 * es / nav,
                "pnl": pnl, "contributions": varmod.component_es(pnl_pos.reset_index(drop=True), alpha)}

    def var_comparison(self, alpha: float = 0.99, horizon: int = 1, n: int = 20000) -> pd.DataFrame:
        runs = {m: self.var(m, alpha, horizon, n)
                for m in ("parametric", "gaussian", "student_t", "fhs", "historical", "regime")}
        runs["stressed (gaussian)"] = self.var("gaussian", alpha, horizon, n, regime="stress")
        return pd.DataFrame({m: {"VaR $": r["VaR"], "ES $": r["ES"], "VaR %": r["VaR_%"],
                                 "ES %": r["ES_%"]} for m, r in runs.items()}).T

    def backtest(self, alpha: float = 0.99, days: int = 1500, window: int = 500,
                 method: str = "historical"):
        pnl = varmod.historical_pnl(self.portfolio, self.ctx, days=days).sum(axis=1)
        return varmod.backtest_var(pnl, alpha, window, method)

    def risk_cone(self, months: int = 12, n_paths: int = 2000, annual_drift: float = 0.0,
                  method: str = "student_t") -> np.ndarray:
        return engine.simulate_paths(self.portfolio, self.ctx, self.simulator(method),
                                     n_paths, months, 21, annual_drift)

    # ----- scenarios ------------------------------------------------------------------
    def scenario(self, shocks: Dict[str, float], name: str = "Custom", horizon_days: int = 21,
                 security_shocks: Optional[Dict[str, float]] = None, regime: str = "current"):
        return run_scenario(self.portfolio, self.ctx, self.covariance(regime), shocks, name,
                            horizon_days, security_shocks)

    def historical_scenarios(self, regime: str = "current"):
        cov = self.covariance(regime)
        return [r for r in (library.run_historical(self.portfolio, self.ctx, cov, ep)
                            for ep in library.HISTORICAL
                            if pd.Timestamp(ep.end) <= self.ctx.asof) if r is not None]

    def hypothetical_scenarios(self, regime: str = "current", severity: float = 1.0,
                               horizon_days: int = 21):
        return library.run_library(self.portfolio, self.ctx, self.covariance(regime),
                                   horizon_days, severity)

    def bubble_scan(self, swin: int = 36, level: float = 0.95) -> pd.DataFrame:
        return bubble.scan(self.ctx, self.portfolio, swin, level)

    def bubble_scenarios(self, flags: Optional[pd.DataFrame] = None, regime: str = "stress"):
        flags = self.bubble_scan() if flags is None else flags
        return bubble.burst_scenarios(self.portfolio, self.ctx, self.covariance(regime), flags)
