"""Cross-asset factor simulation.

All simulators return (F, Z) for an h-day horizon:
    F : (n, K) factor moves in natural units
    Z : {specific key: (n,) standardized shocks scaled by sqrt(h)}

GaussianSimulator          multivariate normal with the risk-model covariance
StudentTSimulator          multivariate t (common chi-square mixing, so factor AND
                           specific shocks share tail events), variance-matched
FilteredHistoricalSimulator  Barone-Adesi / Hull-White FHS: devolatilize history with
                           EWMA vols, bootstrap whole dates (keeps the empirical
                           cross-asset dependence and tails), re-volatilize at today's vols
RegimeMixtureSimulator     calm/stress Gaussian mixture; the stress probability can be
                           driven by a signal (e.g. systemic explosiveness)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _chol(cov: np.ndarray) -> np.ndarray:
    C = np.nan_to_num(cov)
    vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
    return vecs * np.sqrt(np.clip(vals, 0, None))        # L with L L' = C


class _Base:
    def __init__(self, spec_keys: Optional[Sequence[str]] = None):
        self.spec_keys = list(spec_keys or [])

    def _normal_spec(self, n, h, rng, scale=None):
        Z = {}
        for k in self.spec_keys:
            z = rng.standard_normal(n) * np.sqrt(h)
            Z[k] = z if scale is None else z * scale
        return Z


class GaussianSimulator(_Base):
    name = "Gaussian"

    def __init__(self, cov: pd.DataFrame, spec_keys=None):
        super().__init__(spec_keys)
        self.L = _chol(cov.values)

    def draw(self, n: int, horizon: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        F = rng.standard_normal((n, self.L.shape[1])) @ self.L.T * np.sqrt(horizon)
        return F, self._normal_spec(n, horizon, rng)


class StudentTSimulator(_Base):
    name = "Student-t"

    def __init__(self, cov: pd.DataFrame, dof: float = 5.0, spec_keys=None):
        super().__init__(spec_keys)
        self.L, self.dof = _chol(cov.values), dof

    def draw(self, n: int, horizon: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        # variance-matched: Var = 1 for the standardized t draw
        w = np.sqrt((self.dof - 2) / rng.chisquare(self.dof, n))
        F = (rng.standard_normal((n, self.L.shape[1])) * w[:, None]) @ self.L.T * np.sqrt(horizon)
        return F, self._normal_spec(n, horizon, rng, scale=w)


class FilteredHistoricalSimulator(_Base):
    name = "Filtered historical"

    def __init__(self, moves: pd.DataFrame, residuals: Optional[pd.DataFrame] = None,
                 vol_halflife: float = 63, lookback: int = 2520, spec_keys=None):
        super().__init__(spec_keys)
        X = moves.iloc[-lookback:]
        lam = 0.5 ** (1 / vol_halflife)
        sq = X.pow(2)
        # EWMA variance known at the start of each day (lagged one day)
        ew = sq.ewm(alpha=1 - lam, ignore_na=True, min_periods=20).mean()
        prior = ew.shift(1)
        self.std_moves = (X / np.sqrt(prior)).replace([np.inf, -np.inf], np.nan)
        self.current_vol = np.sqrt(ew.iloc[-1]).values
        self.resid = None
        if residuals is not None and len(residuals.columns):
            R = residuals.reindex(X.index)
            rv = R.pow(2).ewm(alpha=1 - lam, ignore_na=True, min_periods=20).mean()
            self.resid = (R / np.sqrt(rv.shift(1))).replace([np.inf, -np.inf], np.nan)
        valid = self.std_moves.notna().mean(axis=1) > 0.5
        self.dates = np.flatnonzero(valid.values)

    def draw(self, n: int, horizon: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        S = np.nan_to_num(self.std_moves.values)
        R = None if self.resid is None else np.nan_to_num(self.resid.values)
        F = np.zeros((n, S.shape[1]))
        Z = {k: np.zeros(n) for k in self.spec_keys}
        for _ in range(horizon):
            idx = rng.choice(self.dates, size=n)
            F += S[idx] * np.nan_to_num(self.current_vol)
            for k in self.spec_keys:
                if R is not None and k in self.resid.columns:
                    j = self.resid.columns.get_loc(k)
                    z = R[idx, j]
                    z = np.where(z == 0, rng.standard_normal(n), z)
                else:
                    z = rng.standard_normal(n)
                Z[k] += z
        return F, Z


class RegimeMixtureSimulator(_Base):
    name = "Regime mixture"

    def __init__(self, cov_calm: pd.DataFrame, cov_stress: pd.DataFrame, p_stress: float = 0.1,
                 spec_keys=None, stress_spec_multiplier: float = 1.5):
        super().__init__(spec_keys)
        self.calm = GaussianSimulator(cov_calm, spec_keys)
        self.stress = GaussianSimulator(cov_stress, spec_keys)
        self.p, self.m = p_stress, stress_spec_multiplier

    def draw(self, n: int, horizon: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        is_stress = rng.random(n) < self.p
        Fc, Zc = self.calm.draw(n, horizon, rng)
        Fs, Zs = self.stress.draw(n, horizon, rng)
        F = np.where(is_stress[:, None], Fs, Fc)
        Z = {k: np.where(is_stress, self.m * Zs[k], Zc[k]) for k in self.spec_keys}
        return F, Z


def simulate_portfolio(portfolio, ctx, simulator, n: int = 20000, horizon: int = 1, seed=7):
    """Full-revaluation P&L: returns (total (n,), by_position DataFrame, F)."""
    F, Z = simulator.draw(n, horizon, seed)
    pnl = portfolio.revalue(ctx, F, Z)
    return pnl.sum(axis=1).values, pnl, F


def simulate_paths(portfolio, ctx, simulator, n_paths: int = 2000, n_steps: int = 12,
                   step_days: int = 21, annual_drift: float = 0.0, seed=11):
    """Linear (exposure-based) multi-period NAV paths for a risk cone.
    Returns an (n_paths, n_steps + 1) array of NAV relative to today."""
    rng = np.random.default_rng(seed)
    nav = portfolio.nav(ctx)
    e = portfolio.exposures(ctx).values / nav
    spec = portfolio.specific_terms(ctx)
    paths = np.ones((n_paths, n_steps + 1))
    mu = annual_drift * step_days / 252
    for s in range(n_steps):
        F, Z = simulator.draw(n_paths, step_days, rng)
        r = F @ e + mu
        for _, row in spec.iterrows():
            if row.key in Z:
                r = r + row.dollars / nav * row.unit_vol * Z[row.key]
        paths[:, s + 1] = paths[:, s] * np.exp(r - 0.5 * np.var(r))
    return paths
