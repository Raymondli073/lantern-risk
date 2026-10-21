"""European equity options (Black-Scholes-Merton), repriced exactly in every scenario.

The underlying's log return is its factor-implied return plus its specific shock (shared
with the underlying position), and implied volatility moves with a volatility factor:
    sigma' = max(sigma + vol_beta * dVIX, 1%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import Instrument


def bs_price(S, K, T, r, q, sigma, right: str):
    S, sigma = np.asarray(S, float), np.asarray(sigma, float)
    T = max(T, 1e-8)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    if right == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, right: str) -> dict:
    T = max(T, 1e-8)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    pdf = norm.pdf(d1)
    delta = np.exp(-q * T) * (norm.cdf(d1) if right == "call" else norm.cdf(d1) - 1)
    gamma = np.exp(-q * T) * pdf / (S * sq)
    vega = S * np.exp(-q * T) * pdf * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "vega": vega}


@dataclass
class EquityOption(Instrument):
    underlying: str = "SPY"
    right: str = "put"
    strike: Optional[float] = None
    strike_pct: Optional[float] = None   # strike as a fraction of spot when strike is None
    expiry: str = "2027-03-19"
    quantity: float = 0.0                # contracts; negative = short
    multiplier: float = 100.0
    implied_vol: Optional[float] = None  # default: vol_beta * VIX + vol_spread
    vol_factor: str = "VOL.VIX"
    vol_beta: float = 1.0
    vol_spread: float = 0.0
    div_yield: float = 0.013
    underlying_betas: Optional[dict] = None
    asset_class: str = "Derivatives"

    def _inputs(self, ctx):
        S = ctx.price(self.underlying)
        K = self.strike if self.strike is not None else S * (self.strike_pct or 1.0)
        T = max((pd.Timestamp(self.expiry) - ctx.asof).days / 365.25, 1e-6)
        r = float(ctx.curve().zero([T])[0])
        sig = self.implied_vol if self.implied_vol is not None else \
            self.vol_beta * ctx.level(self.vol_factor) + self.vol_spread
        return S, K, T, r, sig

    def _n(self):
        return self.quantity * self.multiplier

    def value(self, ctx) -> float:
        S, K, T, r, sig = self._inputs(ctx)
        return float(self._n() * bs_price(S, K, T, r, self.div_yield, sig, self.right))

    def greeks(self, ctx) -> dict:
        S, K, T, r, sig = self._inputs(ctx)
        g = bs_greeks(S, K, T, r, self.div_yield, sig, self.right)
        return {"spot": S, "strike": K, "T": T, "vol": sig, **g}

    def exposures(self, ctx) -> np.ndarray:
        g = self.greeks(ctx)
        beta = ctx.beta_vector(self.underlying, self.underlying_betas)
        e = self._n() * g["delta"] * g["spot"] * beta
        e[ctx.index(self.vol_factor)] += self._n() * g["vega"] * self.vol_beta
        return e

    def _uvol(self, ctx):
        return 0.0 if self.underlying_betas else ctx.fit(self.underlying).specific_vol

    def specific(self, ctx):
        g = self.greeks(ctx)
        return [(self.underlying, self._n() * g["delta"] * g["spot"], self._uvol(ctx))]

    def revalue(self, ctx, F, Z):
        S, K, T, r, sig = self._inputs(ctx)
        beta = ctx.beta_vector(self.underlying, self.underlying_betas)
        x = F @ beta
        if self.underlying in Z:
            x = x + self._uvol(ctx) * Z[self.underlying]
        s_new = S * np.exp(x)
        v_new = np.maximum(sig + self.vol_beta * F[:, ctx.index(self.vol_factor)], 0.01)
        p0 = bs_price(S, K, T, r, self.div_yield, sig, self.right)
        return self._n() * (bs_price(s_new, K, T, r, self.div_yield, v_new, self.right) - p0)

    def describe(self, ctx):
        d = super().describe(ctx)
        g = self.greeks(ctx)
        d.update(underlying=self.underlying, right=self.right, strike=round(g["strike"], 2),
                 expiry=self.expiry, vol=round(g["vol"], 4), delta=round(float(g["delta"]), 3))
        return d
