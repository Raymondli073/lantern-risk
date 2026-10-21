"""Equity-like securities (stocks, ETFs, funds, crypto) mapped by return regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Instrument


@dataclass
class Equity(Instrument):
    symbol: str = ""
    quantity: Optional[float] = None
    market_value: Optional[float] = None
    betas: Optional[dict] = None          # override the regression with known exposures
    specific_vol: Optional[float] = None  # daily, overrides the regression residual vol
    asset_class: str = "Equity"

    def value(self, ctx) -> float:
        if self.quantity is not None:
            return self.quantity * ctx.price(self.symbol)
        return float(self.market_value)

    def _beta(self, ctx):
        return ctx.beta_vector(self.symbol, self.betas)

    def _uvol(self, ctx):
        if self.specific_vol is not None:
            return float(self.specific_vol)
        return 0.0 if self.betas else ctx.fit(self.symbol).specific_vol

    def exposures(self, ctx) -> np.ndarray:
        return self.value(ctx) * self._beta(ctx)

    def specific(self, ctx):
        return [(self.symbol, self.value(ctx), self._uvol(ctx))]

    def revalue(self, ctx, F, Z):
        r = F @ self._beta(ctx)
        z = Z.get(self.symbol)
        if z is not None:
            r = r + self._uvol(ctx) * z
        return self.value(ctx) * np.expm1(r)

    def describe(self, ctx):
        d = super().describe(ctx)
        d.update(symbol=self.symbol, price=ctx.price(self.symbol) if self.quantity is not None
                 else None, quantity=self.quantity)
        return d


@dataclass
class FXForward(Instrument):
    """Long `notional_usd` of foreign currency against USD (factor = FX.xxx)."""
    factor: str = "FX.EUR"
    notional_usd: float = 0.0
    asset_class: str = "FX"

    def value(self, ctx) -> float:
        return 0.0                         # an at-market forward has zero value

    def exposures(self, ctx):
        e = np.zeros(len(ctx.factor_ids))
        e[ctx.index(self.factor)] = self.notional_usd
        return e

    def revalue(self, ctx, F, Z):
        return self.notional_usd * np.expm1(F[:, ctx.index(self.factor)])


@dataclass
class Cash(Instrument):
    amount: float = 0.0
    asset_class: str = "Cash"

    def value(self, ctx):
        return float(self.amount)

    def exposures(self, ctx):
        return np.zeros(len(ctx.factor_ids))

    def revalue(self, ctx, F, Z):
        return np.zeros(F.shape[0])


@dataclass
class FactorProxy(Instrument):
    """A position described directly by its factor exposures ("building block" mapping),
    e.g. a private-markets fund or an OTC overlay: P&L = value * (exp(b.F) - 1)."""
    market_value: float = 0.0
    betas: Optional[dict] = None
    specific_vol: float = 0.0
    asset_class: str = "Alternatives"

    def value(self, ctx):
        return float(self.market_value)

    def _beta(self, ctx):
        return ctx.beta_vector("", self.betas or {})

    def exposures(self, ctx):
        return self.market_value * self._beta(ctx)

    def specific(self, ctx):
        return [(self.id, self.market_value, self.specific_vol)]

    def revalue(self, ctx, F, Z):
        r = F @ self._beta(ctx)
        if self.id in Z:
            r = r + self.specific_vol * Z[self.id]
        return self.market_value * np.expm1(r)
