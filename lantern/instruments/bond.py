"""Fixed-coupon bonds priced off the key-rate curve plus a credit spread."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import Instrument


@dataclass
class Bond(Instrument):
    face: float = 1_000_000.0
    coupon: float = 0.04                 # annual coupon rate
    maturity: str = "2035-08-15"
    frequency: int = 2
    spread: float = 0.0                  # spread over the Treasury curve, decimal
    spread_factor: Optional[str] = None  # e.g. "CREDIT.IG"; None for Treasuries
    spread_vol: float = 0.0              # daily idiosyncratic spread vol, decimal
    asset_class: str = "Fixed Income"

    def cashflows(self, asof):
        asof, mat = pd.Timestamp(asof), pd.Timestamp(self.maturity)
        months = 12 // self.frequency
        dates, d = [], mat
        while d > asof:
            dates.append(d)
            d = d - pd.DateOffset(months=months)
        dates = dates[::-1]
        t = np.array([(x - asof).days / 365.25 for x in dates])
        cf = np.full(t.size, self.coupon / self.frequency)
        if cf.size:
            cf[-1] += 1.0
        return t, cf

    def _parts(self, ctx):
        curve = ctx.curve()
        t, cf = self.cashflows(ctx.asof)
        W = curve.weights(t)
        z = W @ curve.yields
        df = cf * np.exp(-(z + self.spread) * t)
        return curve, t, W, z, df

    def value(self, ctx) -> float:
        return float(self.face * self._parts(ctx)[4].sum())

    def exposures(self, ctx) -> np.ndarray:
        curve, t, W, z, df = self._parts(ctx)
        e = np.zeros(len(ctx.factor_ids))
        krd = -self.face * (df * t) @ W                 # dV / dy_k for each node
        for fid, v in zip(curve.ids, krd):
            e[ctx.index(fid)] = v
        if self.spread_factor:
            e[ctx.index(self.spread_factor)] = -self.face * (df * t).sum()
        return e

    def specific(self, ctx):
        if not self.spread_vol:
            return []
        _, t, _, _, df = self._parts(ctx)
        return [(self.id, -self.face * (df * t).sum(), self.spread_vol)]

    def revalue(self, ctx, F, Z):
        curve, t, W, z, df0 = self._parts(ctx)
        dY = F[:, [ctx.index(f) for f in curve.ids]]          # (n, nodes)
        zz = z[None, :] + dY @ W.T                              # (n, cashflows)
        s = np.full(F.shape[0], self.spread)
        if self.spread_factor:
            s = s + F[:, ctx.index(self.spread_factor)]
        if self.spread_vol and self.id in Z:
            s = s + self.spread_vol * Z[self.id]
        _, cf = self.cashflows(ctx.asof)
        v = self.face * (cf[None, :] * np.exp(-(zz + s[:, None]) * t[None, :])).sum(axis=1)
        return v - self.face * df0.sum()

    def analytics(self, ctx) -> dict:
        curve, t, W, z, df = self._parts(ctx)
        v = df.sum()
        dur = (df * t).sum() / v
        conv = (df * t * t).sum() / v
        ytm = float(np.average(z, weights=df) + self.spread)
        krd = (df * t) @ W / v
        return {"price_per_100": 100 * v, "duration": dur, "convexity": conv,
                "yield": ytm, "dv01": self.face * (df * t).sum() * 1e-4,
                **{f"KRD {fid}": k for fid, k in zip(curve.ids, krd)}}

    def describe(self, ctx):
        d = super().describe(ctx)
        a = self.analytics(ctx)
        d.update(coupon=self.coupon, maturity=self.maturity, spread_bp=1e4 * self.spread,
                 duration=round(a["duration"], 2), yield_pct=round(100 * a["yield"], 3))
        return d
