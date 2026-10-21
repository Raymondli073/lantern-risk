"""MarketContext: everything an instrument needs to price and to map itself to factors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .factors.build import build_factors
from .factors.exposures import ExposureFit, estimate_exposures
from .factors.universe import DEFAULT_UNIVERSE, by_id


@dataclass
class MarketContext:
    market: object                        # MarketData
    universe: list = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    asof: Optional[pd.Timestamp] = None
    exposure_method: str = "adaptive_lasso"
    exposure_halflife: float = 252
    exposure_lookback: int = 756
    regression_exclude: tuple = ("VOL.VIX",)

    def __post_init__(self):
        self.moves, self.levels = build_factors(self.market, self.universe)
        if self.asof is None:
            self.asof = self.market.asof
        self.asof = pd.Timestamp(self.asof)
        self.moves = self.moves.loc[:self.asof]
        self.levels = self.levels.loc[:self.asof]
        self.specs = by_id(self.universe)
        self.factor_ids = [f.id for f in self.universe]
        self._fits: Dict[str, ExposureFit] = {}

    # ----- factor state -------------------------------------------------------------
    def index(self, fid: str) -> int:
        return self.factor_ids.index(fid)

    def level(self, fid: str) -> float:
        s = self.levels[fid].dropna()
        return float(s.iloc[-1])

    def rate_nodes(self):
        """(tenors, current yields, factor ids) of the key-rate factors, tenor-sorted."""
        rates = sorted([f for f in self.universe if f.kind == "rate"], key=lambda f: f.tenor)
        return (np.array([f.tenor for f in rates]),
                np.array([self.level(f.id) for f in rates]),
                [f.id for f in rates])

    # ----- securities ---------------------------------------------------------------
    def price(self, symbol: str) -> float:
        return self.market.price(symbol, self.asof)

    def returns(self, symbol: str) -> pd.Series:
        p = self.market.prices[symbol].loc[:self.asof]
        return np.log(p.where(p > 0)).diff().rename(symbol)

    def fit(self, symbol: str) -> ExposureFit:
        if symbol not in self._fits:
            X = self.moves.drop(columns=[c for c in self.regression_exclude if c in self.moves])
            fit = estimate_exposures(self.returns(symbol), X, method=self.exposure_method,
                                     halflife=self.exposure_halflife,
                                     lookback=self.exposure_lookback)
            fit.beta = fit.beta.reindex(self.factor_ids, fill_value=0.0)
            self._fits[symbol] = fit
        return self._fits[symbol]

    def beta_vector(self, symbol: str, override: Optional[dict] = None) -> np.ndarray:
        if override:
            b = pd.Series(0.0, index=self.factor_ids)
            for k, v in override.items():
                b[k] = v
            return b.values
        return self.fit(symbol).beta.values

    def curve(self):
        from .instruments.curve import Curve
        return Curve(*self.rate_nodes())
