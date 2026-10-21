"""Instrument interface.

Every instrument provides
  value(ctx)              current market value in USD
  exposures(ctx)          linear $ exposures: dV per unit move of each factor (length K)
  specific(ctx)           list of (key, $ per unit z, unit vol): idiosyncratic risk terms.
                          The z draws are shared by key, so an option and its underlying
                          stock see the same specific shock.
  revalue(ctx, F, Z)      full revaluation P&L for n scenarios: F is (n, K) factor moves,
                          Z maps key -> (n,) standardized specific shocks (already scaled
                          to the horizon). Non-linear instruments are repriced exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Instrument:
    id: str

    asset_class: str = "Other"

    def value(self, ctx) -> float:
        raise NotImplementedError

    def exposures(self, ctx) -> np.ndarray:
        raise NotImplementedError

    def specific(self, ctx) -> List[Tuple[str, float, float]]:
        return []

    def revalue(self, ctx, F: np.ndarray, Z: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def describe(self, ctx) -> dict:
        return {"id": self.id, "asset_class": self.asset_class}
