"""Portfolio: a list of instruments, with aggregation of values, exposures and risk terms."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .instruments import Equity, Instrument, make


@dataclass
class Portfolio:
    name: str
    positions: List[Instrument] = field(default_factory=list)
    benchmark: Optional[Dict[str, float]] = None      # symbol -> weight (equity-like)
    description: str = ""

    # ----- construction ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        pos = []
        for i, p in enumerate(d.get("positions", [])):
            p = dict(p)
            p.setdefault("id", p.get("symbol") or f"pos{i}")
            pos.append(make(p))
        return cls(name=d.get("name", "Portfolio"), positions=pos,
                   benchmark=d.get("benchmark"), description=d.get("description", ""))

    @classmethod
    def from_yaml(cls, path) -> "Portfolio":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_weights(cls, weights: Dict[str, float], nav: float, name: str = "Benchmark"):
        return cls(name=name, positions=[Equity(id=s, symbol=s, market_value=w * nav)
                                         for s, w in weights.items() if w])

    def benchmark_portfolio(self, ctx) -> Optional["Portfolio"]:
        if not self.benchmark:
            return None
        tot = sum(self.benchmark.values())
        w = {k: v / tot for k, v in self.benchmark.items()}
        return Portfolio.from_weights(w, self.nav(ctx), name=f"{self.name} benchmark")

    # ----- aggregation ----------------------------------------------------------------
    @property
    def ids(self) -> List[str]:
        return [p.id for p in self.positions]

    def values(self, ctx) -> pd.Series:
        return pd.Series([p.value(ctx) for p in self.positions], index=self.ids)

    def nav(self, ctx) -> float:
        return float(self.values(ctx).sum())

    def exposure_matrix(self, ctx) -> pd.DataFrame:
        """positions x factors, $ per unit factor move."""
        return pd.DataFrame([p.exposures(ctx) for p in self.positions], index=self.ids,
                            columns=ctx.factor_ids)

    def exposures(self, ctx) -> pd.Series:
        return self.exposure_matrix(ctx).sum()

    def specific_terms(self, ctx) -> pd.DataFrame:
        rows = []
        for p in self.positions:
            for key, dollars, uvol in p.specific(ctx):
                rows.append({"position": p.id, "key": key, "dollars": dollars, "unit_vol": uvol})
        return pd.DataFrame(rows, columns=["position", "key", "dollars", "unit_vol"])

    def specific_keys(self, ctx) -> pd.Series:
        """key -> unit vol (daily) of each distinct specific risk source."""
        t = self.specific_terms(ctx)
        return t.groupby("key")["unit_vol"].first() if len(t) else pd.Series(dtype=float)

    def revalue(self, ctx, F: np.ndarray, Z: Dict[str, np.ndarray]) -> pd.DataFrame:
        """(n scenarios x positions) full-revaluation P&L."""
        return pd.DataFrame({p.id: p.revalue(ctx, F, Z) for p in self.positions})

    def holdings_table(self, ctx) -> pd.DataFrame:
        v = self.values(ctx)
        rows = []
        for p in self.positions:
            d = p.describe(ctx)
            d["market_value"] = v[p.id]
            rows.append(d)
        df = pd.DataFrame(rows).set_index("id")
        nav = v.sum()
        df["weight_%"] = 100 * df["market_value"] / nav
        return df
