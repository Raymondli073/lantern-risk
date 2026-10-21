"""Explosiveness-triggered ("bubble-burst") scenarios.

Narrative scenarios rely on judgment for *which* shock to run and *how big*. Here the
real-time explosiveness test supplies both: every factor or holding whose month-end
BSADF statistic exceeds its critical value is flagged, the PSY origination date of
the live episode is located, and the scenario reverses a fraction of the log run-up
since origination (25% / 50% / 100% = good / bad / ugly). The shock is propagated to
all other factors with the conditional-expectation engine, by default under a stressed
covariance, because bubbles tend to burst in stressed, high-correlation markets.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..systematic.explosive import current_state
from .propagation import run_scenario


def scan(ctx, portfolio=None, swin: int = 36, level: float = 0.95,
         factor_kinds=("return",), min_months: int = 60) -> pd.DataFrame:
    """Explosiveness state of every return-type factor and every equity-like holding."""
    rows = []
    monthly_levels = ctx.levels.resample("ME").last()
    monthly_levels.index = monthly_levels.index[:-1].append(pd.DatetimeIndex([ctx.levels.index[-1]]))
    for f in ctx.universe:
        if f.kind not in factor_kinds:
            continue
        s = monthly_levels[f.id].dropna()
        if len(s) < max(min_months, swin + 2):
            continue
        st = current_state(s, swin, level)
        rows.append({"target": f.id, "type": "factor", "group": f.group, **_row(st)})
    if portfolio is not None:
        # securities that *are* a factor (SPY = EQ.US, BTC-USD = CRYPTO.BTC) are covered above
        seen = {f.recipe.split(":", 1)[1] for f in ctx.universe if f.recipe.startswith("ret:")}
        for p in portfolio.positions:
            sym = getattr(p, "symbol", None)
            if not sym or sym in seen or sym not in ctx.market.prices:
                continue
            seen.add(sym)
            s = ctx.market.monthly([sym])[sym].loc[:ctx.asof].dropna()
            if len(s) < max(min_months, swin + 2):
                continue
            st = current_state(s, swin, level)
            rows.append({"target": sym, "type": "security", "group": p.asset_class, **_row(st)})
    return pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()


def _row(st):
    return {"E": st["E"], "cv": st["cv"], "explosive": st["explosive"], "signal": st["signal"],
            "origination": st.get("origination"), "runup_%": 100 * np.expm1(st.get("runup_log", 0.0)),
            "runup_log": st.get("runup_log", 0.0)}


def burst_scenarios(portfolio, ctx, cov: pd.DataFrame, flags: pd.DataFrame,
                    retracements=(0.25, 0.5, 1.0), horizon_days: int = 21,
                    combined: bool = True) -> List:
    """One scenario per flagged target and retracement, plus a joint burst."""
    hit = flags[flags["explosive"] & (flags["runup_log"] > 0)] if len(flags) else flags
    out = []
    labels = {0.25: "good", 0.5: "bad", 1.0: "ugly"}
    for target, row in hit.iterrows():
        for a in retracements:
            shock = -a * row["runup_log"]
            fs, ss = ({target: shock}, None) if row["type"] == "factor" else (None, {target: shock})
            out.append(run_scenario(portfolio, ctx, cov, fs or {}, security_shocks=ss,
                                    name=f"{target} burst: {int(100 * a)}% retracement "
                                         f"({labels.get(a, '')})",
                                    horizon_days=horizon_days,
                                    description=f"Explosive since {row['origination']:%b %Y}; "
                                                f"run-up {row['runup_%']:.0f}%."))
    if combined and len(hit) > 1:
        fs = {t: -0.5 * r["runup_log"] for t, r in hit.iterrows() if r["type"] == "factor"}
        ss = {t: -0.5 * r["runup_log"] for t, r in hit.iterrows() if r["type"] == "security"}
        out.append(run_scenario(portfolio, ctx, cov, fs, security_shocks=ss,
                                name="Joint burst of all explosive targets (50%)",
                                horizon_days=horizon_days))
    return out
