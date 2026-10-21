"""End-to-end tour of Lantern. Run with `python examples/quickstart.py [--synthetic]`."""
import sys
from pathlib import Path

import pandas as pd

from lantern import Lantern, Portfolio
from lantern.scenarios.library import bp, pct, summary_table

pd.set_option("display.width", 160)
synthetic = "--synthetic" in sys.argv
lt = Lantern(Portfolio.from_yaml(Path(__file__).with_name("multi_asset.yaml")), synthetic=synthetic)

print("== Ex-ante risk")
r = lt.risk()
print(pd.Series(r.summary()).round(3).to_string())
print(r.by_group.round(2).to_string())

print("\n== VaR / ES (1 day, 99%) by method")
print(lt.var_comparison().round(2).to_string())

print("\n== A market-driven scenario: equities -15%, 10y -40 bp over a month")
s = lt.scenario({"EQ.US": pct(-0.15), "UST.10Y": bp(-40)}, horizon_days=21)
print(f"P&L {s.pnl:,.0f} ({s.pnl_pct:+.2f}% of NAV) ± {s.pred_error_95:,.0f}; "
      f"shock size {s.mahalanobis:.1f} sigma")
print(s.table(lt.ctx.specs).round(2).head(12).to_string())

print("\n== Historical replays")
print(summary_table(lt.historical_scenarios())[["P&L %", "±95% $"]].round(2).to_string())

print("\n== Explosiveness monitor")
flags = lt.bubble_scan()
print(flags[flags["explosive"]][["type", "E", "cv", "origination", "runup_%"]].round({"E": 2, "cv": 2, "runup_%": 1}).to_string())
