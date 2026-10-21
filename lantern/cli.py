"""Command line: `lantern report | risk | zoo | app`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "multi_asset.yaml"


def _lantern(args):
    from . import Lantern, Portfolio
    return Lantern(Portfolio.from_yaml(args.portfolio), synthetic=args.synthetic,
                   refresh=getattr(args, "refresh", False))


def _zoo(args):
    from .data import french
    if getattr(args, "jkp", None):
        from .data.jkp import load_jkp
        zoo = load_jkp(args.jkp)
    else:
        zoo = french.factor_zoo()
    ff3 = french.us_factors()[["Mkt-RF", "SMB", "HML"]]
    return zoo, ff3


def cmd_report(args):
    from .reporting.html import build_report
    lt = _lantern(args)
    zoo = ff3 = None
    if not args.no_zoo and not args.synthetic:
        try:
            zoo, ff3 = _zoo(args)
        except Exception as e:                      # the library may be unreachable
            print(f"factor zoo skipped: {e}", file=sys.stderr)
    p = build_report(lt, args.out, zoo, ff3)
    print(f"wrote {p}")


def cmd_risk(args):
    from .scenarios.library import summary_table
    lt = _lantern(args)
    pd.set_option("display.width", 160)
    r = lt.risk()
    print(pd.Series(r.summary()).round(3).to_string(), "\n")
    print(r.by_group.round(2).to_string(), "\n")
    print(lt.var_comparison().round(2).to_string(), "\n")
    print(summary_table(lt.historical_scenarios()).round(2).to_string(), "\n")
    flags = lt.bubble_scan()
    if len(flags):
        print(flags[flags["explosive"]][["type", "E", "cv", "origination", "runup_%"]].to_string())


def cmd_zoo(args):
    from .analytics.performance import stats_table
    from .systematic import signals
    zoo, ff3 = _zoo(args)
    R = signals.run_all(zoo, swin=args.swin, cost_bps=args.cost)
    pd.set_option("display.width", 160)
    print(stats_table(R, factors=ff3).round(2).to_string())


def cmd_app(args):
    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lantern", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("report", cmd_report), ("risk", cmd_risk)):
        p = sub.add_parser(name)
        p.add_argument("--portfolio", default=str(EXAMPLE))
        p.add_argument("--synthetic", action="store_true", help="offline synthetic market")
        p.add_argument("--refresh", action="store_true", help="re-download prices")
        if name == "report":
            p.add_argument("--out", default="reports/lantern_report.html")
            p.add_argument("--no-zoo", action="store_true")
            p.add_argument("--jkp", help="JKP factor CSV (long format) instead of French")
        p.set_defaults(fn=fn)
    p = sub.add_parser("zoo")
    p.add_argument("--jkp", help="path to a JKP factor CSV (jkpfactors.com, long format)")
    p.add_argument("--swin", type=int, default=36)
    p.add_argument("--cost", type=float, default=10.0, help="round-trip cost, bp")
    p.set_defaults(fn=cmd_zoo)
    sub.add_parser("app").set_defaults(fn=cmd_app)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
