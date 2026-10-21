"""A self-contained HTML risk report (one file; Plotly from a CDN)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline

from . import figures as fg
from ..factors.covariance import annualized_vol, correlation
from ..scenarios.library import summary_table

CSS = """
:root{--surface:#fcfcfb;--surface-2:#f4f3f0;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#7a7974;
--rule:#e7e6e2;--accent:#2a78d6;--critical:#d03b3b;--good:#0ca30c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--surface-2:#232321;--ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#9a9990;--rule:#33332f;--accent:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--surface-2:#232321;--ink:#ffffff;--ink-2:#c3c2b7;
--ink-3:#9a9990;--rule:#33332f;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 Inter,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 4px;padding-top:16px;border-top:1px solid var(--rule)}
h3{font-size:15px;margin:20px 0 6px}
p,li{color:var(--ink-2)}.muted{color:var(--ink-3);font-size:12.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface-2);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--ink-3)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:860px){.grid2{grid-template-columns:1fr}}
.tbl{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink-3);font-weight:600}
.fig{min-height:120px}
.flag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.flag.on{background:color-mix(in srgb,var(--critical) 16%,transparent);color:var(--critical)}
.flag.off{background:var(--surface-2);color:var(--ink-3)}
footer{margin-top:48px;font-size:12px;color:var(--ink-3)}
"""

JS = """
(function(){
const L2D={"#2a78d6":"#3987e5","#eb6834":"#d95926","#1baf7a":"#199e70","#eda100":"#c98500",
"#e87ba4":"#d55181","#4a3aa7":"#9085e9","#e34948":"#e66767","#fcfcfb":"#1a1a19","#0b0b0b":"#ffffff",
"#52514e":"#c3c2b7","#e7e6e2":"#33332f","#f0efec":"#383835","white":"#262624","#cde2fb":"#0d366b",
"#9ec5f4":"#184f95","#6da7ec":"#1c5cab","#3987e5":"#5598e7","#256abf":"#86b6ef","#184f95":"#9ec5f4"};
const D2L=Object.fromEntries(Object.entries(L2D).map(([a,b])=>[b,a]));
function walk(o,m){if(typeof o==="string"){return m[o.toLowerCase()]||o}
 if(Array.isArray(o))return o.map(x=>walk(x,m));
 if(o&&typeof o==="object"){const r={};for(const k in o)r[k]=walk(o[k],m);return r}return o}
let dark=null;
function apply(){const want=document.documentElement.dataset.theme?document.documentElement.dataset.theme==="dark":
 matchMedia("(prefers-color-scheme: dark)").matches;if(want===dark)return;const m=want?L2D:D2L;
 document.querySelectorAll(".js-plotly-plot").forEach(d=>{Plotly.react(d,walk(d.data,m),walk(d.layout,m))});dark=want}
window.addEventListener("load",()=>{dark=false;apply()});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",apply);
new MutationObserver(apply).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
})();
"""


def _fig(fig) -> str:
    return '<div class="fig">' + fig.to_html(full_html=False, include_plotlyjs=False,
                                              config={"displaylogo": False, "responsive": True}) + "</div>"


def _years(v):
    if pd.isna(v):
        return ""
    return "> 10,000" if v > 1e4 else f"{v:,.1f}"


def _table(df: pd.DataFrame, fmt: dict = None, index=True, max_rows: int = 60) -> str:
    df = df.head(max_rows).copy()
    fmt = fmt or {}
    for c in df.columns:
        f = fmt.get(c)
        if callable(f):
            df[c] = df[c].map(f)
        elif f:
            df[c] = df[c].map(lambda v, f=f: "" if pd.isna(v) else format(v, f))
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
    df.index.name = None
    df.columns = [str(c).replace("_", " ") for c in df.columns]
    return '<div class="tbl">' + df.to_html(index=index, border=0, escape=True, na_rep="") + "</div>"


def _kpi(label, value):
    return f'<div class="kpi"><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div></div>'


def build_report(lt, out: str = "lantern_report.html", factor_zoo: pd.DataFrame = None,
                 ff3: pd.DataFrame = None, n_sims: int = 20000) -> Path:
    """Run the full analysis for a Lantern object and write one HTML file."""
    pf, ctx, specs = lt.portfolio, lt.ctx, lt.ctx.specs
    nav = pf.nav(ctx)
    risk = lt.risk()
    te = lt.active_risk() if pf.benchmark else None
    mc = lt.var("student_t", 0.99, 1, n_sims)
    cmp_ = lt.var_comparison(0.99, 1, n_sims)
    bt = lt.backtest()
    hist = lt.historical_scenarios()
    hyp = lt.hypothetical_scenarios()
    flags = lt.bubble_scan()
    bursts = lt.bubble_scenarios(flags)
    cone = lt.risk_cone(12, 2000)
    H, Y = summary_table(hist), summary_table(hyp)
    B = summary_table(bursts) if bursts else pd.DataFrame()
    for T in (H, Y, B):
        if len(T):
            T["±95% %"] = 100 * T["±95% $"] / nav

    worst = H["P&L %"].idxmin() if len(H) else "n/a"
    n_flag = int(flags["explosive"].sum()) if len(flags) else 0
    parts = [f"<h1>{html.escape(pf.name)}</h1>",
             f'<p class="muted">As of {ctx.asof:%d %b %Y} · data: {html.escape(lt.market.source)} · '
             f'{len(ctx.factor_ids)} risk factors · generated {datetime.now():%d %b %Y %H:%M} by '
             f'Lantern</p>', f"<p>{html.escape(pf.description)}</p>", '<div class="kpis">',
             _kpi("NAV", f"${nav / 1e6:,.2f}m"),
             _kpi("Ex-ante volatility (ann.)", f"{100 * risk.total_vol:.2f}%"),
             _kpi("Tracking error vs benchmark", f"{100 * te.total_vol:.2f}%" if te else "n/a"),
             _kpi("1-day 99% VaR (MC, Student-t)", f"{mc['VaR_%']:.2f}%"),
             _kpi("1-day 99% Expected Shortfall", f"{mc['ES_%']:.2f}%"),
             _kpi(f"Worst replay: {worst}"[:60], f"{H['P&L %'].min():+.1f}%" if len(H) else "n/a"),
             _kpi("Explosive holdings / factors", str(n_flag)), "</div>"]

    hold = lt.holdings()
    cols = [c for c in ["asset_class", "symbol", "market_value", "weight_%", "duration",
                        "yield_pct", "strike", "expiry", "delta"] if c in hold]
    parts += ["<h2>Holdings</h2>", _table(hold[cols], {"market_value": ",.0f", "weight_%": ".1f"})]

    bf = risk.by_factor.copy()
    scale = {"return": 0.01, "rate": 1e-4, "spread": 1e-4, "vol": 0.01}
    bf["sensitivity_$"] = [e * scale[k] for e, k in zip(bf["exposure_$"], bf["kind"])]
    bf["per"] = bf["kind"].map({"return": "+1%", "rate": "+1 bp", "spread": "+1 bp", "vol": "+1 vol pt"})
    parts += ["<h2>Ex-ante risk decomposition</h2>",
              "<p>Euler contributions to annualized volatility (they add up to the total). "
              "Negative bars are hedges.</p>", '<div class="grid2">',
              _fig(fg.hbar(risk.by_group["contribution_%"], "Risk by factor group (% of NAV, ann.)", polarity=True)),
              _fig(fg.hbar(bf["contribution_%"], "Top factor contributions", top=12, polarity=True)),
              "</div>",
              _fig(fg.hbar(risk.by_position["contribution_%"], "Risk contribution by position (% of NAV)",
                           polarity=True)),
              _table(bf.sort_values("contribution_%", ascending=False)
                     [["group", "sensitivity_$", "per", "stand_alone_%", "contribution_%",
                       "share_of_risk_%"]], {"sensitivity_$": ",.0f"})]
    if te:
        parts += ["<h3>Active risk vs benchmark " + html.escape(str(pf.benchmark)) + "</h3>",
                  _fig(fg.hbar(te.by_group["contribution_%"], "Tracking error by factor group", polarity=True))]

    ex = pf.exposures(ctx)
    parts += ["<h2>Factor exposures</h2>", '<div class="grid2">',
              _fig(fg.exposure_bars(ex, specs, nav, ("return",))),
              _fig(fg.exposure_bars(ex, specs, nav, ("rate", "spread"))), "</div>"]

    parts += ["<h2>Value-at-Risk and Expected Shortfall</h2>",
              "<p>Monte Carlo with full revaluation (bonds and options are repriced in every path), "
              "compared across simulators. Stressed VaR uses the 2008–09 and 2020 covariance.</p>",
              _table(cmp_, {"VaR $": ",.0f", "ES $": ",.0f", "VaR %": ".2f", "ES %": ".2f"}),
              _fig(fg.pnl_histogram(mc["pnl"], mc["VaR"], mc["ES"], nav,
                                    "1-day P&L distribution (Student-t factor simulation)")),
              "<h3>Expected-shortfall contributions</h3>",
              _table(mc["contributions"], {"ES_contribution": ",.0f", "stand_alone_ES": ",.0f",
                                           "share_%": ".1f"}),
              "<h3>Backtest of historical-simulation VaR</h3>",
              f"<p>{bt.exceptions} exceptions in {bt.n} days (expected {bt.expected:.1f}); "
              f"Kupiec p = {bt.kupiec_p:.3f}, Christoffersen p = {bt.christoffersen_p:.3f}; "
              f"Basel zone over the last 250 days: <b>{bt.traffic_light}</b>.</p>",
              _fig(fg.backtest_chart(bt.series, nav))]

    fmt = {"P&L $": ",.0f", "P&L %": "+.2f", "±95% $": ",.0f", "±95% %": ".2f",
           "shock size (σ)": ".1f", "1-in-N years": _years, "horizon (d)": ".0f"}
    parts += ["<h2>Scenarios</h2>",
              "<p>Market-driven scenarios: explicit shocks on a few variables, every other factor "
              "implied by its conditional expectation; the bar whiskers are the 95% prediction "
              "error. Historical replays fill factors that did not exist yet the same way.</p>",
              _fig(fg.scenario_bars(H, "Historical replays")),
              _table(H.drop(columns=["1-in-N years"]), fmt),
              _fig(fg.scenario_bars(Y, "Hypothetical scenario library")), _table(Y, fmt)]

    parts += ["<h2>Explosiveness monitor and bubble-burst scenarios</h2>",
              "<p>Real-time BSADF statistic on month-end prices (expanding sample, 36-month minimum "
              "window, 95% Monte Carlo critical values). Flagged targets generate burst scenarios that "
              "retrace 25/50/100% of the log run-up since the PSY origination date, propagated "
              "under the stressed covariance.</p>"]
    if len(flags):
        F = flags.copy()
        F["state"] = np.where(F["explosive"], "explosive", "")
        F["origination"] = F["origination"].map(lambda d: "" if pd.isna(d) else f"{d:%b %Y}")
        parts.append(_table(F[["type", "group", "E", "cv", "state", "origination", "runup_%"]]
                            .sort_values("E", ascending=False), {"runup_%": ".0f"}))
        top = flags["E"].idxmax()
        st = _state_for(lt, top, flags.loc[top, "type"])
        if st is not None:
            parts.append(_fig(fg.explosiveness_chart(st["E_series"], st["cv_series"],
                                                     f"BSADF: {top}")))
    if len(B):
        parts += [_fig(fg.scenario_bars(B, "Bubble-burst scenarios")), _table(B, fmt)]

    parts += ["<h2>Monte Carlo risk cone (12 months)</h2>",
              "<p>2,000 Student-t factor paths, monthly steps, zero expected return: a picture of "
              "risk, not a forecast.</p>", _fig(fg.risk_cone(cone, nav))]

    corr = correlation(lt.cov.dropna(how="all").dropna(axis=1, how="all"))
    vols = annualized_vol(lt.cov)
    parts += ["<h2>Factor model</h2>", _fig(fg.correlation_heatmap(corr)),
              _table(pd.DataFrame({"group": [specs[f].group for f in vols.index],
                                   "kind": [specs[f].kind for f in vols.index],
                                   "ann. vol (natural units)": vols.values,
                                   "recipe": [specs[f].recipe for f in vols.index]},
                                  index=vols.index), {"ann. vol (natural units)": ".4f"})]

    if factor_zoo is not None:
        from ..systematic import signals
        from ..analytics.performance import stats_table
        R = signals.run_all(factor_zoo, cost_bps=10)
        S = stats_table(R, factors=ff3)
        parts += ["<h2>Systematic factor timing: explosiveness vs momentum</h2>",
                  f"<p>{factor_zoo.shape[1]} long-short factors from the Kenneth French library, monthly, "
                  "net of 10 bp round-trip costs. TSFE goes long factors whose BSADF statistic is above "
                  "its critical value with a positive last return; TSFM goes long factors with a positive "
                  "trailing 12-month return.</p>",
                  _fig(fg.cumulative_lines(R, "Cumulative returns")), _table(S)]

    parts.append('<footer>Lantern is an independent open-source project, modelled on the publicly '
                 'described design of institutional risk platforms. It is not affiliated with or '
                 'endorsed by BlackRock. Not investment advice.</footer>')
    v = plotly.offline.get_plotlyjs_version()
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Lantern Risk Report</title><style>{CSS}</style>'
           f'<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{v}/plotly.min.js"></script>'
           f'</head><body><main>{"".join(parts)}</main><script>{JS}</script></body></html>')
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def _state_for(lt, target, kind):
    from ..systematic.explosive import current_state
    if kind == "factor":
        lv = lt.ctx.levels[target].resample("ME").last().dropna()
    else:
        lv = lt.market.monthly([target])[target].loc[:lt.ctx.asof].dropna()
    st = current_state(lv)
    return st if "E_series" in st else None
