"""Lantern dashboard: `streamlit run app/streamlit_app.py` (or `lantern app`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lantern import Lantern, Portfolio                                   # noqa: E402
from lantern.factors.build import from_display, to_display, unit_label   # noqa: E402
from lantern.factors.covariance import correlation                       # noqa: E402
from lantern.reporting import figures as fg                              # noqa: E402
from lantern.scenarios.library import HISTORICAL, summary_table          # noqa: E402

st.set_page_config(page_title="Lantern", page_icon="🏮", layout="wide")
PLOT = {"width": "stretch", "config": {"displaylogo": False}}


@st.cache_resource(show_spinner="Loading market data and fitting the factor model…")
def build(portfolio_yaml: str, synthetic: bool, method: str, vol_hl: int, corr_hl: int):
    import yaml
    pf = Portfolio.from_dict(yaml.safe_load(portfolio_yaml))
    lt = Lantern(pf, synthetic=synthetic, exposure_method=method, vol_halflife=vol_hl,
                 corr_halflife=corr_hl)
    lt.risk()                                    # fit exposures up front
    return lt


@st.cache_data(show_spinner="Downloading the factor zoo…")
def zoo_data():
    from lantern.data import french
    return french.factor_zoo(), french.us_factors()[["Mkt-RF", "SMB", "HML"]]


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🏮 Lantern")
st.sidebar.caption("Cross-asset factor risk · scenarios · systematic factor timing")
default_yaml = (ROOT / "examples" / "multi_asset.yaml").read_text()
up = st.sidebar.file_uploader("Portfolio (YAML)", type=["yaml", "yml"])
portfolio_yaml = up.read().decode() if up else default_yaml
synthetic = st.sidebar.toggle("Synthetic market (offline)", value=False,
                              help="Use the built-in synthetic market instead of Yahoo/French data.")
with st.sidebar.expander("Risk model settings"):
    method = st.selectbox("Exposure estimator", ["adaptive_lasso", "ridge", "ols"])
    vol_hl = st.slider("Volatility half-life (days)", 10, 252, 63)
    corr_hl = st.slider("Correlation half-life (days)", 63, 1260, 252)
regime = st.sidebar.radio("Covariance regime", ["current", "stress"], horizontal=True,
                          help="'stress' = equal-weighted covariance over 2008-09 and Feb-Apr 2020.")

try:
    lt = build(portfolio_yaml, synthetic, method, vol_hl, corr_hl)
except Exception as e:                                         # network or data problems
    st.error(f"Could not build the platform: {e}. Try the synthetic market.")
    st.stop()

pf, ctx = lt.portfolio, lt.ctx
nav = pf.nav(ctx)
st.sidebar.caption(f"As of {ctx.asof:%d %b %Y} · {lt.market.source} data · "
                   f"{len(ctx.factor_ids)} factors")

tabs = st.tabs(["Overview", "Risk", "VaR & simulation", "Scenarios", "Bubble monitor",
                "Factor lab", "Optimizer", "Factor model"])

# ----------------------------------------------------------------------------- overview
with tabs[0]:
    st.header(pf.name)
    st.caption(pf.description)
    risk = lt.risk(regime)
    mc = lt.var("student_t", 0.99, 1, 20000, regime=regime)
    c = st.columns(5)
    c[0].metric("NAV", f"${nav / 1e6:,.2f}m")
    c[1].metric("Ex-ante vol (ann.)", f"{100 * risk.total_vol:.2f}%")
    if pf.benchmark:
        c[2].metric("Tracking error", f"{100 * lt.active_risk(regime).total_vol:.2f}%")
    c[3].metric("1d 99% VaR", f"{mc['VaR_%']:.2f}%")
    c[4].metric("1d 99% ES", f"{mc['ES_%']:.2f}%")
    hold = lt.holdings()
    l, r = st.columns([3, 2])
    l.dataframe(hold, width="stretch")
    ac = hold.groupby("asset_class")["market_value"].sum() / nav * 100
    r.plotly_chart(fg.hbar(ac, "Allocation by asset class (% of NAV)", fmt=".1f"), **PLOT)

# ----------------------------------------------------------------------------- risk
with tabs[1]:
    risk = lt.risk(regime)
    st.subheader("Ex-ante volatility decomposition")
    st.json({k: round(float(v), 3) for k, v in risk.summary().items()}, expanded=False)
    a, b = st.columns(2)
    a.plotly_chart(fg.hbar(risk.by_group["contribution_%"], "By factor group (% of NAV)",
                           polarity=True), **PLOT)
    b.plotly_chart(fg.hbar(risk.by_factor["contribution_%"], "Top factors", polarity=True, top=12), **PLOT)
    st.plotly_chart(fg.hbar(risk.by_position["contribution_%"], "By position (% of NAV)",
                            polarity=True), **PLOT)
    st.dataframe(risk.by_factor.sort_values("contribution_%", ascending=False), width="stretch")
    if pf.benchmark:
        te = lt.active_risk(regime)
        st.subheader(f"Active risk vs {pf.benchmark}")
        st.plotly_chart(fg.hbar(te.by_group["contribution_%"], "Tracking error by group",
                                polarity=True), **PLOT)
    ex = pf.exposures(ctx)
    a, b = st.columns(2)
    a.plotly_chart(fg.exposure_bars(ex, ctx.specs, nav, ("return",)), **PLOT)
    b.plotly_chart(fg.exposure_bars(ex, ctx.specs, nav, ("rate", "spread")), **PLOT)

# ----------------------------------------------------------------------------- VaR
with tabs[2]:
    c = st.columns(4)
    sim = c[0].selectbox("Simulator", ["student_t", "gaussian", "fhs", "regime", "historical"],
                         format_func={"student_t": "Student-t", "gaussian": "Gaussian",
                                      "fhs": "Filtered historical", "regime": "Calm/stress mixture",
                                      "historical": "Historical simulation"}.get)
    alpha = c[1].select_slider("Confidence", [0.95, 0.975, 0.99, 0.995], value=0.99)
    horizon = c[2].select_slider("Horizon (days)", [1, 5, 10, 21], value=1)
    n = c[3].select_slider("Paths", [5000, 20000, 50000], value=20000)
    kw = {"p_stress": st.slider("Stress probability (mixture)", 0.0, 0.5, 0.1)} if sim == "regime" else {}
    res = lt.var(sim, alpha, horizon, n, regime=regime, **kw)
    m = st.columns(2)
    m[0].metric(f"VaR {alpha:.1%}", f"${res['VaR']:,.0f}", f"{res['VaR_%']:.2f}% of NAV",
                delta_color="off")
    m[1].metric(f"ES {alpha:.1%}", f"${res['ES']:,.0f}", f"{res['ES_%']:.2f}% of NAV",
                delta_color="off")
    st.plotly_chart(fg.pnl_histogram(res["pnl"], res["VaR"], res["ES"], nav,
                                     f"{horizon}-day P&L distribution", alpha), **PLOT)
    st.dataframe(res["contributions"], width="stretch")
    st.subheader("All methods")
    st.dataframe(lt.var_comparison(alpha, horizon, n).round(2), width="stretch")
    st.subheader("VaR backtest (historical simulation, 500-day window)")
    bt = lt.backtest(alpha)
    st.write(f"{bt.exceptions} exceptions in {bt.n} days (expected {bt.expected:.1f}) · "
             f"Kupiec p = {bt.kupiec_p:.3f} · Christoffersen p = {bt.christoffersen_p:.3f} · "
             f"Basel zone (last 250d): **{bt.traffic_light}**")
    st.plotly_chart(fg.backtest_chart(bt.series, nav), **PLOT)
    st.subheader("12-month risk cone")
    drift = st.slider("Expected annual return (%)", -5.0, 10.0, 0.0) / 100
    st.plotly_chart(fg.risk_cone(lt.risk_cone(12, 2000, drift), nav), **PLOT)

# ----------------------------------------------------------------------------- scenarios
with tabs[3]:
    st.subheader("Build a market-driven scenario")
    st.caption("Set a few shocks; every other factor moves by its conditional expectation "
               "under the selected covariance regime.")
    pick = st.multiselect("Shocked factors", ctx.factor_ids, default=["EQ.US", "UST.10Y"])
    horizon = st.select_slider("Scenario horizon (days)", [5, 10, 21, 63], value=21, key="sh")
    shocks = {}
    cols = st.columns(max(len(pick), 1))
    for i, f in enumerate(pick):
        k = ctx.specs[f].kind
        default = {"return": -10.0, "rate": 50.0, "spread": 100.0, "vol": 10.0}[k]
        v = cols[i].number_input(f"{f} ({unit_label(k)})", value=default, step=1.0, key=f"s_{f}")
        shocks[f] = from_display(v, k)
    sec = st.multiselect("Security-level shocks (log return, %)",
                         sorted({getattr(p, "symbol", None) for p in pf.positions} - {None}))
    sshocks = {s: np.log1p(st.number_input(f"{s} move (%)", value=-20.0, key=f"q_{s}") / 100)
               for s in sec}
    if shocks or sshocks:
        r = lt.scenario(shocks, "Custom", horizon, sshocks, regime)
        m = st.columns(3)
        m[0].metric("Scenario P&L", f"${r.pnl:,.0f}", f"{r.pnl_pct:+.2f}% of NAV", delta_color="off")
        m[1].metric("95% prediction error", f"± ${r.pred_error_95:,.0f}")
        m[2].metric("Shock size", f"{r.mahalanobis:.1f} σ",
                    f"1 in {r.return_period_years:,.0f} yrs" if np.isfinite(r.return_period_years)
                    else "beyond model support", delta_color="off")
        t = r.table(ctx.specs)
        a, b = st.columns(2)
        a.plotly_chart(fg.hbar(r.pnl_by_position / nav * 100, "P&L by position (% of NAV)",
                               polarity=True), **PLOT)
        b.dataframe(t.style.format({"move": "{:+.2f}"}), width="stretch", height=520)

    st.subheader("Historical replays")
    H = summary_table(lt.historical_scenarios(regime))
    H["±95% %"] = 100 * H["±95% $"] / nav
    st.plotly_chart(fg.scenario_bars(H, "Historical replays"), **PLOT)
    st.dataframe(H.drop(columns=["1-in-N years"]).round(2), width="stretch")
    st.subheader("Hypothetical library")
    sev = st.slider("Severity multiplier", 0.25, 2.0, 1.0, 0.25)
    Y = summary_table(lt.hypothetical_scenarios(regime, sev))
    Y["±95% %"] = 100 * Y["±95% $"] / nav
    st.plotly_chart(fg.scenario_bars(Y, "Hypothetical scenarios"), **PLOT)
    st.dataframe(Y.round(2), width="stretch")

# ----------------------------------------------------------------------------- bubbles
with tabs[4]:
    st.subheader("Real-time explosiveness monitor")
    st.caption("BSADF on month-end prices (latest month uses today's close), 36-month minimum "
               "window, 95% Monte Carlo critical values. Flagged targets generate burst scenarios.")
    flags = lt.bubble_scan()
    if len(flags):
        st.dataframe(flags.drop(columns=["runup_log"]).sort_values("E", ascending=False)
                     .style.format({"E": "{:.2f}", "cv": "{:.2f}", "runup_%": "{:.0f}"}),
                     width="stretch")
        target = st.selectbox("Inspect", flags.sort_values("E", ascending=False).index)
        from lantern.systematic.explosive import current_state
        lv = (ctx.levels[target].resample("ME").last() if flags.loc[target, "type"] == "factor"
              else lt.market.monthly([target])[target].loc[:ctx.asof]).dropna()
        s = current_state(lv)
        if "E_series" in s:
            st.plotly_chart(fg.explosiveness_chart(s["E_series"], s["cv_series"], f"BSADF: {target}"), **PLOT)
        B = lt.bubble_scenarios(flags)
        if B:
            T = summary_table(B)
            T["±95% %"] = 100 * T["±95% $"] / nav
            st.plotly_chart(fg.scenario_bars(T, "Bubble-burst scenarios (stressed covariance)"), **PLOT)
            st.dataframe(T.round(2), width="stretch")
        else:
            st.info("No explosive holdings or factors right now.")
    from lantern.systematic.common import systemic_explosiveness
    eq = [s for s in ("SPY", "EFA", "EEM", "XLK", "XLF", "XLE") if s in lt.market.prices]
    if len(eq) >= 3:
        sysx = systemic_explosiveness(lt.market.monthly(eq).loc[:ctx.asof].dropna())
        st.subheader("Systemic (common-factor) explosiveness")
        st.caption("Real-time PC1 (CCIPCA) of " + ", ".join(eq) + " log prices, then BSADF.")
        st.plotly_chart(fg.explosiveness_chart(sysx["E_series"], sysx["cv_series"], "BSADF of PC1"), **PLOT)

# ----------------------------------------------------------------------------- factor lab
with tabs[5]:
    st.subheader("Timing the factor zoo: explosiveness (TSFE) vs momentum (TSFM)")
    from lantern.analytics.performance import stats_table
    from lantern.systematic import signals
    try:
        zoo, ff3 = zoo_data()
    except Exception as e:
        st.warning(f"French data library unavailable: {e}")
        zoo = None
    if zoo is not None:
        c = st.columns(3)
        regions = c[0].multiselect("Regions", sorted({x.split(".")[0] for x in zoo.columns}),
                                   default=sorted({x.split(".")[0] for x in zoo.columns}))
        swin = c[1].slider("Minimum window (months)", 24, 60, 36)
        cost = c[2].slider("Round-trip cost (bp)", 0, 50, 10)
        Z = zoo[[x for x in zoo.columns if x.split(".")[0] in regions]]
        R = signals.run_all(Z, swin=swin, cost_bps=cost)
        st.plotly_chart(fg.cumulative_lines(R, f"{Z.shape[1]} factors, monthly"), **PLOT)
        st.dataframe(stats_table(R, factors=ff3).round(2), width="stretch")
        I, E, CV = signals.tsfe_signal(Z, swin)
        st.caption(f"Share of factors explosive in the latest month: {I.iloc[-1].mean():.0%}")
        st.plotly_chart(fg.hbar(100 * I.mean().sort_values(), "Share of months explosive (%)",
                                fmt=".0f"), **PLOT)

# ----------------------------------------------------------------------------- optimizer
with tabs[6]:
    from lantern.systematic import optimizer as opt
    st.subheader("Factor workbench: target exposures with minimum tracking error")
    universe = st.multiselect("Investable ETFs", [s for s in ["SPY", "EFA", "EEM", "IWM", "VLUE", "MTUM",
                              "QUAL", "USMV", "XLK", "XLF", "XLE", "IEF", "TLT", "LQD", "HYG", "TIP",
                              "GLD", "USO", "DBB"] if s in lt.market.prices],
                              default=["SPY", "EFA", "EEM", "MTUM", "VLUE", "QUAL", "USMV", "IEF", "LQD", "GLD"])
    hi = st.slider("Max weight", 0.1, 1.0, 0.4)
    tg = st.multiselect("Targeted factors", ctx.factor_ids, default=["EQ.US", "STYLE.MOM", "UST.10Y"])
    targets = {}
    cols = st.columns(max(len(tg), 1))
    for i, f in enumerate(tg):
        targets[f] = cols[i].number_input(f"{f} exposure per $1", value={"EQ.US": 0.5, "STYLE.MOM": 0.1,
                                          "UST.10Y": -2.0}.get(f, 0.0), step=0.05, key=f"t_{f}")
    if universe and targets:
        out = opt.factor_targeting(ctx, universe, lt.cov, targets, hi=hi)
        m = st.columns(2)
        m[0].metric("Tracking error to equal weight", f"{100 * out['tracking_error_ann']:.2f}%")
        m[1].metric("Portfolio vol", f"{100 * out['vol_ann']:.2f}%")
        a, b = st.columns(2)
        a.plotly_chart(fg.hbar(100 * out["weights"], "Weights (%)", fmt=".1f"), **PLOT)
        b.dataframe(pd.DataFrame({"target": out["targets"], "achieved": out["achieved"]}).round(3))
    st.subheader("Risk-based construction")
    meth = st.radio("Method", ["risk_parity", "min_variance", "max_diversification"], horizontal=True)
    if universe:
        w = opt.construct(ctx, universe, lt.cov, meth, hi=hi)
        Sigma, _ = opt.asset_covariance(ctx, universe, lt.cov)
        a, b = st.columns(2)
        a.plotly_chart(fg.hbar(100 * w, "Weights (%)", fmt=".1f"), **PLOT)
        b.plotly_chart(fg.hbar(pd.Series(100 * opt.risk_contributions(w.values, Sigma), index=universe),
                               "Risk contributions (%)", fmt=".1f"), **PLOT)

# ----------------------------------------------------------------------------- model
with tabs[7]:
    st.subheader("Factor universe")
    from lantern.factors.build import spec_frame
    from lantern.factors.covariance import annualized_vol
    sf = spec_frame(ctx.universe)
    sf["ann. vol"] = annualized_vol(lt.covariance(regime)).reindex(sf.index)
    st.dataframe(sf, width="stretch")
    st.plotly_chart(fg.correlation_heatmap(correlation(lt.covariance(regime))), **PLOT)
    sym = st.selectbox("Exposure fit", sorted({getattr(p, "symbol", None) for p in pf.positions} - {None}))
    f = ctx.fit(sym)
    st.write(f"{f.method} · R² = {f.r2:.2f} · specific vol = {100 * f.specific_vol * np.sqrt(252):.1f}% ann. "
             f"· {f.n_obs} obs")
    st.plotly_chart(fg.hbar(f.beta[f.beta.abs() > 1e-8], f"{sym} factor betas", unit="",
                            polarity=True, fmt=".3f"), **PLOT)

st.sidebar.markdown("---")
st.sidebar.caption("Independent open-source project, not affiliated with BlackRock. Not investment advice.")
