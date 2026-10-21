"""Plotly figures shared by the Streamlit app and the static HTML report.

Colors: fixed categorical order (never cycled), one hue for single-series magnitude,
blue <-> red with a gray midpoint for polarity (gains/losses, correlations).
Solid hairline grids, thin marks, a legend whenever there are >= 2 series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
GAIN, LOSS, MID = "#2a78d6", "#e34948", "#f0efec"
CRITICAL = "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"


def _layout(fig: go.Figure, title: str = "", height: int = 380, **kw) -> go.Figure:
    opts = dict(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height, margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color=INK2),
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        bargap=0.25)
    opts.update(kw)
    fig.update_layout(**opts)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="", automargin=True)
    return fig


def hbar(values: pd.Series, title: str, unit: str = "%", polarity: bool = False,
         height: int = None, fmt: str = ".2f", top: int = None) -> go.Figure:
    """Horizontal bars, largest at top. polarity=True colors by sign (gain/loss)."""
    s = values.dropna()
    if top:
        s = s.reindex(s.abs().sort_values(ascending=False).index[:top])
    s = s.sort_values()
    colors = [GAIN if v >= 0 else LOSS for v in s] if polarity else SERIES[0]
    fig = go.Figure(go.Bar(
        x=s.values, y=[str(i) for i in s.index], orientation="h", marker_color=colors,
        marker_line_width=0, text=[f"{v:{fmt}}{unit}" for v in s.values], textposition="outside",
        cliponaxis=False, hovertemplate=f"<b>%{{x:{fmt}}}{unit}</b><br>%{{y}}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK2, line_width=1)
    fig.update_xaxes(range=_padded_range(s.values))
    return _layout(fig, title, height or max(260, 26 * len(s) + 80), showlegend=False)


def _padded_range(v, lo_extra=None, hi_extra=None):
    """Axis range with room for outside labels on both sides of zero."""
    lo = min(0.0, float(np.min(v if lo_extra is None else lo_extra)))
    hi = max(0.0, float(np.max(v if hi_extra is None else hi_extra)))
    span = (hi - lo) or 1.0
    return [lo - (0.22 * span if lo < 0 else 0.0), hi + (0.18 * span if hi > 0 else 0.02 * span)]


def correlation_heatmap(corr: pd.DataFrame, title: str = "Factor correlations") -> go.Figure:
    scale = [[0, LOSS], [0.5, MID], [1, GAIN]]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=scale, xgap=1, ygap=1, colorbar=dict(thickness=10, outlinewidth=0),
        hovertemplate="<b>%{z:.2f}</b><br>%{y} vs %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return _layout(fig, title, 620)


def pnl_histogram(pnl: np.ndarray, var: float, es: float, nav: float, title: str,
                  alpha: float = 0.99) -> go.Figure:
    x = 100 * np.asarray(pnl) / nav
    fig = go.Figure(go.Histogram(x=x, nbinsx=120, marker_color=SERIES[0], marker_line_width=0,
                                 hovertemplate="P&L %{x:.2f}%<br><b>%{y}</b> scenarios<extra></extra>",
                                 name="Simulated P&L"))
    for v, lab, pos in ((var, f"VaR {alpha:.0%}", "top right"), (es, f"ES {alpha:.0%}", "top left")):
        fig.add_vline(x=-100 * v / nav, line_color=CRITICAL if lab.startswith("ES") else INK2,
                      line_width=2, annotation_text=f"{lab}: {100 * v / nav:.2f}%",
                      annotation_position=pos, annotation_font_color=INK)
    fig.update_xaxes(title="P&L, % of NAV")
    fig.update_yaxes(title="scenarios")
    return _layout(fig, title, 360, showlegend=False)


def scenario_bars(table: pd.DataFrame, title: str = "Scenario P&L") -> go.Figure:
    """table indexed by scenario with columns 'P&L %' and optional '±95% %'."""
    t = table.sort_values("P&L %")
    v = t["P&L %"].values
    err = t["±95% %"].values if "±95% %" in t else np.zeros(len(t))
    y = [str(i) for i in t.index]
    fig = go.Figure(go.Bar(
        x=v, y=y, orientation="h", marker_color=[GAIN if x >= 0 else LOSS for x in v],
        marker_line_width=0,
        error_x=dict(type="data", array=err, color=INK2, thickness=1, width=3),
        customdata=err, hovertemplate="<b>%{x:+.2f}% of NAV</b> ± %{customdata:.2f}<br>%{y}<extra></extra>"))
    end = v + np.sign(v) * err
    fig.add_trace(go.Scatter(x=end, y=y, mode="text", text=[f" {x:+.1f}% " for x in v],
                             textposition=["middle right" if x >= 0 else "middle left" for x in v],
                             textfont=dict(color=INK, size=11), hoverinfo="skip"))
    fig.add_vline(x=0, line_color=INK2, line_width=1)
    fig.update_xaxes(title="P&L, % of NAV (whiskers: 95% prediction error)",
                     range=_padded_range(v, np.minimum(end, 0), np.maximum(end, 0)))
    return _layout(fig, title, max(300, 30 * len(t) + 90), showlegend=False)


def risk_cone(paths: np.ndarray, nav: float, step_label: str = "month",
              title: str = "Monte Carlo risk cone") -> go.Figure:
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0) * nav / 1e6
    x = np.arange(paths.shape[1])
    fig = go.Figure()
    bands = (((0, 4), SEQ[1], "5th–95th pct"), ((1, 3), SEQ[3], "25th–75th pct"))
    for (lo, hi), col, name in bands:
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[hi], q[lo][::-1]], fill="toself",
                                 fillcolor=col, line=dict(width=0), name=name, hoverinfo="skip",
                                 opacity=0.55))
    fig.add_trace(go.Scatter(x=x, y=q[2], mode="lines", line=dict(color=SERIES[0], width=2), name="Median",
                             hovertemplate="<b>$%{y:.2f}m</b> median<extra></extra>"))
    for i, name in ((0, "5th pct"), (4, "95th pct")):
        fig.add_trace(go.Scatter(x=x, y=q[i], mode="lines", line=dict(color=SEQ[4], width=1), name=name,
                                 showlegend=False, hovertemplate=f"<b>$%{{y:.2f}}m</b> {name}<extra></extra>"))
    fig.update_xaxes(title=f"{step_label}s ahead")
    fig.update_yaxes(title="NAV, $m")
    return _layout(fig, title, 380, hovermode="x unified")


def backtest_chart(df: pd.DataFrame, nav: float, title: str = "VaR backtest") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=100 * df["pnl"] / nav, marker_color=SERIES[0],
                         marker_line_width=0, name="Daily P&L",
                         hovertemplate="<b>%{y:.2f}%</b> P&L<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=-100 * df["VaR"] / nav, line=dict(color=SERIES[1], width=2),
                             name="−VaR forecast", hovertemplate="<b>%{y:.2f}%</b> −VaR<extra></extra>"))
    ex = df[df["exception"]]
    fig.add_trace(go.Scatter(x=ex.index, y=100 * ex["pnl"] / nav, mode="markers",
                             marker=dict(color=CRITICAL, size=8, symbol="x", line=dict(width=0)),
                             name="✕ Exception", hovertemplate="<b>%{y:.2f}%</b> exception<extra></extra>"))
    fig.update_yaxes(title="% of NAV")
    return _layout(fig, title, 360, hovermode="x unified", bargap=0)


def explosiveness_chart(E: pd.Series, cv: pd.Series, title: str, price: pd.Series = None) -> go.Figure:
    fig = go.Figure()
    above = (E > cv)
    # shade explosive months
    runs, start = [], None
    for t, a in above.items():
        if a and start is None:
            start = t
        if not a and start is not None:
            runs.append((start, t)); start = None
    if start is not None:
        runs.append((start, E.index[-1]))
    for a, b in runs:
        fig.add_vrect(x0=a, x1=b, fillcolor=SEQ[0], opacity=0.6, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=E.index, y=E.values, line=dict(color=SERIES[0], width=2),
                             name="BSADF statistic", hovertemplate="<b>%{y:.2f}</b> BSADF<extra></extra>"))
    fig.add_trace(go.Scatter(x=cv.index, y=cv.values, line=dict(color=SERIES[1], width=2),
                             name="95% critical value", hovertemplate="<b>%{y:.2f}</b> cv<extra></extra>"))
    fig.update_yaxes(title="t-statistic")
    return _layout(fig, title + "  (shaded: explosive regime)", 340, hovermode="x unified")


def cumulative_lines(R: pd.DataFrame, title: str, log: bool = True) -> go.Figure:
    W = (1 + R.fillna(0)).cumprod()
    fig = go.Figure()
    for i, c in enumerate(W.columns):
        fig.add_trace(go.Scatter(x=W.index, y=W[c], name=c, mode="lines",
                                 line=dict(color=SERIES[i % 8], width=2),
                                 hovertemplate=f"<b>%{{y:.2f}}</b> {c}<extra></extra>"))
    # direct end labels, nudged apart so they never collide
    ends = np.log10(W.iloc[-1].clip(lower=1e-6)) if log else W.iloc[-1]
    span = float(np.log10(W.max().max() / W.min().min())) if log else float(W.max().max() - W.min().min())
    gap, placed = 0.045 * span, {}
    for c in ends.sort_values().index:
        y = ends[c]
        if placed and y - max(placed.values()) < gap:
            y = max(placed.values()) + gap
        placed[c] = y
    for c, y in placed.items():
        fig.add_annotation(x=W.index[-1], y=y, text=c, showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK2, size=11))
    if log:
        lo, hi = W.min().min(), W.max().max()
        ticks = [v for v in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo / 1.5 <= v <= hi * 1.5]
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{v:g}" for v in ticks])
    fig.update_yaxes(title="growth of $1")
    return _layout(fig, title, 420, hovermode="x unified", margin=dict(l=8, r=110, t=48, b=8))


def exposure_bars(exposures: pd.Series, specs: dict, nav: float, kinds=("return",),
                  title: str = "") -> go.Figure:
    """Return factors: beta in % of NAV per 1% factor move. Rate/spread factors: $ per bp
    (DV01 / CS01). One unit per chart."""
    keep = [f for f in exposures.index if specs[f].kind in kinds and abs(exposures[f]) > 1e-9]
    e = exposures[keep]
    if kinds == ("return",):
        return hbar(e / nav, title or "Factor betas (portfolio % per 1% factor move)",
                    unit="", polarity=True, fmt=".3f")
    return hbar(e * 1e-4, title or "Rate and spread sensitivities ($ per +1 bp)", unit="",
                polarity=True, fmt=",.0f")
