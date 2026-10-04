"""Plotly chart builders. One y-axis per chart, hover on every mark, thin marks, recessive grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui import INK, INK2, MUTED, GRID, SURFACE, BLUE, ORANGE, AQUA, YELLOW, VIOLET, TIER_COLOR, SEV_COLOR

CONFIG = {"displayModeBar": False, "responsive": True}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]   # unique categories
PILLAR_COLOR = {"sales": BLUE, "momentum": ORANGE, "financial": AQUA, "customer": YELLOW}   # fixed slot order
FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=12)


def _base(fig: go.Figure, height=320, title=None, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=40 if title else 28 if legend else 12, b=8), paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, font=FONT, hoverlabel=dict(bgcolor="white", font_size=12, font_color=INK),
        title_text=f"<b>{title}</b>" if title else "", title_font=dict(size=14, color=INK), title_x=0.01,
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor=GRID, tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor=GRID, tickfont=dict(color=MUTED), automargin=True)
    return fig


def empty(msg="No data for this selection", height=260):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=13), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _base(fig, height, legend=False)


# ---------------------------------------------------------------- attention board
def quadrant(df: pd.DataFrame, height=430) -> go.Figure:
    if df.empty:
        return empty(height=height)
    x = df.sales_growth_pct.clip(-60, 100)
    y = df.share_change_pp.clip(-15, 15)
    size = np.clip(np.sqrt(df.units_l12m.to_numpy(float)) / 2.2, 5, 34)
    fig = go.Figure()
    for tier in ["Strong", "Stable", "Watch", "Critical"]:
        m = (df.tier == tier).to_numpy()
        if not m.any():
            continue
        d = df[m]
        fig.add_trace(go.Scatter(
            x=x[m], y=y[m], mode="markers", name=tier,
            marker=dict(size=size[m], color=TIER_COLOR[tier], opacity=0.78, line=dict(color=SURFACE, width=1.5)),
            customdata=np.column_stack([d.territory, d.state_name, d.health_score.round(0), d.primary_issue,
                                        d.units_l12m.map("{:,.0f}".format), d.sales_growth_pct.round(1),
                                        d.share_change_pp.round(1)]),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Health %{customdata[2]} · %{customdata[3]}"
                          "<br>Sales %{customdata[5]}% YoY · Share %{customdata[6]} pp<br>%{customdata[4]} units / 12m"
                          "<extra></extra>"))
    lab = dict(font=dict(size=11, color=MUTED), showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(text="growing · gaining", x=0.99, y=0.99, xanchor="right", yanchor="top", **lab)
    fig.add_annotation(text="<b>growing · losing share</b>", x=0.99, y=0.01, xanchor="right", yanchor="bottom", **lab)
    fig.add_annotation(text="falling · losing", x=0.01, y=0.01, xanchor="left", yanchor="bottom", **lab)
    fig.add_annotation(text="falling · holding share", x=0.01, y=0.99, xanchor="left", yanchor="top", **lab)
    fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1)); fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    fig.update_xaxes(title=dict(text="Sales growth, last 12m vs prior 12m (%)", font=dict(size=11)), ticksuffix="%")
    fig.update_yaxes(title=dict(text="Change in market share (pp)", font=dict(size=11)))
    return _base(fig, height)


def driver_bars(counts: pd.DataFrame, height=300) -> go.Figure:
    """counts: columns title, pillar, n"""
    if counts.empty:
        return empty("No risk alerts in this scope", height)
    c = counts.sort_values("n")
    fig = go.Figure(go.Bar(
        x=c.n, y=c.title, orientation="h", marker=dict(color=[PILLAR_COLOR[p] for p in c.pillar], cornerradius=4),
        text=c.n, textposition="outside", textfont=dict(color=INK2, size=11), cliponaxis=False,
        hovertemplate="%{y}: %{x} dealers<extra></extra>"))
    fig.update_xaxes(showgrid=False, visible=False, range=[0, c.n.max() * 1.15 + 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2, size=12))
    return _base(fig, height, legend=False)


def tier_by_group(df: pd.DataFrame, group: str, height=320, min_n=5) -> go.Figure:
    """100%-stacked tier mix per group, sorted by share needing attention (groups with < min_n dealers dropped)."""
    if df.empty:
        return empty(height=height)
    t = pd.crosstab(df[group], df.tier).reindex(columns=["Critical", "Watch", "Stable", "Strong"], fill_value=0)
    t = t[t.sum(axis=1) >= min(min_n, int(t.sum(axis=1).max()))]
    n = t.sum(axis=1)
    pct = t.div(n, axis=0) * 100
    pct = pct.loc[pct[["Critical", "Watch"]].sum(axis=1).sort_values().index]
    labels = [f"{g}  ({n[g]})" for g in pct.index]
    fig = go.Figure()
    for tier in pct.columns:
        fig.add_trace(go.Bar(y=labels, x=pct[tier], name=tier, orientation="h",
                             marker=dict(color=TIER_COLOR[tier], line=dict(color=SURFACE, width=1.5)),
                             customdata=t.loc[pct.index, tier],
                             hovertemplate="%{y}<br>" + tier + ": %{x:.0f}% (%{customdata} dealers)<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK2))
    fig.update_xaxes(title=dict(text="% of dealer territories (count in brackets)", font=dict(size=11)),
                     ticksuffix="%", range=[0, 100])
    return _base(fig, max(height, 24 * len(pct) + 90))


# ---------------------------------------------------------------- deep dive
def pillar_bars(row: pd.Series, pillars: dict, sources: dict, height=210) -> go.Figure:
    keys = list(pillars)[::-1]
    vals = [row.get(f"pillar_{k}") for k in keys]
    vals = [v if v == v and v is not None else 0 for v in vals]
    labels = [f"{pillars[k]}  <span style='color:{MUTED};font-size:10px'>{sources[k].upper()}</span>" for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[100] * len(keys), y=labels, orientation="h", marker=dict(color="#efeee9", cornerradius=4),
                         hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=[PILLAR_COLOR[k] for k in keys], cornerradius=4),
                         text=[f"{v:.0f}" for v in vals], textposition="outside", textfont=dict(color=INK, size=12),
                         cliponaxis=False, hovertemplate="%{y}: %{x:.0f}/100<extra></extra>", showlegend=False))
    fig.update_layout(barmode="overlay", bargap=0.38)
    fig.update_xaxes(range=[0, 112], visible=False, showgrid=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=INK))
    return _base(fig, height, legend=False)


def sales_vs_target(m: pd.DataFrame, height=300) -> go.Figure:
    """m: date, oem_units, target_units"""
    if m.empty:
        return empty(height=height)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m.date, y=m.oem_units, name="Actual retail (VAHAN)", marker=dict(color=BLUE, cornerradius=3),
                         hovertemplate="%{x|%b %Y}<br>Actual: %{y:,.0f}<extra></extra>"))
    t = m.dropna(subset=["target_units"])
    fig.add_trace(go.Scatter(x=t.date, y=t.target_units, name="Target (modelled)", mode="lines+markers",
                             line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=5),
                             hovertemplate="%{x|%b %Y}<br>Target: %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(title=dict(text="Units / month", font=dict(size=11)))
    fig.update_layout(bargap=0.25)
    return _base(fig, height)


def share_trend(m: pd.DataFrame, expected: float, height=300) -> go.Figure:
    """m: date, share (3m rolling %), state_share (3m rolling %)"""
    if m.empty:
        return empty(height=height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.date, y=m.share, name="This territory", mode="lines", line=dict(color=BLUE, width=2.5),
                             hovertemplate="%{x|%b %Y}<br>Share: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=m.date, y=m.state_share, name="State average", mode="lines",
                             line=dict(color=MUTED, width=1.6), hovertemplate="%{x|%b %Y}<br>State: %{y:.1f}%<extra></extra>"))
    if expected == expected and expected is not None:
        fig.add_hline(y=expected, line=dict(color=ORANGE, width=1.6, dash="dash"),
                      annotation_text=f"Expected for this market type: {expected:.1f}%", annotation_position="top left",
                      annotation_font=dict(size=11, color=ORANGE))
    fig.update_yaxes(title=dict(text="Market share, 3-month rolling", font=dict(size=11)), ticksuffix="%")
    return _base(fig, height)


def competitor_shift(gains: pd.Series, height=300) -> go.Figure:
    if gains.empty:
        return empty(height=height)
    g = gains.sort_values()
    fig = go.Figure(go.Bar(
        x=g.values, y=g.index, orientation="h",
        marker=dict(color=[SEV_COLOR["high"] if v > 0 else "#9e9c95" for v in g.values], cornerradius=4),
        text=[f"{v:+.1f}" for v in g.values], textposition="outside", cliponaxis=False, textfont=dict(size=11, color=INK2),
        hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"))
    lim = max(abs(g.min()), abs(g.max()), 0.5) * 1.3
    fig.update_xaxes(range=[-lim, lim], title=dict(text="Change in competitor share, last 12m vs prior 12m (pp)", font=dict(size=11)))
    fig.update_yaxes(showgrid=False)
    return _base(fig, height, legend=False)


def ops_small_multiples(o: pd.DataFrame, med: pd.DataFrame, height=420) -> go.Figure:
    """o: dealer monthly ops; med: network monthly medians. 4 panels, each with its own axis."""
    if o.empty:
        return empty(height=height)
    spec = [("inventory_days", "Inventory cover (days)", 65), ("payment_delay_days", "Payment delay (days past due)", 18),
            ("csi_score", "Service CSI (/100)", 76), ("complaints_per_100", "Complaints per 100 units", 2.5)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in spec], horizontal_spacing=0.09, vertical_spacing=0.2)
    for i, (col, label, alert) in enumerate(spec):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=med.date, y=med[col], mode="lines", line=dict(color="#b4b2aa", width=1.5, dash="dot"),
                                 name="Network median", showlegend=i == 0,
                                 hovertemplate="%{x|%b %Y}<br>Network median: %{y:.1f}<extra></extra>"), row=r, col=c)
        fig.add_trace(go.Scatter(x=o.date, y=o[col], mode="lines+markers", line=dict(color=BLUE, width=2),
                                 marker=dict(size=5), name="This dealer", showlegend=i == 0,
                                 hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:.1f}<extra></extra>"), row=r, col=c)
        fig.add_hline(y=alert, line=dict(color=SEV_COLOR["high"], width=1, dash="dash"), row=r, col=c)
    fig.update_annotations(font=dict(size=12, color=INK))
    fig = _base(fig, height)
    fig.update_layout(margin=dict(t=64), legend=dict(y=1.08, yanchor="bottom"))
    return fig


def health_trend(h: pd.DataFrame, height=300) -> go.Figure:
    if h.empty:
        return empty(height=height)
    fig = go.Figure()
    bands = [(0, 45, "Critical"), (45, 60, "Watch"), (60, 75, "Stable"), (75, 100, "Strong")]
    for lo, hi, t in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=TIER_COLOR[t], opacity=0.07, line_width=0,
                      annotation_text=t, annotation_position="right", annotation_font=dict(size=10, color=MUTED))
    fig.add_trace(go.Scatter(x=h.as_of, y=h.health_score, mode="lines+markers", line=dict(color=INK, width=2.2),
                             marker=dict(size=7, color=[TIER_COLOR[t] for t in h.tier], line=dict(color=SURFACE, width=1.5)),
                             customdata=h.tier, hovertemplate="%{x|%b %Y}<br>Health %{y:.0f} · %{customdata}<extra></extra>",
                             showlegend=False))
    fig.update_yaxes(range=[0, 100], title=dict(text="Health score", font=dict(size=11)))
    fig.update_layout(margin=dict(r=60))
    return _base(fig, height, legend=False)


# ---------------------------------------------------------------- network insights
def brand_shift(d: pd.Series, height=360) -> go.Figure:
    d = d.sort_values()
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h",
                           marker=dict(color=[AQUA if v > 0 else SEV_COLOR["high"] for v in d.values], cornerradius=4),
                           text=[f"{v:+.1f}" for v in d.values], textposition="outside", cliponaxis=False,
                           textfont=dict(size=11, color=INK2), hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"))
    lim = max(abs(d.min()), abs(d.max())) * 1.25
    fig.update_xaxes(range=[-lim, lim], title=dict(text="Change in national car-market share (pp)", font=dict(size=11)))
    fig.update_yaxes(showgrid=False)
    return _base(fig, height, legend=False)


def seasonality(idx: pd.DataFrame, height=360) -> go.Figure:
    fig = go.Figure()
    colors = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300"]          # categorical slots 1-6 in order
    for (zone, row), c in zip(idx.iterrows(), colors):
        fig.add_trace(go.Scatter(x=MONTHS, y=row.reindex(range(1, 13)).values, name=zone, mode="lines+markers",
                                 line=dict(color=c, width=2), marker=dict(size=5),
                                 hovertemplate=zone + " · %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color="#c3c2b7", width=1))
    fig.update_yaxes(title=dict(text="Monthly sales index (avg month = 1)", font=dict(size=11)))
    return _base(fig, height)
