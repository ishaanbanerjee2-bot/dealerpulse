"""Visual system: CSS, colour tokens and small HTML components (all user text is escaped)."""
from __future__ import annotations

import html
import re
import streamlit as st

# colour tokens (dataviz reference palette, light mode)
INK, INK2, MUTED, GRID, SURFACE, PLANE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#f9f9f7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
TIER_COLOR = {"Critical": "#d03b3b", "Watch": "#ec835a", "Stable": "#9e9c95", "Strong": "#0ca30c"}
TIER_ICON = {"Critical": "🔴", "Watch": "🟠", "Stable": "⚪", "Strong": "🟢"}
SEV_COLOR = {"high": "#d03b3b", "medium": "#ec835a", "info": "#9e9c95", "opportunity": "#0ca30c"}
SEV_LABEL = {"high": "High", "medium": "Medium", "info": "Context", "opportunity": "Opportunity"}
ACTION_COLOR = {"Corrective": "#d03b3b", "Support": BLUE, "Grow": "#0b7a0b", "Maintain": "#6b6a66"}

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; --surface:{SURFACE}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1560px; }}
[data-testid="stSidebar"] {{ background: #f3f2ee; border-right: 1px solid {GRID}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.app-title {{ font-size: 1.55rem; font-weight: 700; color: {INK}; margin: 0; line-height: 1.2; }}
.app-sub {{ color: {INK2}; font-size: 0.92rem; margin-top: 2px; }}
.brand {{ display:flex; align-items:center; gap:10px; margin: 0 0 6px 0; }}
.brand-mark {{ width:34px; height:34px; border-radius:9px; background:{BLUE}; color:white; display:flex;
  align-items:center; justify-content:center; font-weight:800; font-size:16px; }}
.brand-name {{ font-weight:800; font-size:1.15rem; color:{INK}; line-height:1.05; }}
.brand-tag {{ font-size:0.74rem; color:{INK2}; }}
.section-label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
  color:{MUTED}; margin: 0.6rem 0 0.2rem 0; }}
.card {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:12px; padding:14px 16px; }}
.hero {{ display:flex; gap:22px; align-items:stretch; }}
.hero-main {{ flex: 1 1 auto; }}
.hero-name {{ font-size:1.35rem; font-weight:700; color:{INK}; margin:0; }}
.hero-meta {{ color:{INK2}; font-size:0.86rem; margin:2px 0 8px 0; }}
.hero-summary {{ color:{INK}; font-size:0.95rem; line-height:1.45; }}
.hero-score {{ flex: 0 0 150px; text-align:center; border-left:1px solid {GRID}; padding-left:18px; }}
.score-num {{ font-size:3.0rem; font-weight:800; line-height:1; color:{INK}; }}
.score-cap {{ font-size:0.75rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; }}
.delta-up {{ color:#006300; font-weight:700; font-size:0.9rem; }}
.delta-down {{ color:#b42727; font-weight:700; font-size:0.9rem; }}
.delta-flat {{ color:{INK2}; font-weight:600; font-size:0.9rem; }}
.chip {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.74rem; font-weight:700;
  border:1px solid currentColor; margin-right:6px; white-space:nowrap; }}
.tier-chip {{ display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:999px;
  font-size:0.8rem; font-weight:700; color:{INK}; background:#ffffff; border:1px solid rgba(11,11,11,0.14); }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.src {{ display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.06em; padding:1px 6px;
  border-radius:4px; margin-left:6px; vertical-align:middle; }}
.src-real {{ background:#e3eefb; color:#1c5cab; }}
.src-syn {{ background:#efedf9; color:#4a3aa7; }}
.src-mod {{ background:#f6efe2; color:#8a5a12; }}
.src-own {{ background:#e2f4ec; color:#0b6b48; }}
.alert {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.08); border-left:5px solid; border-radius:10px;
  padding:10px 12px; margin-bottom:8px; }}
.alert-title {{ font-weight:700; font-size:0.92rem; color:{INK}; }}
.alert-ev {{ font-size:0.86rem; color:{INK2}; margin-top:3px; line-height:1.4; }}
.alert-sev {{ font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; }}
.action {{ background:#ffffff; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:10px 12px; margin-bottom:8px; }}
.action-text {{ font-size:0.9rem; color:{INK}; line-height:1.4; margin-top:5px; }}
.action-meta {{ font-size:0.76rem; color:{MUTED}; margin-top:5px; }}
.flow {{ display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; }}
.flow-step {{ flex:1 1 160px; background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:10px 12px; }}
.flow-step b {{ color:{BLUE}; }}
.flow-arrow {{ align-self:center; color:{MUTED}; font-size:1.2rem; }}
.small-note {{ font-size:0.78rem; color:{MUTED}; }}
.legend-row {{ font-size:0.78rem; color:{INK2}; line-height:1.7; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; }}
div[data-testid="stMetricValue"] {{ font-size: 1.65rem; }}
div[data-testid="stMetricLabel"] p {{ font-size: 0.76rem; color:{INK2}; }}
div[data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}
div[data-testid="stMetric"] > div:first-child {{ padding: 13px 10px; }}
[data-testid="stExpander"] details {{ border-radius: 10px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def md_bold_to_html(text: str) -> str:
    """Escape, then turn **bold** markdown into <b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))


def src_badge(source: str) -> str:
    cls = {"Real": "src-real", "Modelled": "src-mod", "Your data": "src-own"}.get(source, "src-syn")
    return f'<span class="src {cls}">{esc(source).upper()}</span>'


def tier_chip(tier: str) -> str:
    c = TIER_COLOR.get(tier, "#9e9c95")
    return f'<span class="tier-chip"><span class="dot" style="background:{c}"></span>{esc(tier)}</span>'


def action_chip(kind: str) -> str:
    return f'<span class="chip" style="color:{ACTION_COLOR.get(kind, INK2)}">{esc(kind)}</span>'


def brand_block():
    st.markdown(
        '<div class="brand"><div class="brand-mark">DP</div><div><div class="brand-name">DealerPulse</div>'
        '<div class="brand-tag">Dealer attention navigator</div></div></div>', unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{esc(text)}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    st.markdown(f'<p class="app-title">{esc(title)}</p><p class="app-sub">{esc(subtitle)}</p>', unsafe_allow_html=True)


def delta_html(d) -> str:
    if isinstance(d, str):
        return ""
    if d is None or d != d:
        return '<span class="delta-flat">no history</span>'
    if d >= 1:
        return f'<span class="delta-up">▲ {d:+.0f} vs 6 months ago</span>'
    if d <= -1:
        return f'<span class="delta-down">▼ {d:+.0f} vs 6 months ago</span>'
    return '<span class="delta-flat">● unchanged vs 6 months ago</span>'


def hero_card(name, meta, tier, issue, action_type, score, delta, summary):
    st.markdown(f"""
<div class="card hero">
  <div class="hero-main">
    <p class="hero-name">{esc(name)}</p>
    <div class="hero-meta">{esc(meta)}</div>
    <div style="margin-bottom:8px">{tier_chip(tier)}&nbsp;&nbsp;<span class="chip" style="color:{INK2}">{esc(issue)}</span>{action_chip(action_type)}</div>
    <div class="hero-summary">{md_bold_to_html(summary)}</div>
  </div>
  <div class="hero-score">
    <div class="score-cap">Health score</div>
    <div class="score-num">{score:.0f}</div>
    <div class="score-cap" style="margin-bottom:6px">out of 100</div>
    {delta_html(delta)}
  </div>
</div>""", unsafe_allow_html=True)


def alert_card(a: dict, source: str):
    c = SEV_COLOR[a["severity"]]
    st.markdown(f"""
<div class="alert" style="border-left-color:{c}">
  <span class="alert-sev" style="color:{c}">{SEV_LABEL[a["severity"]]}</span>{src_badge(source)}
  <div class="alert-title">{esc(a["title"])}</div>
  <div class="alert-ev">{esc(a["evidence"])}</div>
</div>""", unsafe_allow_html=True)


def action_card(a: dict):
    st.markdown(f"""
<div class="action">{action_chip(a["action_type"])}<span class="small-note">{esc(a["title"])}</span>
  <div class="action-text">{esc(a["action"])}</div>
  <div class="action-meta">Owner: {esc(a["owner"])} · Within {esc(a["horizon"])}</div>
</div>""", unsafe_allow_html=True)


def flow(steps: list[tuple[str, str]]):
    parts = []
    for i, (t, d) in enumerate(steps):
        parts.append(f'<div class="flow-step"><b>{i + 1}. {esc(t)}</b><div class="alert-ev">{esc(d)}</div></div>')
        if i < len(steps) - 1:
            parts.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)
