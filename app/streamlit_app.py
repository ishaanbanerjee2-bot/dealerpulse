"""
DealerPulse - which dealers need management attention, why, and what to do.
Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import siblings however the app is launched
import charts as C  # noqa: E402
import engine as E
import ingest as I
import ui as U

st.set_page_config(page_title="DealerPulse · Dealer Attention Navigator", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
U.inject_css()

DATA = Path(__file__).resolve().parents[1] / "data" / "app"
VIEWS = ["Attention Board", "Dealer Deep-Dive", "Quick Assess", "Network Insights", "Method & Data"]
VIEW_ICON = {"Attention Board": ":material/target:", "Dealer Deep-Dive": ":material/search_insights:",
             "Quick Assess": ":material/edit_note:", "Network Insights": ":material/insights:",
             "Method & Data": ":material/menu_book:"}
COMPETITORS = ["Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW", "Renault"]
ISSUES = ["Sales slump → stock stress", "Financial stress", "Losing to competition", "Below target", "Slowing momentum", "Service & CX issue",
          "Market-driven dip", "Untapped potential", "On track"]


# =============================================================================== data
@st.cache_data(show_spinner=False)
def load_sample() -> dict:
    hist = pd.read_parquet(DATA / "dealer_inputs_history.parquet")
    hist["as_of"] = pd.to_datetime(hist.as_of)
    return {"mode": "sample", "key": "sample", "hist": hist,
            "ops": pd.read_parquet(DATA / "ops_monthly.parquet"),
            "panel": pd.read_parquet(DATA / "panel_recent.parquet"),
            "targets": pd.read_parquet(DATA / "targets_monthly.parquet"),
            "nat": pd.read_parquet(DATA / "national_brand_monthly.parquet"),
            "validation": json.loads((DATA / "validation.json").read_text()),
            "competitors": COMPETITORS, "meta": None}


@st.cache_data(show_spinner="Reading your dealer file…", max_entries=4)
def load_upload(name: str, data: bytes) -> dict:
    import hashlib
    ds = I.build_dataset(name, data)
    ds.update(mode="upload", key=hashlib.sha1(data).hexdigest(), nat=None, validation=None,
              competitors=ds["meta"]["competitors"])
    return ds


@st.cache_data(show_spinner="Scoring the dealer network…", max_entries=8)
def score_all(ds_key: str, weights_items: tuple, _hist: pd.DataFrame) -> pd.DataFrame:
    w = dict(weights_items)
    parts = [E.score_dealers(g, w, volume_ref=g.units_l12m) for _, g in _hist.groupby("as_of")]
    return pd.concat(parts, ignore_index=True)


# per-run context: where each pillar's data comes from, and wording that depends on the data source
CTX = {"mode": "sample", "sources": dict(E.PILLAR_SOURCE)}


def src(pillar_or_kind: str) -> str:
    if CTX["mode"] == "upload":
        return "Your data"
    return CTX["sources"].get(pillar_or_kind, pillar_or_kind)


@st.cache_data(show_spinner=False)
def network_ops_median(ops: pd.DataFrame) -> pd.DataFrame:
    o = ops.copy()
    o["complaints_per_100"] = o.complaints / o.retail_units.clip(lower=1) * 100
    return o.groupby("date")[["inventory_days", "payment_delay_days", "csi_score", "complaints_per_100"]].median().reset_index()


@st.cache_data(show_spinner=False)
def state_share_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    s = panel.groupby(["state_name", "date"])[["oem_units", "car_market"]].sum().reset_index()
    s = s.sort_values("date")
    g = s.groupby("state_name")
    s["state_share"] = (g.oem_units.transform(lambda x: x.rolling(3, min_periods=1).sum()) /
                        g.car_market.transform(lambda x: x.rolling(3, min_periods=1).sum()) * 100)
    return s[["state_name", "date", "state_share"]]


def scope_frame(allsc: pd.DataFrame, as_of, zone, state) -> pd.DataFrame:
    cur = allsc[allsc.as_of == as_of].copy()
    prev = allsc[allsc.as_of == as_of - pd.DateOffset(months=6)][["dealer_id", "health_score", "tier"]]
    cur = cur.merge(prev.rename(columns={"health_score": "health_6m", "tier": "tier_6m"}), on="dealer_id", how="left")
    cur["newly_flagged"] = cur.tier.isin(["Critical", "Watch"]) & cur.tier_6m.isin(["Stable", "Strong"])
    cur["delta_6m"] = cur.health_score - cur.health_6m
    if zone != "All India":
        cur = cur[cur.zone == zone]
    if state != "All states":
        cur = cur[cur.state_name == state]
    cur = cur.sort_values(["priority_index", "dealer_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    cur["scope_rank"] = np.arange(1, len(cur) + 1)
    return cur


def show(fig, key):
    st.plotly_chart(fig, theme=None, config=C.CONFIG, key=key)


def fmt_delta(x, fmt, suffix=""):
    """Metric delta text, or None when the input is missing (no misleading '+nan')."""
    return None if x is None or not np.isfinite(x) else format(x, fmt) + suffix


def fmt_num(x, nd=0, suffix=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{nd}f}{suffix}"


def kpi(col, label, value, sub, source=None):
    badge = U.src_badge(source) if source else ""
    col.markdown(f"""<div class="card" style="height:100%">
<div class="small-note" style="font-weight:600;color:{U.INK2}">{U.esc(label)}{badge}</div>
<div style="font-size:1.75rem;font-weight:800;color:{U.INK};line-height:1.25;margin-top:2px">{U.esc(value)}</div>
<div class="small-note" style="margin-top:2px">{sub}</div></div>""", unsafe_allow_html=True)


# =============================================================================== state helpers
def _init_state():
    st.session_state.setdefault("nav", VIEWS[0])
    st.session_state.setdefault("_last_nav", VIEWS[0])
    st.session_state.setdefault("board_nonce", 0)
    for p, v in E.DEFAULT_WEIGHTS.items():
        st.session_state.setdefault(f"w_{p}", v)


def _nav_guard():
    if st.session_state.get("nav") is None:          # clicking the active segment de-selects it
        st.session_state.nav = st.session_state.get("_last_nav", VIEWS[0])


def _src_guard():
    if st.session_state.get("data_source") is None:
        st.session_state.data_source = st.session_state.get("_last_source", "Sample network")


def _store_upload():
    f = st.session_state.get("byod_uploader")
    if f is not None:
        st.session_state.byod_file = (f.name, f.getvalue())


def _clear_upload():
    st.session_state.pop("byod_file", None)


def _reset_weights():
    for p, v in E.DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{p}"] = v


def _open_from_board():
    key = f"board_table_{st.session_state.board_nonce}"
    rows = st.session_state.get(key, {}).get("selection", {}).get("rows", [])
    ids = st.session_state.get("_board_ids", [])
    if rows and rows[0] < len(ids):
        st.session_state.dealer_id = ids[rows[0]]
        st.session_state.nav = "Dealer Deep-Dive"
        st.session_state.board_nonce += 1            # fresh table next time -> same row can be re-opened


def _step_dealer(options, step):
    cur = st.session_state.get("dealer_id")
    i = options.index(cur) if cur in options else 0
    st.session_state.dealer_id = options[(i + step) % len(options)]


# =============================================================================== sidebar
def data_sidebar():
    """Data-source switch. Returns (dataset or None, error message or None)."""
    with st.sidebar:
        U.brand_block()
        U.section_label("Data")
        st.session_state.setdefault("data_source", "Sample network")
        st.segmented_control("Data source", ["Sample network", "Your data"], key="data_source",
                             on_change=_src_guard, label_visibility="collapsed")
        source = st.session_state.get("data_source") or "Sample network"
        st.session_state._last_source = source
        if source == "Sample network":
            return load_sample(), None
        stored = st.session_state.get("byod_file")
        if stored is None:
            st.file_uploader("Monthly dealer file (CSV or Excel)", type=["csv", "xlsx"], key="byod_uploader",
                             on_change=_store_upload, help="One row per dealer per month · at least 24 months")
            return None, None
        name, data = stored
        try:
            ds = load_upload(name, data)
        except E.InputError as e:
            st.button("Remove file", on_click=_clear_upload, width="stretch", icon=":material/close:")
            return None, f"{name}: {e}"
        except Exception as e:                       # unexpected parser failure: never crash the app
            st.button("Remove file", on_click=_clear_upload, width="stretch", icon=":material/close:")
            return None, f"{name}: could not read the file ({type(e).__name__}). Please start from the template."
        mt = ds["meta"]
        st.markdown(f'<div class="legend-row">{U.src_badge("Your data")} <b>{U.esc(name)}</b><br>'
                    f'{mt["dealers"]:,} dealers · {mt["months"]} months ({mt["first"]:%b %Y} – {mt["last"]:%b %Y})</div>',
                    unsafe_allow_html=True)
        for w in mt["warnings"]:
            st.caption("⚠️ " + w)
        st.button("Remove file", on_click=_clear_upload, width="stretch", icon=":material/close:")
        return ds, None


def sidebar(ds):
    hist = ds["hist"]
    with st.sidebar:
        U.section_label("Scope")
        zones = sorted(hist.zone.unique())
        if st.session_state.get("zone") not in ["All India"] + zones:
            st.session_state.zone = "All India"
        zone = st.selectbox("Zone", ["All India"] + zones, key="zone",
                            help="Regional managers usually work one zone; 'All India' shows the full network.")
        states = sorted(hist.loc[hist.zone == zone, "state_name"].unique()) if zone != "All India" \
            else sorted(hist.state_name.unique())
        if st.session_state.get("state") not in ["All states"] + states:
            st.session_state.state = "All states"
        state = st.selectbox("State", ["All states"] + states, key="state")
        months = [pd.Timestamp(m) for m in sorted(hist.as_of.unique())][-6:]
        if st.session_state.get("as_of") not in months:
            st.session_state.as_of = months[-1]
        if len(months) > 1:
            as_of = st.select_slider("Data as of", options=months, key="as_of",
                                     format_func=lambda d: d.strftime("%b %Y"),
                                     help="Rewind to see how concerns emerged. Each snapshot uses the 12 months to that date.")
        else:
            as_of = months[0]
            st.caption(f"Data as of {as_of:%b %Y} (add more months of history to rewind).")
        U.section_label("Focus (list & charts)")
        tiers = st.pills("Tier", E.TIERS, selection_mode="multi", default=E.TIERS, key="tiers",
                         format_func=lambda t: f"{U.TIER_ICON[t]} {t}")
        if not tiers:
            st.caption("No tier selected — showing all tiers.")
        issues = st.multiselect("Main issue", ISSUES, key="issues", placeholder="All issues")
        with st.expander("Scoring weights", icon=":material/tune:"):
            for p, label in E.PILLARS.items():
                st.slider(f"{label} · {src(p).lower()}", 0, 60, step=5, key=f"w_{p}")
            raw = {p: st.session_state[f"w_{p}"] for p in E.PILLARS}
            if sum(raw.values()) == 0:
                st.warning("All weights are zero — using the default weights.")
            nw = E.normalise_weights(raw)
            st.caption("Effective: " + " · ".join(f"{E.PILLARS[p].split(' ')[0]} {v:.0%}" for p, v in nw.items()))
            st.button("Reset to default", on_click=_reset_weights, width="stretch", icon=":material/restart_alt:")
        st.markdown("---")
        if ds["mode"] == "sample":
            st.markdown(
                f'<div class="legend-row">{U.src_badge("Real")} VAHAN registrations (MoRTH), Jan-2019 – May-2024<br>'
                f'{U.src_badge("Synthetic")} inventory, payments, service, complaints (modelled)<br>'
                f'{U.src_badge("Modelled")} targets: prior-year sales × market growth</div>', unsafe_allow_html=True)
        else:
            mt = ds["meta"]
            missing = [c for c in I.M_OPS if c not in mt["ops_columns"]]
            st.markdown(
                f'<div class="legend-row">{U.src_badge("Your data")} every number comes from your file<br>'
                f'Share benchmark: {U.esc(mt["benchmark"])}<br>'
                + (f'Not in your file (skipped, score re-weighted): {U.esc(", ".join(missing))}' if missing else
                   'All operating metrics supplied') + '</div>', unsafe_allow_html=True)
    weights = {p: st.session_state[f"w_{p}"] for p in E.PILLARS}
    return zone, state, as_of, (tiers or E.TIERS), issues, weights


def view_byod_landing(error):
    U.page_header("Bring your own dealer data",
                  "Upload your monthly dealer file — every view then runs on your own network, with the same scoring engine")
    if error:
        st.error(f"Upload rejected — {error}", icon=":material/error:")
    U.flow([
        ("Download the template", "Or the 150-dealer sample file to try the flow right now"),
        ("Fill one row per dealer per month", "At least 24 months; units, target and local market are required"),
        ("Upload in the sidebar", "CSV or Excel; checked row by row with clear messages"),
        ("Use every view", "Attention board, deep-dives, trends, what-if and insights on your network"),
    ])
    st.write("")
    a, b, _ = st.columns([2, 2, 3])
    a.download_button("Download template (CSV)", I.template_monthly().to_csv(index=False).encode("utf-8"),
                      file_name="dealerpulse_monthly_template.csv", mime="text/csv", width="stretch",
                      icon=":material/download:")
    b.download_button("Sample file · 150 dealers", (DATA / "sample_monthly_upload.csv").read_bytes(),
                      file_name="dealerpulse_sample_150_dealers.csv", mime="text/csv", width="stretch",
                      icon=":material/dataset:")
    st.markdown("**File layout** <span class='small-note'>column names are case-insensitive; extra columns are ignored</span>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ("dealer_id", "Required", "Your dealer code", "—"),
        ("month", "Required", "YYYY-MM (also 2024-05-01, May-2024, 202405)", "—"),
        ("units_sold", "Required", "Retail units in the month", "Sales, growth, share"),
        ("target_units", "Required", "Monthly sales target", "Target achievement"),
        ("local_car_market", "Required", "All-brand car sales in the dealer's market (e.g. VAHAN for its RTOs)", "Market share & potential"),
        ("dealer_name, state, zone", "Optional", "Names and regions", "Readable names, zone/state filters"),
        ("expected_share_pct", "Optional", "Your share benchmark for the dealer's market", "Fair benchmark (else state/zone average)"),
        ("inventory_days, aged_stock_pct, payment_delay_days", "Optional", "Stock cover, % stock > 60 days, avg days past due", "Inventory & payments pillar"),
        ("csi_score, complaints", "Optional", "Service CSI (0–100), complaint count in the month", "Service & complaints pillar"),
        ("comp_<Brand>", "Optional", "Competitor units in the dealer's market, e.g. comp_Hyundai", "Competition view & conquest actions"),
    ], columns=["Column", "Need", "Meaning", "Unlocks"]), hide_index=True, column_config={
        "Column": st.column_config.TextColumn(width=255), "Need": st.column_config.TextColumn(width=75),
        "Meaning": st.column_config.TextColumn(width=390), "Unlocks": st.column_config.TextColumn(width=235)})
    st.caption("Missing optional columns are skipped and the health score is re-weighted over what you supply. "
               "For a one-off check of a few dealers without history, use Quick Assess in the sample network.")


# =============================================================================== views
def view_board(df, scope_label, as_of):
    U.page_header("Which dealers need attention?",
                  f"{scope_label} · {len(df):,} dealer territories · data as of {as_of:%b %Y}")
    if df.empty:
        st.info("No dealer territories in this scope.")
        return
    n_c, n_w = int((df.tier == "Critical").sum()), int((df.tier == "Watch").sum())
    has_prev = df.health_6m.notna().any()
    new_att = int(df.newly_flagged.sum())
    share_now = df.units_l12m.sum() / df.car_market_l12m.sum() * 100
    share_prev = df.units_p12m.sum() / df.car_market_p12m.sum() * 100
    stress = df.alerts.apply(lambda fl: any(f["code"] in ("INVENTORY", "PAYMENT") for f in fl)).sum()
    k = st.columns(5, gap="small")
    kpi(k[0], "Need attention now", f"{n_c + n_w:,}",
        f"🔴 {n_c} critical · 🟠 {n_w} watch · of {len(df):,}")
    kpi(k[1], "Newly needing attention", f"{new_att:,}" if has_prev else "—",
        "were Stable/Strong 6 months ago — emerging concerns" if has_prev else "needs 6 more months of history")
    kpi(k[2], "Revenue gap vs target", f"₹{df.revenue_gap_cr.sum():,.0f} cr",
        f"{df.shortfall_units.sum():,.0f} units below target · 12m", src("Modelled"))
    kpi(k[3], "Market share · 12m", f"{share_now:.1f}%",
        f"{share_now - share_prev:+.1f} pp vs prior 12 months", src("Real"))
    med_stock = df.inventory_days.median()
    kpi(k[4], "Stock / credit stress", f"{int(stress):,}" if df.inventory_days.notna().any() or df.payment_delay_days.notna().any() else "—",
        f"median stock {med_stock:.0f} days" if pd.notna(med_stock) else "add inventory / payment columns",
        src("Synthetic"))
    st.write("")

    f = df[df.tier.isin(st.session_state.get("tiers") or E.TIERS)]
    if st.session_state.get("issues"):
        f = f[f.primary_issue.isin(st.session_state.issues)]
    left, right = st.columns([5, 3], gap="medium")
    with left:
        h1, h2 = st.columns([3, 2], vertical_alignment="bottom")
        h1.markdown("**Priority attention list**  \n"
                    "<span class='small-note'>Ranked by risk × business size. Click a row to open the dealer.</span>",
                    unsafe_allow_html=True)
        q = h2.text_input("Search", placeholder="Search dealer, state or ID", label_visibility="collapsed",
                          key="board_search")
        if q:
            ql = q.strip().lower()
            f = f[f.territory.str.lower().str.contains(ql, regex=False) |
                  f.state_name.str.lower().str.contains(ql, regex=False) |
                  f.dealer_id.str.lower().str.contains(ql, regex=False)]
        table = pd.DataFrame({
            "#": f.scope_rank, "Dealer territory": f.territory, "State": f.state_name,
            "Tier": f.tier.map(lambda t: f"{U.TIER_ICON[t]} {t}"), "Health": f.health_score.round(0),
            "Δ6m": f.delta_6m.round(0), "Main issue": f.primary_issue, "Sales 12m": f.units_l12m.astype(int),
        })
        if f.state_name.nunique() <= 1:
            table = table.drop(columns=["State"])
        st.session_state._board_ids = f.dealer_id.tolist()
        st.dataframe(
            table, hide_index=True, height=560, key=f"board_table_{st.session_state.board_nonce}",
            on_select=_open_from_board, selection_mode="single-row",
            column_config={
                "#": st.column_config.NumberColumn(width=34),
                "Dealer territory": st.column_config.TextColumn(width=132),
                "State": st.column_config.TextColumn(width=92),
                "Tier": st.column_config.TextColumn(width=80),
                "Health": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d", width=70),
                "Δ6m": st.column_config.NumberColumn(format="%+d", width=46, help="Change in health score vs 6 months earlier"),
                "Main issue": st.column_config.TextColumn(width=162),
                "Sales 12m": st.column_config.NumberColumn(format="localized", width=64, help="Units, last 12 months"),
            })
        if f.empty:
            st.caption("No dealers match the focus filters.")
        export = f[["scope_rank", "dealer_id", "territory", "office_name", "state_name", "zone", "tier", "health_score", "delta_6m",
                    "primary_issue", "action_type", "units_l12m", "target_ach_pct", "share_change_pp"]].copy()
        export["top_action"] = [E.top_action(fl, iss) for fl, iss in zip(f.alerts, f.primary_issue)]
        st.download_button("Download attention list (CSV)", export.round(2).to_csv(index=False).encode("utf-8"),
                           file_name=f"dealer_attention_{as_of:%Y%m}.csv", mime="text/csv",
                           icon=":material/download:")
    with right:
        st.markdown("**Where the risk sits** <span class='small-note'>bubble = sales volume · hover for detail</span>",
                    unsafe_allow_html=True)
        show(C.quadrant(f, height=360), "board_quadrant")
        st.markdown("**What's driving attention** <span class='small-note'>dealers with each alert</span>",
                    unsafe_allow_html=True)
        rows = [(a["title"].replace(" (early warning)", "").replace(" in a stronghold", ""), a["pillar"])
                for fl in f.alerts for a in fl if a["severity"] in ("high", "medium")]
        counts = pd.DataFrame(rows, columns=["title", "pillar"]).value_counts().rename("n").reset_index() if rows \
            else pd.DataFrame(columns=["title", "pillar", "n"])
        show(C.driver_bars(counts, height=250), "board_drivers")


def view_deep_dive(df, allsc, hist, ops, panel, targets, as_of, weights):
    U.page_header("Dealer deep-dive", "Why this dealer is flagged, what to do, and the evidence behind it")
    if df.empty:
        st.info("No dealer territories in this scope — widen the zone/state filter.")
        return
    options = df.dealer_id.tolist()
    if st.session_state.get("dealer_id") not in options:
        st.session_state.dealer_id = options[0]
    labels = dict(zip(df.dealer_id, [f"#{r} · {n} — {s}  ({t})" for r, n, s, t in
                                     zip(df.scope_rank, df.territory, df.state_name, df.tier)]))
    c1, c2, c3 = st.columns([8, 1, 1], vertical_alignment="bottom")
    c1.selectbox("Dealer territory (sorted by priority)", options, key="dealer_id", format_func=lambda d: labels.get(d, d))
    c2.button("Prev", on_click=_step_dealer, args=(options, -1), width="stretch", icon=":material/chevron_left:")
    c3.button("Next", on_click=_step_dealer, args=(options, 1), width="stretch", icon=":material/chevron_right:")
    r = df[df.dealer_id == st.session_state.dealer_id].iloc[0]
    did = r.dealer_id

    meta = (f"Maruti territory {did} · office: {r.office_name} · {r.state_name} · {r.zone} zone · "
            if CTX["mode"] == "sample" else f"Dealer {did} · {r.state_name} · {r.zone} · ")
    U.hero_card(r.territory, meta + f"priority #{r.scope_rank} of {len(df)}",
                r.tier, r.primary_issue, r.action_type, r.health_score,
                r.delta_6m if pd.notna(r.delta_6m) else None, E.summary_sentence(r))
    st.write("")

    m = st.columns(8, gap="small")
    sample = CTX["mode"] == "sample"
    tag = lambda p: f" · {src(p).lower()}"
    m[0].metric("Sales 12m", fmt_num(r.units_l12m), fmt_delta(r.sales_growth_pct, "+.1f", "%"),
                border=True, help=("Maruti registrations in this territory, last 12 months (VAHAN)" if sample
                                   else "Units sold, last 12 months (your data)") + "; delta = YoY growth")
    m[1].metric("vs Target", fmt_num(r.target_ach_pct, 0, "%"),
                fmt_delta(r.target_ach_pct - 100, "+.0f", " pts"), border=True,
                help="12-month sales ÷ " + ("modelled target" if sample else "your target"))
    m[2].metric("Share", fmt_num(r.share_l12m_pct, 1, "%"), fmt_delta(r.share_change_pp, "+.1f", " pp"),
                border=True, help="Share of the local car market, change vs prior 12 months")
    m[3].metric("Expected", fmt_num(r.benchmark_share_pct, 1, "%"), fmt_delta(r.share_gap_pp, "+.1f", " pp"),
                border=True, help=("ML benchmark share for this market type (state, car penetration, 2W/tractor mix, size, growth)"
                                   if sample else f"Benchmark: {CTX['ds']['meta']['benchmark']}") + "; delta = actual minus expected")
    m[4].metric("Stock", fmt_num(r.inventory_days, 0, " d"), fmt_delta(r.inventory_change, "+.0f", " d"),
                delta_color="inverse", border=True, help="Inventory cover in days (3-month average); delta = change over 3 months" + tag("financial"))
    m[5].metric("Overdue", fmt_num(r.payment_delay_days, 0, " d"), fmt_delta(r.payment_delay_change, "+.0f", " d"),
                delta_color="inverse", border=True, help="Average days past due on OEM dues; delta = change over 3 months" + tag("financial"))
    m[6].metric("CSI", fmt_num(r.csi_score, 1), fmt_delta(r.csi_change, "+.1f"), border=True,
                help="Service customer-satisfaction index, 0–100; delta = change over 3 months" + tag("customer"))
    m[7].metric("Complaints", fmt_num(r.complaints_per_100_units, 1), None, border=True,
                help="Complaints per 100 units sold, last 3 months" + tag("customer"))

    left, right = st.columns([5, 7], gap="medium")
    alerts = E.ordered_alerts(r.alerts, r.primary_issue)
    with left:
        st.markdown("**Health breakdown**")
        show(C.pillar_bars(r, E.PILLARS, {p: src(p) for p in E.PILLARS}, height=190), f"pillars_{did}")
        nw = E.normalise_weights(weights)
        st.caption("Weights: " + " · ".join(f"{E.PILLARS[p]} {v:.0%}" for p, v in nw.items()))
        st.markdown("**Why it's flagged**")
        if not alerts:
            st.success("No alerts — performing in line with its market and operating norms.", icon=":material/check_circle:")
        for a in alerts:
            U.alert_card(a, src(a["pillar"]))
    with right:
        st.markdown("**Recommended actions**")
        acts = [a for a in alerts if a["severity"] != "info" or a["code"] == "MARKET_DIP"]
        if not acts:
            st.info("Maintain: keep the standard monthly review cadence.", icon=":material/thumb_up:")
        for a in acts[:5]:
            U.action_card(a)
        with st.expander("What-if: how much would fixing it help?", icon=":material/science:"):
            what_if(r, hist, as_of, weights, did)

    st.write("")
    tabs = st.tabs(["Sales vs target", "Market share", "Competition", "Inventory, payments & service", "Health trend"])
    pm = panel[panel.dealer_id == did].sort_values("date")
    pm = pm[(pm.date <= as_of) & (pm.date > as_of - pd.DateOffset(months=24))]
    with tabs[0]:
        t = targets[targets.dealer_id == did][["date", "target_units"]]
        show(C.sales_vs_target(pm[["date", "oem_units"]].merge(t, on="date", how="left"), height=320), f"sales_{did}")
        st.caption("Actual = Maruti registrations in this RTO (VAHAN). Target = modelled: prior fiscal-year sales × "
                   "(1 + zone market growth, capped 3–12%), phased by the zone's 2019 seasonality." if sample else
                   "Actual and target = units_sold and target_units from your file.")
    with tabs[1]:
        sh = pm[["date", "oem_units", "car_market"]].copy()
        sh["share"] = sh.oem_units.rolling(3, min_periods=1).sum() / sh.car_market.rolling(3, min_periods=1).sum() * 100
        ss = state_share_monthly(panel)
        sh = sh.merge(ss[ss.state_name == r.state_name][["date", "state_share"]], on="date", how="left")
        show(C.share_trend(sh, r.benchmark_share_pct, height=320), f"share_{did}")
    with tabs[2]:
        cur = pm[pm.date > as_of - pd.DateOffset(months=12)]
        prv = panel[(panel.dealer_id == did) & (panel.date <= as_of - pd.DateOffset(months=12)) &
                    (panel.date > as_of - pd.DateOffset(months=24))]
        comps = [c for c in CTX["ds"]["competitors"] if c in panel.columns]
        if not comps:
            st.info("Add competitor columns (comp_<Brand>, e.g. comp_Hyundai) to your file to see who is taking share "
                    "in each dealer's market.", icon=":material/info:")
        elif len(cur) and len(prv) and cur.car_market.sum() > 0 and prv.car_market.sum() > 0:
            own = "Maruti Suzuki (this dealer)" if sample else "Your brand (this dealer)"
            gains = pd.Series({c: (cur[c].sum() / cur.car_market.sum() - prv[c].sum() / prv.car_market.sum()) * 100
                               for c in comps + ["oem_units"]}).rename({"oem_units": own}).dropna()
            show(C.competitor_shift(gains, height=330), f"comp_{did}")
        else:
            show(C.empty(), f"comp_{did}")
    with tabs[3]:
        if ops is None:
            st.info("Add inventory_days, payment_delay_days, csi_score or complaints columns to your file to see "
                    "operating trends.", icon=":material/info:")
        else:
            o = ops[(ops.dealer_id == did) & (ops.date <= as_of) & (ops.date > as_of - pd.DateOffset(months=15))].copy()
            o["complaints_per_100"] = (o.complaints.rolling(3, min_periods=1).sum() /
                                       o.retail_units.rolling(3, min_periods=1).sum().clip(lower=1) * 100)
            med = network_ops_median(ops)
            med = med[(med.date <= as_of) & (med.date > as_of - pd.DateOffset(months=15))]
            show(C.ops_small_multiples(o, med, height=430), f"ops_{did}")
            st.caption(("Synthetic operating data, mechanistically linked to real sales: stock builds when dispatches "
                        "(to target) exceed actual retail. " if sample else "Operating data from your file. ")
                       + "Red dashed line = alert threshold.")
    with tabs[4]:
        hh = allsc[(allsc.dealer_id == did) & (allsc.as_of <= as_of)][["as_of", "health_score", "tier"]].sort_values("as_of")
        show(C.health_trend(hh, height=320), f"trend_{did}")


def what_if(r, hist, as_of, weights, did):
    raw = hist[(hist.dealer_id == did) & (hist.as_of == as_of)]
    if raw.empty:
        st.caption("No raw inputs available for this dealer.")
        return
    base = raw.iloc[0].to_dict()
    k = f"{did}_{as_of:%Y%m}"
    c1, c2 = st.columns(2)

    def slider(col, label, field, lo, hi, step=1.0, nd=0):
        """Slider that starts at the dealer's real value; if left untouched, the exact original value is used."""
        orig = float(base[field]) if base.get(field) is not None else float("nan")
        if not np.isfinite(orig):
            col.caption(f"{label}: not in your data")
            return orig
        start = round(orig, nd)
        hi = max(hi, float(np.ceil(orig)))
        lo = min(lo, float(np.floor(orig)))
        val = col.slider(label, float(lo), float(hi), float(start), step=step, key=f"wi_{field}_{k}",
                         format="%.1f" if nd else "%.0f")
        return orig if abs(val - start) < 1e-9 else val

    inv = slider(c1, "Inventory cover (days)", "inventory_days", 10, 150)
    aged = slider(c1, "Stock older than 60 days (%)", "aged_stock_pct", 0, 80)
    pay = slider(c1, "Payment delay (days)", "payment_delay_days", 0, 60)
    csi = slider(c2, "Service CSI", "csi_score", 55, 100)
    comp = slider(c2, "Complaints per 100 units", "complaints_per_100_units", 0, 8, step=0.1, nd=1)
    max_extra = int(max(base["target_l12m"] - base["units_l12m"], 0) + 0.25 * base["units_l12m"])
    extra = c2.slider("Extra units sold in the last 12 months", 0, max(max_extra, 10), 0, key=f"wi_extra_{k}")
    new = dict(base, inventory_days=inv, aged_stock_pct=aged, payment_delay_days=pay, csi_score=csi,
               complaints_per_100_units=comp, units_l12m=base["units_l12m"] + extra,
               car_market_l12m=base["car_market_l12m"] + extra,
               units_last6m=base["units_last6m"] + extra / 2, car_market_last6m=base["car_market_last6m"] + extra / 2,
               units_last3m=base["units_last3m"] + extra / 4)
    res = E.score_dealers(pd.DataFrame([new]), weights, volume_ref=hist[hist.as_of == as_of].units_l12m).iloc[0]
    a, b, c = st.columns(3)
    a.metric("Health now", f"{r.health_score:.0f}", r.tier, delta_color="off")
    b.metric("Health after changes", f"{res.health_score:.0f}", f"{res.health_score - r.health_score:+.0f} pts")
    c.metric("Tier after changes", res.tier, "improves" if E.TIERS.index(res.tier) > E.TIERS.index(r.tier)
             else ("worsens" if E.TIERS.index(res.tier) < E.TIERS.index(r.tier) else "unchanged"), delta_color="off")
    st.caption("Change (3-month) metrics keep their earlier reference values, so improvements also show up as positive trends.")


def view_quick_assess(weights, hist_latest):
    U.page_header("Quick assess", "Enter one dealer's numbers — or upload a file for many — and get a score, "
                                  "diagnosis and action plan instantly")
    t = E.template_frame().iloc[0]
    state_share = (hist_latest.groupby("state_name").units_l12m.sum() /
                   hist_latest.groupby("state_name").car_market_l12m.sum() * 100).round(1)
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.form("qa_form", border=True):
            st.markdown("**Sales & market** " + U.src_badge(src("Real")), unsafe_allow_html=True)
            a, b = st.columns(2)
            name = a.text_input("Dealer name", "Example Motors", max_chars=60)
            bench = b.number_input("Expected market share (%)", 1.0, 95.0, float(t.benchmark_share_pct), 0.5,
                                   help="Benchmark for this market. State averages are listed below the form.")
            ul = a.number_input("Units sold · last 12m", 1, 500000, int(t.units_l12m), 10)
            up_ = b.number_input("Units sold · prior 12m", 1, 500000, int(t.units_p12m), 10)
            tg = a.number_input("Target · last 12m", 1, 500000, int(t.target_l12m), 10)
            u3 = b.number_input("Units sold · last 3m", 0, 200000, int(t.units_last3m), 5)
            ml = a.number_input("Local car market · last 12m", 1, 5000000, int(t.car_market_l12m), 50)
            mp = b.number_input("Local car market · prior 12m", 1, 5000000, int(t.car_market_p12m), 50)
            t3 = a.number_input("Target · last 3m", 1, 200000, int(t.target_last3m), 5)
            cg = b.selectbox("Competitor gaining most", ["—"] + COMPETITORS, index=1 + COMPETITORS.index("Mahindra"))
            st.markdown("**Inventory & payments** " + U.src_badge(src("Synthetic")), unsafe_allow_html=True)
            a, b = st.columns(2)
            inv = a.number_input("Inventory cover (days)", 0.0, 365.0, float(t.inventory_days), 1.0)
            inv3 = b.number_input("Inventory 3 months ago (days)", 0.0, 365.0, float(t.inventory_days_3m_ago), 1.0)
            aged = a.number_input("Stock older than 60 days (%)", 0.0, 100.0, float(t.aged_stock_pct), 1.0)
            pdd = b.number_input("Payment delay (avg days past due)", 0.0, 180.0, float(t.payment_delay_days), 1.0)
            pdd3 = a.number_input("Payment delay 3 months ago", 0.0, 180.0, float(t.payment_delay_days_3m_ago), 1.0)
            st.markdown("**Service & customers** " + U.src_badge(src("Synthetic")), unsafe_allow_html=True)
            a, b = st.columns(2)
            csi = a.number_input("Service CSI (0–100)", 0.0, 100.0, float(t.csi_score), 0.5)
            csi3 = b.number_input("CSI 3 months ago", 0.0, 100.0, float(t.csi_score_3m_ago), 0.5)
            cp = a.number_input("Complaints per 100 units", 0.0, 50.0, float(t.complaints_per_100_units), 0.1)
            submitted = st.form_submit_button("Assess dealer", type="primary", width="stretch", icon=":material/bolt:")
        with st.expander("State average share (benchmark reference)"):
            st.dataframe(state_share.rename("Share %").reset_index().rename(columns={"state_name": "State"}),
                         hide_index=True, height=240)
    raw = dict(dealer_id="QA-1", dealer_name=name or "Dealer", units_l12m=ul, units_p12m=up_, target_l12m=tg,
               car_market_l12m=ml, car_market_p12m=mp, benchmark_share_pct=bench, units_last3m=u3, target_last3m=t3,
               inventory_days=inv, inventory_days_3m_ago=inv3, aged_stock_pct=aged, payment_delay_days=pdd,
               payment_delay_days_3m_ago=pdd3, csi_score=csi, csi_score_3m_ago=csi3, complaints_per_100_units=cp,
               top_competitor_gainer=None if cg == "—" else cg)
    with right:
        try:
            clean, warns = E.validate_inputs(pd.DataFrame([raw]))
            res = E.score_dealers(clean, weights, volume_ref=hist_latest.units_l12m).iloc[0]
        except E.InputError as e:
            st.error(f"Please check the inputs: {e}", icon=":material/error:")
            return
        if submitted:
            st.toast("Assessment updated", icon=":material/check:")
        U.hero_card(res.dealer_name, "Manual assessment · benchmarked against the "
                    + ("sample network" if CTX["mode"] == "sample" else "uploaded network"), res.tier,
                    res.primary_issue, res.action_type, res.health_score, "hide", E.summary_sentence(res))
        st.write("")
        show(C.pillar_bars(res, E.PILLARS, {p: src(p) for p in E.PILLARS}, height=190), "qa_pillars")
        if res.data_coverage < 0.999:
            st.caption(f"Data coverage {res.data_coverage:.0%}: momentum metrics need 6-month market data, which "
                       "this form doesn't collect, so the score is re-weighted over the available metrics.")
        al = E.ordered_alerts(res.alerts, res.primary_issue)
        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown("**Why it's flagged**")
            if not al:
                st.success("No alerts.", icon=":material/check_circle:")
            for x in al:
                U.alert_card(x, src(x["pillar"]))
        with b:
            st.markdown("**Recommended actions**")
            for x in [x for x in al if x["severity"] != "info" or x["code"] == "MARKET_DIP"][:5]:
                U.action_card(x)
    st.markdown("---")
    batch_upload(weights, hist_latest)


def batch_upload(weights, hist_latest):
    st.markdown("**Batch assessment** <span class='small-note'>score your own dealer list with the same engine</span>",
                unsafe_allow_html=True)
    a, b = st.columns([2, 5], gap="medium")
    with a:
        st.download_button("Download CSV template", E.template_frame().to_csv(index=False).encode("utf-8"),
                           file_name="dealerpulse_template.csv", mime="text/csv", width="stretch",
                           icon=":material/download:")
        st.caption("Required: " + ", ".join(E.REQUIRED) + ". All other columns are optional — missing metrics are "
                   "skipped and the score is re-weighted.")
    with b:
        up = st.file_uploader("Upload dealer file", type=["csv", "xlsx"], key="uploader", label_visibility="collapsed")
    if up is None:
        return
    try:
        out, warns = E.score_upload(up.name, up.getvalue(), weights, hist_latest.units_l12m)
    except E.InputError as e:
        st.error(f"Upload rejected: {e}", icon=":material/error:")
        return
    for w in warns:
        st.warning(w, icon=":material/warning:")
    out["Tier"] = out.Tier.map(lambda t: f"{U.TIER_ICON[t]} {t}")
    st.success(f"Scored {len(out):,} dealer(s).", icon=":material/check_circle:")
    st.dataframe(out, hide_index=True, column_config={
        "Health": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Data coverage": st.column_config.NumberColumn(format="%d%%"),
        "Top recommended action": st.column_config.TextColumn(width="large")})
    st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"), "dealerpulse_results.csv",
                       "text/csv", icon=":material/download:")


def view_insights(df_scope, allsc_latest, panel, nat, as_of, scope_label):
    U.page_header("Network insights", f"Context behind the alerts · {scope_label} · as of {as_of:%b %Y}")
    a, b = st.columns(2, gap="medium")
    with a:
        if nat is not None:
            st.markdown("**Who is taking share nationally** <span class='small-note'>2019 vs last 12 months · REAL</span>",
                        unsafe_allow_html=True)
            share = nat.div(nat.sum(axis=1), axis=0) * 100
            y19 = share[share.index.year == 2019].mean()
            l12 = share[(share.index <= as_of) & (share.index > as_of - pd.DateOffset(months=12))].mean()
            d = (l12 - y19)
            show(C.brand_shift(d[d.abs() > 0.4], height=360), "ins_brand")
            st.caption(f"Maruti: {y19['Maruti Suzuki']:.1f}% (2019) → {l12['Maruti Suzuki']:.1f}% (last 12m). "
                       "Car market = 16 car brands; Tata/Mahindra include some light commercial vehicles.")
        else:
            st.markdown("**Who is taking share in your markets** <span class='small-note'>last 12 months vs prior 12 · "
                        "your data</span>", unsafe_allow_html=True)
            comps = [c for c in CTX["ds"]["competitors"] if c in panel.columns]
            cur = panel[(panel.date <= as_of) & (panel.date > as_of - pd.DateOffset(months=12))]
            prv = panel[(panel.date <= as_of - pd.DateOffset(months=12)) & (panel.date > as_of - pd.DateOffset(months=24))]
            if comps and cur.car_market.sum() > 0 and prv.car_market.sum() > 0:
                d = pd.Series({c: (cur[c].sum() / cur.car_market.sum() - prv[c].sum() / prv.car_market.sum()) * 100
                               for c in comps + ["oem_units"]}).rename({"oem_units": "Your brand"}).dropna()
                show(C.brand_shift(d, height=360), "ins_brand")
                st.caption("Share of the combined local car market across all uploaded dealers.")
            else:
                st.info("Add comp_<Brand> columns to your file to see which competitors are gaining share.",
                        icon=":material/info:")
    with b:
        if CTX["mode"] == "sample":
            st.markdown("**Seasonality differs by zone** <span class='small-note'>2022–23 average · REAL</span>",
                        unsafe_allow_html=True)
            p = panel[panel.date.dt.year.isin([2022, 2023])]
        else:
            st.markdown("**Seasonality by zone** <span class='small-note'>last 24 months · your data</span>",
                        unsafe_allow_html=True)
            p = panel[(panel.date <= as_of) & (panel.date > as_of - pd.DateOffset(months=24))]
        zm = p.groupby(["zone", p.date.dt.month]).oem_units.sum().unstack()
        idx = zm.div(zm.mean(axis=1), axis=0)
        show(C.seasonality(idx, height=360), "ins_season")
        if CTX["mode"] == "sample":
            st.caption("South runs high through Aug–Sep (Onam, Ganesh Chaturthi) while West/East peak in October "
                       "(Navratri, Diwali); North and South also spike in January (new model-year registrations). "
                       "Targets and stock planning should follow the local calendar.")
    a, b = st.columns([6, 5], gap="medium")
    with a:
        st.markdown("**Zone scorecard** <span class='small-note'>all zones</span>", unsafe_allow_html=True)
        z = allsc_latest.groupby("zone").agg(
            territories=("dealer_id", "size"), units=("units_l12m", "sum"), units_p=("units_p12m", "sum"),
            mkt=("car_market_l12m", "sum"), mkt_p=("car_market_p12m", "sum"),
            attention=("tier", lambda t: t.isin(["Critical", "Watch"]).mean() * 100),
            health=("health_score", "median"))
        lose = allsc_latest[allsc_latest.share_z <= -1.96]
        topc = lose.groupby("zone").top_competitor_gainer.agg(lambda s: s.mode().iat[0] if len(s.mode()) else "—")
        tab = pd.DataFrame({
            "Zone": z.index, "Territories": z.territories, "Share %": (z.units / z.mkt * 100).round(1),
            "Share Δ pp": ((z.units / z.mkt - z.units_p / z.mkt_p) * 100).round(1),
            "Sales growth %": ((z.units / z.units_p - 1) * 100).round(1),
            "Need attention %": z.attention.round(0),
            "Main competitor gaining": topc.reindex(z.index).fillna("—")}).sort_values("Need attention %", ascending=False)
        st.dataframe(tab, hide_index=True, column_config={
            "Zone": st.column_config.TextColumn(width=85), "Territories": st.column_config.NumberColumn(width=78),
            "Share %": st.column_config.NumberColumn(width=62), "Sales growth %": st.column_config.NumberColumn(width=95),
            "Need attention %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%", width=120),
            "Share Δ pp": st.column_config.NumberColumn(format="%+.1f", width=75),
            "Main competitor gaining": st.column_config.TextColumn(width=150)})
        st.markdown("**Competitor threat map** <span class='small-note'>territories with significant share loss, "
                    f"by competitor gaining most · {src('Real').upper()}</span>", unsafe_allow_html=True)
        ct = pd.crosstab(lose.zone, lose.top_competitor_gainer)
        if ct.empty:
            st.caption("No significant share losses in this snapshot.")
        else:
            ct = ct[ct.sum().sort_values(ascending=False).index[:5]]
            st.dataframe(ct, column_config={c: st.column_config.NumberColumn(width=80) for c in ct.columns})
    with b:
        group = "state_name" if len(df_scope.state_name.unique()) > 1 else "primary_issue"
        st.markdown(f"**Where attention concentrates · by {'state' if group == 'state_name' else 'main issue'}** "
                    f"<span class='small-note'>{scope_label} · states with ≥ 5 territories</span>", unsafe_allow_html=True)
        show(C.tier_by_group(df_scope, group, height=420), "ins_tiers")


def view_method(validation, df_scope, as_of):
    U.page_header("Method & data", "How DealerPulse turns sample inputs into a recommendation — and how far to trust it")
    U.flow([
        ("Inputs", "Real VAHAN registrations by RTO & brand; synthetic inventory, payments, service, complaints"),
        ("Signals", f"{len(E.RULES)} metrics: target achievement, growth vs market, significance-tested share "
                    "change, share vs ML-expected, momentum, stock, overdues, CSI, complaints"),
        ("Score", "Each metric scored 0–100 between a red line and a good level; 4 weighted pillars → health score"),
        ("Diagnose", "Rule-based alerts with evidence; tier = score + alert severity; priority = risk × business size"),
        ("Act", "Each alert maps to a playbook action with owner and deadline: support, corrective or grow"),
    ])
    st.write("")
    v = validation
    k = st.columns(4, gap="small")
    if v is None:                                     # uploaded network: describe the data actually used
        mt = CTX["ds"]["meta"]
        kpi(k[0], "Your file", f"{mt['dealers']:,} dealers", f"{mt['months']} months · {mt['first']:%b %Y} – {mt['last']:%b %Y}",
            "Your data")
        kpi(k[1], "Operating metrics supplied", f"{len(mt['ops_columns'])} of {len(I.M_OPS)}",
            ", ".join(mt["ops_columns"]) or "none — score uses sales & momentum only", "Your data")
        kpi(k[2], "Competitors supplied", f"{len(mt['competitors'])}", ", ".join(mt["competitors"][:5]) or "none", "Your data")
        kpi(k[3], "Share benchmark", "Yours" if mt["benchmark"].startswith("your") else "Average",
            mt["benchmark"], "Your data")
        st.caption("The validation evidence (backtests, ML test, detection recall) was produced on the sample network "
                   "— switch Data → Sample network to see it. The scoring engine is identical for your data.")
    else:
        es = v["expected_share_model"]
        kpi(k[0], "Expected-share benchmark", f"R² {es['cv_r2']:.2f}",
            f"error {es['cv_mae_pp']:.1f} pp vs {es['state_average_mae_pp']:.1f} pp for the state average (cross-validated)", "Real")
        bt = v["backtest"]
        kpi(k[1], "Share-loss signal persists", f"{bt['future_share_change_after_significant_loss_pp']:+.2f} pp",
            f"next-12m share change after a significant loss, vs {bt['future_share_change_others_pp']:+.2f} pp for others", "Real")
        ml = v["ml_early_warning"]["test_auc"]
        kpi(k[2], "ML early-warning tested", f"AUC {ml['Logistic regression']:.2f}",
            f"vs {ml['Rule: current share level']:.2f} for a one-line rule → not deployed (no gain, regime shift)", "Real")
        gt = v["ground_truth_recall"]
        rec = np.mean([x["recall_pct"] for x in gt.values()])
        kpi(k[3], "Catches emerging problems", f"{rec:.0f}% recall",
            "on injected stock-push, credit-squeeze and service-disruption events", "Synthetic")
    st.write("")
    a, b = st.columns([6, 5], gap="medium")
    with a:
        st.markdown("**Scoring rules** <span class='small-note'>score = 0 at the red line, 100 at the good level, linear between</span>",
                    unsafe_allow_html=True)
        rules = pd.DataFrame([{
            "Pillar": E.PILLARS[r.pillar], "Metric": r.label, "Red line": f"{r.bad:g} {r.unit}".strip(),
            "Good": f"{r.good:g} {r.unit}".strip(), "Weight": f"{r.weight:.0%}"} for r in E.RULES])
        st.dataframe(rules, hide_index=True, height=530, column_config={
            "Pillar": st.column_config.TextColumn(width=150), "Metric": st.column_config.TextColumn(width=230),
            "Red line": st.column_config.TextColumn(width=75, help="Scores 0"),
            "Good": st.column_config.TextColumn(width=70, help="Scores 100"),
            "Weight": st.column_config.TextColumn(width=60, help="Weight within the pillar")})
        st.caption("* reliability-adjusted: rates measured on few units are shrunk towards neutral by n/(n+K) "
                   "(empirical Bayes), so small dealers are not flagged on noise. Share changes use a "
                   f"quasi-binomial test (dispersion φ = {E.SHARE_DISPERSION:g}).")
        st.caption(f"Tiers: Critical < {E.TIER_CUTS[0]}, Watch < {E.TIER_CUTS[1]}, Stable < {E.TIER_CUTS[2]}, Strong ≥ "
                   f"{E.TIER_CUTS[2]}; one high-severity alert caps the tier at Watch, two at Critical.")
    with b:
        st.markdown("**Data sources**")
        if v is None:
            mt = CTX["ds"]["meta"]
            rows = [("Sales, targets, local market", "Your data", "units_sold, target_units, local_car_market"),
                    ("Share benchmark", "Your data", mt["benchmark"]),
                    ("Operating metrics", "Your data", ", ".join(mt["ops_columns"]) or "not supplied (pillars skipped)"),
                    ("Competitors", "Your data", ", ".join(mt["competitors"]) or "not supplied")]
        else:
            rows = [
                ("Sales, growth, market potential, competition", "Real", "VAHAN registrations by maker & RTO (MoRTH via India Data Portal), Jan-2019 – May-2024"),
                ("Sales targets", "Modelled", "Prior fiscal-year sales × (1 + zone market growth, capped 3–12%), 2019 seasonality"),
                ("Expected share benchmark", "Real + ML", "Gradient boosting on market structure, out-of-fold"),
                ("Inventory & aged stock", "Synthetic", "Dispatches follow target, retail = actual VAHAN sales → stock builds on shortfall"),
                ("Payment delays", "Synthetic", "Rise with stock stress and sales shortfall; latent dealer resilience"),
                ("Service CSI & complaints", "Synthetic", "Latent service quality (linked to real share trend) + stock stress"),
                ("Dealer network structure", "Real", "Telangana RTA dealer roster (benchmark only; no join possible)")]
        st.dataframe(pd.DataFrame(rows, columns=["Attribute", "Source", "How"]), hide_index=True,
                     height=290 if v is not None else 180, column_config={
            "Attribute": st.column_config.TextColumn(width=170), "Source": st.column_config.TextColumn(width=75),
            "How": st.column_config.TextColumn(width=420)})
        st.markdown("**Why these choices**")
        sig_line = ""
        if v is not None:
            sig = v["significance"]
            sig_line = (f" Only {sig['pct_share_changes_significant']:.0f}% of 12-month share changes in the sample are "
                        f"statistically real; of {sig['hidden_risk_raw']} 'growing but losing share' dealers, "
                        f"{sig['hidden_risk_significant']} pass the test — the rest would be false alarms.")
        st.markdown(
            f"- **Significance testing:** share losses must pass a quasi-binomial test.{sig_line}\n"
            f"- **Fair benchmark:** premium metros naturally have lower share, so each dealer is compared with "
            f"the share expected for its *type* of market, not a flat average.\n"
            f"- **No black box:** a predictive model was tested and rejected; every alert shows its evidence.\n"
            f"- **Revenue at stake:** ₹{E.REVENUE_PER_UNIT_LAKH} lakh per unit (Maruti FY24 net sales ÷ volume).")
    st.markdown("**Assumptions & limitations**")
    if v is None:
        st.markdown(
            "- Scores compare each dealer's latest 12 months with the 12 before; dealers with gaps in the last 24 months of "
            "units, target or market data are skipped (listed in the sidebar).\n"
            "- Metrics missing from your file are left out and the health score is re-weighted over what is supplied "
            "(Data coverage).\n"
            "- Alert thresholds and weights are the prototype defaults; weights can be tuned in the sidebar.")
    else:
        st.markdown(
            "- RTO registrations proxy dealer sales: one RTO can be served by several dealers, and buyers sometimes register "
            "elsewhere. Each 'dealer' is a Maruti territory (brand × RTO) with a pseudonymous ID.\n"
            "- Inventory, payments, service and complaints are **synthetic** because OEMs don't publish them; in production "
            "these come from the DMS/ERP/CRM — upload them via Data → Your data and every view runs on them.\n"
            f"- Data ends May 2024; Telangana is not covered by VAHAN in this period. Excluded during cleaning: "
            f"{v['exclusions']['contaminated_offices']} RTOs contaminated with leaked national totals, "
            f"{v['exclusions']['fitness_centres']} fitness/testing centres, {v['exclusions']['special_offices']} special-purpose "
            f"offices and {v['exclusions']['jurisdiction_breaks']} territories whose jurisdiction changed (step change in "
            f"all-brand registrations).")
    st.download_button("Download full scorecard for this scope (CSV)",
                       df_scope.drop(columns=["alerts"]).round(3).to_csv(index=False).encode("utf-8"),
                       file_name=f"dealer_scorecard_{as_of:%Y%m}.csv", mime="text/csv", icon=":material/download:")


# =============================================================================== main
def main():
    _init_state()
    ds, error = data_sidebar()
    if ds is None:
        view_byod_landing(error)
        return
    CTX["mode"], CTX["ds"] = ds["mode"], ds
    hist, ops, panel, targets = ds["hist"], ds["ops"], ds["panel"], ds["targets"]
    zone, state, as_of, tiers, issues, weights = sidebar(ds)
    allsc = score_all(ds["key"], tuple(sorted(weights.items())), hist)
    df = scope_frame(allsc, as_of, zone, state)
    scope_label = "All India" if zone == "All India" else (f"{zone} zone" if state == "All states" else f"{state} ({zone})")
    if ds["mode"] == "upload":
        scope_label = ("Your network" if zone == "All India" else scope_label) + " (uploaded)"

    st.segmented_control("View", VIEWS, key="nav", format_func=lambda v: f"{VIEW_ICON.get(v, '')} {v}",
                         on_change=_nav_guard, label_visibility="collapsed")
    view = st.session_state.get("nav") or VIEWS[0]
    st.session_state._last_nav = view

    if view == "Attention Board":
        view_board(df, scope_label, as_of)
    elif view == "Dealer Deep-Dive":
        view_deep_dive(df, allsc, hist, ops, panel, targets, as_of, weights)
    elif view == "Quick Assess":
        view_quick_assess(weights, hist[hist.as_of == as_of])
    elif view == "Network Insights":
        view_insights(df, allsc[allsc.as_of == as_of], panel, ds["nat"], as_of, scope_label)
    else:
        view_method(ds["validation"], df, as_of)


main()
