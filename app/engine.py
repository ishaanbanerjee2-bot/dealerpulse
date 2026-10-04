"""
Dealer attention engine - one code path for the sample network, uploaded CSVs and manual inputs.

raw inputs  ->  derive_metrics()  ->  score_dealers()  ->  flags / diagnosis / actions / priority
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import numpy as np
import pandas as pd

REVENUE_PER_UNIT_LAKH = 6.32      # Maruti FY24 net sales Rs 13,494 cr / 2.135M vehicles
NORM_STOCK_DAYS = 35              # healthy dealer stock cover used to size corrective actions
# Empirical-Bayes reliability: a rate measured on n units is shrunk by n / (n + K), K = noise var / signal var.
# Growth-vs-market: noise var ~ 2e4/n pp^2, signal sd ~ 12 pp -> K ~ 140. Target achievement: 1e4/n, sd ~ 12 -> K ~ 70.
K_GROWTH, K_TARGET = 140, 70
# Registrations are over-dispersed vs binomial (fleet / bulk registrations): median Pearson dispersion of monthly
# share across territories = 2.3 -> quasi-binomial standard errors use phi = 2.
SHARE_DISPERSION = 2.0

# ----------------------------------------------------------------------------- input schema
REQUIRED = ["dealer_id", "units_l12m", "units_p12m", "target_l12m", "car_market_l12m", "car_market_p12m"]
OPTIONAL_NUMERIC = [
    "benchmark_share_pct", "units_last6m", "units_last6m_prev_year", "car_market_last6m",
    "car_market_last6m_prev_year", "units_last3m", "target_last3m", "inventory_days", "inventory_days_3m_ago",
    "closing_stock",
    "aged_stock_pct", "payment_delay_days", "payment_delay_days_3m_ago", "csi_score", "csi_score_3m_ago",
    "complaints_per_100_units", "top_competitor_gain_pp",
]
OPTIONAL_TEXT = ["dealer_name", "territory", "office_name", "state_name", "zone", "top_competitor_gainer"]
NUMERIC = [c for c in REQUIRED if c != "dealer_id"] + OPTIONAL_NUMERIC

# ----------------------------------------------------------------------------- scoring rules
PILLARS = {
    "sales": "Sales & market share",
    "momentum": "Recent momentum",
    "financial": "Inventory & payments",
    "customer": "Service & complaints",
}
PILLAR_SOURCE = {"sales": "Real", "momentum": "Real", "financial": "Synthetic", "customer": "Synthetic"}
DEFAULT_WEIGHTS = {"sales": 35, "momentum": 15, "financial": 30, "customer": 20}


@dataclass(frozen=True)
class Rule:
    metric: str
    pillar: str
    weight: float
    bad: float      # value that scores 0
    good: float     # value that scores 100 (bad > good means lower is better)
    label: str
    unit: str


RULES = [
    Rule("target_ach_adj", "sales", 0.30, 80, 105, "Target achievement (12m)*", "%"),
    Rule("growth_vs_market_adj", "sales", 0.25, -15, 5, "Sales growth vs market growth*", "pp"),
    Rule("share_z", "sales", 0.25, -3, 1, "Share change significance (z)", "z"),
    Rule("share_gap_pp", "sales", 0.20, -10, 2, "Share vs expected for this market", "pp"),
    Rule("gvm_6m_adj", "momentum", 0.50, -20, 5, "Last-6m growth vs market*", "pp"),
    Rule("target_ach_3m_adj", "momentum", 0.50, 70, 100, "Target achievement (3m)*", "%"),
    Rule("inventory_days", "financial", 0.25, 90, 35, "Inventory cover", "days"),
    Rule("inventory_change", "financial", 0.10, 25, 0, "Change in inventory cover (3m)", "days"),
    Rule("aged_stock_pct", "financial", 0.15, 40, 5, "Stock older than 60 days", "%"),
    Rule("payment_delay_days", "financial", 0.35, 30, 5, "Payment delay (avg days past due)", "days"),
    Rule("payment_delay_change", "financial", 0.15, 10, 0, "Change in payment delay (3m)", "days"),
    Rule("csi_score", "customer", 0.45, 70, 90, "Service CSI", "/100"),
    Rule("csi_change", "customer", 0.20, -6, 0, "Change in CSI (3m)", "pts"),
    Rule("complaints_per_100_units", "customer", 0.35, 3.5, 0.5, "Complaints per 100 units", ""),
]
RULE_BY_METRIC = {r.metric: r for r in RULES}

TIERS = ["Critical", "Watch", "Stable", "Strong"]          # ordered worst -> best
TIER_CUTS = (45, 60, 75)                                     # health score cut-offs between tiers


# ----------------------------------------------------------------------------- validation
class InputError(ValueError):
    """Raised for user-facing input problems (bad upload, missing columns)."""


def validate_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce types and return (clean_df, warnings). Raises InputError for fatal problems."""
    if df is None or len(df) == 0:
        raise InputError("The file has no rows.")
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError("Missing required column(s): " + ", ".join(missing))
    warnings: list[str] = []
    df["dealer_id"] = df["dealer_id"].astype(str).str.strip()
    if (df.dealer_id == "").any() or df.dealer_id.isin(["nan", "None"]).any():
        raise InputError("Every row needs a dealer_id.")
    if df.dealer_id.duplicated().any():
        dups = df.dealer_id[df.dealer_id.duplicated()].unique()[:5]
        raise InputError("Duplicate dealer_id values: " + ", ".join(map(str, dups)))
    for c in NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
            continue
        raw = df[c]
        txt = raw.astype(str).str.replace(",", "").str.replace("%", "").str.strip()
        conv = pd.to_numeric(txt.where(~txt.isin(["", "nan", "None"])), errors="coerce")
        bad = conv.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
        if bad.any():
            warnings.append(f"'{c}': {int(bad.sum())} non-numeric value(s) treated as missing.")
        conv = conv.where(np.isfinite(conv))
        neg = conv < 0
        if c not in ("top_competitor_gain_pp",) and neg.any():
            warnings.append(f"'{c}': {int(neg.sum())} negative value(s) treated as missing.")
            conv = conv.where(~neg)
        df[c] = conv.astype(float)
    for c in REQUIRED[1:]:
        if df[c].isna().any():
            raise InputError(f"Required column '{c}' has missing or invalid values "
                             f"(rows: {', '.join(map(str, (df.index[df[c].isna()] + 2)[:5]))}).")
    if (df.car_market_l12m < df.units_l12m).any() or (df.car_market_p12m < df.units_p12m).any():
        raise InputError("car_market must be at least the dealer's own units (it is the whole local car market).")
    if (df.car_market_l12m == 0).any() or (df.car_market_p12m == 0).any():
        raise InputError("car_market values must be greater than zero.")
    if (df.target_l12m == 0).any():
        raise InputError("target_l12m must be greater than zero.")
    for c in OPTIONAL_TEXT:
        if c not in df.columns:
            df[c] = None
    pct_cols = ["aged_stock_pct", "benchmark_share_pct"]
    for c in pct_cols:
        if (df[c] > 100).any():
            raise InputError(f"'{c}' must be a percentage between 0 and 100.")
    if (df.csi_score > 100).any() or (df.csi_score_3m_ago > 100).any():
        raise InputError("CSI scores must be on a 0-100 scale.")
    return df, warnings


def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV (any common delimiter / encoding) or Excel file. Raises InputError."""
    import csv
    import io
    name = (name or "").lower()
    if not data:
        raise InputError("The uploaded file is empty.")
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception as e:
            raise InputError(f"Could not read the Excel file ({type(e).__name__}).")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise InputError("The uploaded file has no data.")
        except (pd.errors.ParserError, csv.Error, ValueError, TypeError) as e:
            raise InputError(f"Could not parse the CSV ({type(e).__name__}) — please use the template.")
        if df.shape[1] == 1 and "," not in df.columns[0] and len(REQUIRED) > 1:
            raise InputError("Only one column was found — check the delimiter (use comma-separated values).")
        return df
    raise InputError("Could not decode the file — save it as UTF-8 CSV or Excel.")


# ----------------------------------------------------------------------------- metrics
def _safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, np.nan)


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dealer inputs -> scoring metrics. Missing optional inputs give NaN metrics (handled in scoring)."""
    m = df.copy()
    m["share_l12m_pct"] = _safe_div(m.units_l12m, m.car_market_l12m) * 100
    m["share_p12m_pct"] = _safe_div(m.units_p12m, m.car_market_p12m) * 100
    m["share_change_pp"] = m.share_l12m_pct - m.share_p12m_pct
    pl, pp = m.share_l12m_pct / 100, m.share_p12m_pct / 100
    se = np.sqrt(SHARE_DISPERSION * (_safe_div(pl * (1 - pl), m.car_market_l12m) +
                                     _safe_div(pp * (1 - pp), m.car_market_p12m)))
    se = np.where(se > 0, se, np.nan)
    m["share_z"] = np.clip(_safe_div(pl - pp, se), -6, 6)
    m.loc[(pl == pp).to_numpy(), "share_z"] = 0.0                     # no change -> neutral (incl. 0% / 100%)
    m["sales_growth_pct"] = (_safe_div(m.units_l12m, m.units_p12m) - 1) * 100
    m["market_growth_pct"] = (_safe_div(m.car_market_l12m, m.car_market_p12m) - 1) * 100
    m["growth_vs_market_pp"] = m.sales_growth_pct - m.market_growth_pct
    m["target_ach_pct"] = _safe_div(m.units_l12m, m.target_l12m) * 100
    m["share_gap_pp"] = m.share_l12m_pct - m.benchmark_share_pct
    g6 = (_safe_div(m.units_last6m, m.units_last6m_prev_year) - 1) * 100
    gm6 = (_safe_div(m.car_market_last6m, m.car_market_last6m_prev_year) - 1) * 100
    m["sales_growth_6m_pct"] = g6
    m["gvm_6m_pp"] = g6 - gm6
    m["target_ach_3m_pct"] = _safe_div(m.units_last3m, m.target_last3m) * 100
    # reliability-adjusted versions used for scoring (raw values are kept for display)
    n12 = np.minimum(m.units_l12m, m.units_p12m)
    m["growth_vs_market_adj"] = m.growth_vs_market_pp * n12 / (n12 + K_GROWTH)
    n6 = np.minimum(m.units_last6m, m.units_last6m_prev_year)
    m["gvm_6m_adj"] = m.gvm_6m_pp * n6 / (n6 + K_GROWTH)
    m["target_ach_adj"] = 100 + (m.target_ach_pct - 100) * m.units_l12m / (m.units_l12m + K_TARGET)
    m["target_ach_3m_adj"] = 100 + (m.target_ach_3m_pct - 100) * m.units_last3m / (m.units_last3m + K_TARGET)
    m["payment_delay_change"] = m.payment_delay_days - m.payment_delay_days_3m_ago
    m["inventory_change"] = m.inventory_days - m.inventory_days_3m_ago
    m["csi_change"] = m.csi_score - m.csi_score_3m_ago
    m["shortfall_units"] = np.maximum(m.target_l12m - m.units_l12m, 0)
    m["revenue_gap_cr"] = m.shortfall_units * REVENUE_PER_UNIT_LAKH / 100
    m["headroom_units"] = np.maximum((m.benchmark_share_pct - m.share_l12m_pct) / 100 * m.car_market_l12m, 0)
    m["headroom_cr"] = m.headroom_units * REVENUE_PER_UNIT_LAKH / 100
    # excess stock sized from the same cover figure the manager sees (days x average monthly sales);
    # closing_stock is only used when inventory_days is not supplied
    monthly = m.units_l12m / 12
    cover = m.inventory_days.where(m.inventory_days.notna(), _safe_div(m.closing_stock * 30.4, monthly))
    m["excess_stock_units"] = np.maximum((cover - NORM_STOCK_DAYS) / 30.4 * monthly, 0)
    return m


def rule_score(x, rule: Rule):
    x = np.asarray(x, float)
    s = (x - rule.bad) / (rule.good - rule.bad) * 100
    return np.clip(s, 0, 100)


def normalise_weights(weights: dict | None) -> dict:
    w = {k: float(max((weights or {}).get(k, 0), 0)) for k in PILLARS}
    total = sum(w.values())
    if total <= 0 or not math.isfinite(total):
        w = {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}
        total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# ----------------------------------------------------------------------------- flags
def _f(x, nd=0):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{nd}f}"


def _signed(x, nd=1):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.{nd}f}"


def _ok(x):
    """True for any finite real number (python or numpy, int or float); False for None/NaN/inf/bool/text."""
    return isinstance(x, (numbers.Real, np.number)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def dealer_flags(r: pd.Series) -> list[dict]:
    """Rule-based alerts for one dealer. Each flag carries evidence and a concrete action."""
    fl = []
    comp = r.get("top_competitor_gainer") or "the leading competitor"
    comp_gain = r.get("top_competitor_gain_pp")
    comp_txt = f" — {comp} gained {_signed(comp_gain)} pp" if _ok(comp_gain) and comp_gain > 0 else ""
    mg, sg = r.get("market_growth_pct"), r.get("sales_growth_pct")

    def add(code, title, sev, pillar, evidence, action, kind, owner, horizon):
        fl.append(dict(code=code, title=title, severity=sev, pillar=pillar, evidence=evidence, action=action,
                       action_type=kind, owner=owner, horizon=horizon))

    z, dsh = r.get("share_z"), r.get("share_change_pp")
    market_dip = _ok(sg) and sg < 0 and _ok(mg) and mg < 0 and _ok(z) and z > -1.0
    if _ok(z) and z <= -1.96:
        sev = "high" if _ok(dsh) and dsh <= -4 else "medium"
        stronghold = _ok(r.get("share_gap_pp")) and r.get("share_gap_pp") >= 5
        add("SHARE_LOSS", "Losing market share" + (" in a stronghold" if stronghold else ""), sev, "sales",
            f"Share {_signed(dsh)} pp to {_f(r.get('share_l12m_pct'), 1)}% (statistically significant, z={z:.1f})"
            f"{comp_txt}.",
            f"Conquest plan against {comp}: SUV test-drive drive and exchange bonus on {comp} trade-ins; "
            f"audit enquiry-to-booking conversion of the sales team.",
            "Corrective", "Area Sales Manager", "30 days")
    gvm, gvm_adj = r.get("growth_vs_market_pp"), r.get("growth_vs_market_adj")
    if _ok(gvm_adj) and gvm_adj <= -8 and not market_dip and not any(f["code"] == "SHARE_LOSS" for f in fl):
        add("LAG_MARKET", "Growing slower than the local market", "medium", "sales",
            f"Sales {_signed(sg)}% vs market {_signed(mg)}% ({_signed(gvm)} pp).",
            "Review lead follow-up and model availability; add semi-urban outreach (mobile showroom, rural sales points).",
            "Corrective", "Area Sales Manager", "60 days")
    ta = r.get("target_ach_pct")
    if _ok(ta) and ta < 88:
        sev = "high" if ta < 78 else "medium"
        add("BELOW_TARGET", "Below sales target", sev, "sales",
            f"{_f(ta)}% of 12-month target — {_f(r.get('shortfall_units'))} units short "
            f"(≈ ₹{_f(r.get('revenue_gap_cr'), 1)} cr revenue).",
            ("Revise target to market reality and agree a monthly recovery plan." if market_dip else
             "Joint target review with the dealer principal; agree a month-by-month recovery plan."),
            "Support" if market_dip else "Corrective", "Area Sales Manager", "30 days")
    g6, gvm6, gvm6_adj = r.get("sales_growth_6m_pct"), r.get("gvm_6m_pp"), r.get("gvm_6m_adj")
    if _ok(gvm6_adj) and gvm6_adj <= -10 and (not _ok(gvm_adj) or gvm_adj > -5):
        add("MOMENTUM_DROP", "Momentum turning down (early warning)", "medium", "momentum",
            f"Last 6 months: sales {_signed(g6)}% YoY, {_signed(gvm6)} pp vs market — worse than the 12-month view.",
            "Weekly pipeline review (enquiries, bookings, cancellations) for the next 6 weeks.",
            "Corrective", "Area Sales Manager", "30 days")
    if market_dip:
        add("MARKET_DIP", "Market-driven slowdown", "info", "sales",
            f"Local car market {_signed(mg)}% while the dealer held share (share {_signed(dsh)} pp).",
            "Support, don't penalise: revise targets, extend retail support scheme, avoid pushing stock.",
            "Support", "Regional Manager", "30 days")
    inv, aged, ich = r.get("inventory_days"), r.get("aged_stock_pct"), r.get("inventory_change")
    building = _ok(ich) and ich >= 25 and _ok(inv) and inv > 55
    if (_ok(inv) and inv > 65) or (_ok(aged) and aged > 30) or building:
        sev = "high" if (_ok(inv) and inv > 80) or (_ok(aged) and aged > 40) else "medium"
        add("INVENTORY", "Inventory overload" if not (building and inv <= 65) else "Inventory building up fast",
            sev, "financial",
            f"{_f(inv)} days of stock ({_signed(ich, 0)} days in 3 months), {_f(aged)}% older than 60 days "
            f"(≈ {_f(r.get('excess_stock_units'))} units above a {NORM_STOCK_DAYS}-day norm).",
            f"Pause dispatches for ~{_f(r.get('excess_stock_units'))} units, transfer aged stock to faster-moving "
            f"dealers, run a targeted ageing-stock scheme.",
            "Support", "Regional Sales Planning", "30 days")
    pdd, pdc = r.get("payment_delay_days"), r.get("payment_delay_change")
    if (_ok(pdd) and pdd > 18) or (_ok(pdc) and pdc >= 8 and _ok(pdd) and pdd > 10):
        sev = "high" if _ok(pdd) and pdd > 25 else "medium"
        linked = " Stress looks stock-driven — fix inventory first." if any(f["code"] == "INVENTORY" for f in fl) else ""
        add("PAYMENT", "Payment delays", sev, "financial",
            f"Dues {_f(pdd)} days past due on average ({_signed(pdc)} days in 3 months).{linked}",
            "Credit review with the dealer principal and financier; link dispatches to clearing overdues; "
            "consider restructuring inventory funding.",
            "Corrective", "Regional Finance", "15 days")
    csi, csc = r.get("csi_score"), r.get("csi_change")
    if (_ok(csi) and csi < 76) or (_ok(csc) and csc <= -4):
        sev = "high" if (_ok(csi) and csi < 72) or (_ok(csc) and csc <= -7) else "medium"
        add("SERVICE", "Service quality slipping", sev, "customer",
            f"CSI {_f(csi, 1)} ({_signed(csc)} pts in 3 months).",
            "Workshop audit; check technician / service-advisor attrition; refresher training and a CSI recovery plan.",
            "Corrective", "Regional Service Manager", "30 days")
    cp = r.get("complaints_per_100_units")
    if _ok(cp) and cp > 2.5:
        sev = "high" if cp > 3.5 else "medium"
        add("COMPLAINTS", "High customer complaints", sev, "customer",
            f"{_f(cp, 1)} complaints per 100 units sold (network median ≈ 1).",
            "Root-cause the top complaint categories with Customer Care; daily callback on open complaints.",
            "Corrective", "Customer Care Lead", "15 days")
    gap, head = r.get("share_gap_pp"), r.get("headroom_units")
    if _ok(gap) and gap <= -5 and _ok(head) and head >= 150 and _ok(mg) and mg > 0:
        add("UNTAPPED", "Untapped market potential", "opportunity", "sales",
            f"Share {_f(r.get('share_l12m_pct'), 1)}% vs {_f(r.get('benchmark_share_pct'), 1)}% expected for this "
            f"market type — ≈ {_f(head)} units/yr (₹{_f(r.get('headroom_cr'), 0)} cr) of headroom in a growing market.",
            "Growth plan: additional outlet or rural sales points, fleet and corporate tie-ups, local activation.",
            "Grow", "Network Development", "90 days")
    return fl


SEV_RANK = {"high": 0, "medium": 1, "info": 2, "opportunity": 3}
PILLAR_RANK = {"financial": 0, "sales": 1, "customer": 2, "momentum": 3}
DIAGNOSIS = {
    "PAYMENT": "Financial stress", "INVENTORY": "Financial stress", "SHARE_LOSS": "Losing to competition",
    "LAG_MARKET": "Losing to competition", "BELOW_TARGET": "Below target", "MOMENTUM_DROP": "Slowing momentum",
    "SERVICE": "Service & CX issue", "COMPLAINTS": "Service & CX issue", "MARKET_DIP": "Market-driven dip",
    "UNTAPPED": "Untapped potential",
}


SALES_CODES = {"BELOW_TARGET", "SHARE_LOSS", "LAG_MARKET", "MOMENTUM_DROP"}
FIN_CODES = {"INVENTORY", "PAYMENT"}
ROOT_CAUSE = "Sales slump → stock stress"
WEAKEST_PILLAR_ISSUE = {"sales": "Below target", "momentum": "Slowing momentum", "financial": "Financial stress",
                        "customer": "Service & CX issue"}


def _risk(flags):
    return sorted([f for f in flags if f["severity"] in ("high", "medium")],
                  key=lambda x: (SEV_RANK[x["severity"]], PILLAR_RANK[x["pillar"]]))


def primary_issue(flags: list[dict], score: float = float("nan"), pillars: dict | None = None) -> str:
    """Main issue = root cause, not symptom: when weak sales and stock/credit stress co-occur, stock piles up
    because dispatches follow target while retail falls short - so the sales problem is named first."""
    risk = _risk(flags)
    codes = {f["code"] for f in risk}
    if codes & SALES_CODES and codes & FIN_CODES:
        return ROOT_CAUSE
    if risk:
        top = risk[0]
        return WEAKEST_PILLAR_ISSUE[top["pillar"]] if top["code"] == "WEAK_PILLAR" else DIAGNOSIS[top["code"]]
    if any(f["code"] == "MARKET_DIP" for f in flags):
        return "Market-driven dip"
    if _ok(score) and score < TIER_CUTS[1] and pillars:               # low score without a single alert
        valid = {k: v for k, v in pillars.items() if _ok(v)}
        if valid:
            return WEAKEST_PILLAR_ISSUE[min(valid, key=valid.get)]
    if any(f["code"] == "UNTAPPED" for f in flags):
        return "Untapped potential"
    return "On track"


def ordered_alerts(flags: list[dict], issue: str | None = None) -> list[dict]:
    """Alerts in the order a manager should act: root cause first when sales weakness drives stock/credit stress."""
    key = lambda x: (SEV_RANK[x["severity"]], PILLAR_RANK[x["pillar"]])
    if issue == ROOT_CAUSE:
        key = lambda x: (SEV_RANK[x["severity"]] if x["severity"] in ("info", "opportunity") else 0,
                         0 if x["code"] in SALES_CODES else 1 if x["code"] in FIN_CODES else 2,
                         SEV_RANK[x["severity"]], PILLAR_RANK[x["pillar"]])
    return sorted(flags, key=key)


def tier_from(score: float, flags: list[dict]) -> str:
    if not _ok(score):
        return "Watch"
    base = 0 if score < TIER_CUTS[0] else 1 if score < TIER_CUTS[1] else 2 if score < TIER_CUTS[2] else 3
    n_high = sum(f["severity"] == "high" for f in flags)
    n_med = sum(f["severity"] == "medium" for f in flags)
    if n_high >= 2:
        base = min(base, 0)
    elif n_high == 1:
        base = min(base, 1)
    elif n_med >= 2:
        base = min(base, 2)
    return TIERS[base]


# ----------------------------------------------------------------------------- scoring
def score_dealers(raw: pd.DataFrame, weights: dict | None = None, volume_ref: pd.Series | None = None) -> pd.DataFrame:
    """Score a table of raw dealer inputs. Returns one row per dealer with pillar scores, health, tier,
    flags, diagnosis, actions and priority. `volume_ref` (units_l12m of the reference network) keeps the
    business-impact percentile stable when the user filters the view."""
    w = normalise_weights(weights)
    m = derive_metrics(raw)
    for rule in RULES:
        m[f"s_{rule.metric}"] = rule_score(m[rule.metric], rule)
    for p in PILLARS:
        rs = [r for r in RULES if r.pillar == p]
        vals = np.column_stack([m[f"s_{r.metric}"].to_numpy() for r in rs])
        wts = np.array([r.weight for r in rs])
        avail = ~np.isnan(vals)
        num = np.nansum(np.where(avail, vals, 0) * wts, axis=1)
        den = (avail * wts).sum(axis=1)
        m[f"pillar_{p}"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        m[f"coverage_{p}"] = den / wts.sum()
    pv = np.column_stack([m[f"pillar_{p}"].to_numpy() for p in PILLARS])
    pw = np.array([w[p] for p in PILLARS])
    av = ~np.isnan(pv)
    den = (av * pw).sum(axis=1)
    m["health_score"] = np.where(den > 0, np.nansum(np.where(av, pv, 0) * pw, axis=1) / np.where(den > 0, den, 1), np.nan)
    cov = np.column_stack([m[f"coverage_{p}"].to_numpy(float) for p in PILLARS])
    m["data_coverage"] = (cov * pw).sum(axis=1)          # explicit sum: avoids spurious BLAS FP warnings
    flags = [dealer_flags(r) for _, r in m.iterrows()]
    m["alerts"] = flags
    m["n_high"] = [sum(f["severity"] == "high" for f in fl) for fl in flags]
    m["n_flags"] = [sum(f["severity"] in ("high", "medium") for f in fl) for fl in flags]
    m["tier"] = [tier_from(s, fl) for s, fl in zip(m.health_score, flags)]
    pil = [{p: r[f"pillar_{p}"] for p in PILLARS} for _, r in m.iterrows()]
    # a low overall score can come from several metrics each just short of their red line: no single alert fires,
    # but a dealer in an attention tier must still get a reason and an action
    for fl, tier, pp in zip(flags, m.tier, pil):
        if tier in ("Critical", "Watch") and not _risk(fl):
            valid = {k: v for k, v in pp.items() if _ok(v)}
            if valid:
                weak = min(valid, key=valid.get)
                fl.append(dict(code="WEAK_PILLAR", title=f"Weak {PILLARS[weak].lower()}", severity="medium",
                               pillar=weak, evidence=f"{PILLARS[weak]} scores {valid[weak]:.0f}/100 — several metrics "
                               f"below norm, none past its red line yet.",
                               action=f"Joint performance review of {PILLARS[weak].lower()}: agree 2–3 improvement "
                                      f"targets for the next 60 days.",
                               action_type="Corrective", owner="Area Sales Manager", horizon="60 days"))
    m["n_flags"] = [sum(f["severity"] in ("high", "medium") for f in fl) for fl in flags]
    m["primary_issue"] = [primary_issue(fl, sc, pp) for fl, sc, pp in zip(flags, m.health_score, pil)]
    m["action_type"] = [_action_type(fl, iss) for fl, iss in zip(flags, m.primary_issue)]
    ref = volume_ref if volume_ref is not None and len(volume_ref) > 0 else m.units_l12m
    ref = np.sort(np.asarray(ref, float))
    m["volume_pct"] = np.searchsorted(ref, m.units_l12m.to_numpy(float), side="right") / len(ref)
    risk = 100 - m.health_score.fillna(50)
    m["priority_index"] = (risk * (0.6 + 0.4 * m.volume_pct) + 8 * m.n_high).round(1)
    m["priority_rank"] = m.priority_index.rank(ascending=False, method="first").astype(int)
    m["tier_rank"] = m.tier.map({t: i for i, t in enumerate(TIERS)})
    return m.sort_values("priority_rank").reset_index(drop=True)


def _action_type(flags: list[dict], issue: str | None = None) -> str:
    risk = [f for f in flags if f["severity"] in ("high", "medium", "info")]
    if not risk:
        return "Grow" if any(f["code"] == "UNTAPPED" for f in flags) else "Maintain"
    return ordered_alerts(risk, issue)[0]["action_type"]


def summary_sentence(r: pd.Series) -> str:
    """One-line explanation for a manager."""
    name = r.get("territory") or r.get("dealer_name") or r.get("office_name") or r.get("dealer_id")
    risk = _risk(r["alerts"])
    base = f"{name} is **{r['tier']}** with a health score of **{r['health_score']:.0f}/100**."
    if not risk:
        info = [f for f in r["alerts"] if f["severity"] in ("info", "opportunity")]
        if info:
            return base + f" No risk alerts; note: {info[0]['title'].lower()} — {info[0]['evidence']}"
        return base + " No alerts: performing in line with its market."
    sales = [f for f in risk if f["code"] in SALES_CODES]
    fin = [f for f in risk if f["code"] in FIN_CODES]
    if sales and fin:
        return base + (f" Root cause: **{sales[0]['title'].lower()}** — {sales[0]['evidence']} "
                       f"Knock-on effect: **{fin[0]['title'].lower()}** — {fin[0]['evidence']}")
    lead = risk[0]
    tail = f" Also: {risk[1]['title'].lower()}." if len(risk) > 1 else ""
    return base + f" Main issue: **{lead['title'].lower()}** — {lead['evidence']}{tail}"


def score_history(history_raw: pd.DataFrame, weights: dict | None = None, volume_ref=None) -> pd.DataFrame:
    """Score every snapshot month; returns dealer_id x as_of health + tier (for trends and 'deteriorating')."""
    out = []
    for t, g in history_raw.groupby("as_of"):
        s = score_dealers(g, weights, volume_ref)
        out.append(s[["dealer_id", "health_score", "tier"]].assign(as_of=t))
    return pd.concat(out, ignore_index=True)


def top_action(alerts: list[dict], issue: str | None) -> str:
    """The first action a manager should take (same order as the deep-dive)."""
    acts = [a for a in ordered_alerts(alerts, issue) if a["severity"] != "info" or a["code"] == "MARKET_DIP"]
    return acts[0]["action"] if acts else "Maintain: standard monthly review."


MAX_UPLOAD_ROWS = 5000


def score_upload(name: str, data: bytes, weights: dict | None = None, volume_ref=None,
                 max_rows: int = MAX_UPLOAD_ROWS) -> tuple[pd.DataFrame, list[str]]:
    """Uploaded file -> (manager-ready results table, warnings). Raises InputError for anything unusable."""
    try:
        raw = read_table(name, data)
    except InputError:
        raise
    except Exception as e:                                   # corrupted / unexpected file
        raise InputError(f"Could not read the file ({type(e).__name__}). Please use the CSV or Excel template.")
    if len(raw) > max_rows:
        raise InputError(f"File has {len(raw):,} rows; the prototype accepts up to {max_rows:,}.")
    clean, warns = validate_inputs(raw)
    res = score_dealers(clean, weights, volume_ref)

    def text(col):
        s = res[col].astype(object).where(res[col].notna(), None)
        return s.map(lambda v: None if v is None or str(v).strip() == "" else str(v).strip())

    name_col = text("dealer_name").fillna(text("territory")).fillna(text("office_name")).fillna(res.dealer_id)
    out = pd.DataFrame({
        "Priority": res.priority_rank, "Dealer": name_col, "Tier": res.tier, "Health": res.health_score.round(0),
        "Main issue": res.primary_issue, "Action": res.action_type,
        "Top recommended action": [top_action(fl, iss) for fl, iss in zip(res.alerts, res.primary_issue)],
        "Data coverage": (res.data_coverage * 100).round(0),
    })
    return out, warns


def template_frame() -> pd.DataFrame:
    """Example rows for the upload template / manual form defaults."""
    return pd.DataFrame([
        dict(dealer_id="DLR-001", dealer_name="Example Motors", state_name="Rajasthan", units_l12m=1450,
             units_p12m=1500, target_l12m=1700, car_market_l12m=4200, car_market_p12m=3900, benchmark_share_pct=38,
             units_last6m=650, units_last6m_prev_year=760, car_market_last6m=2050, car_market_last6m_prev_year=1950,
             units_last3m=300, target_last3m=420, inventory_days=78, inventory_days_3m_ago=55, closing_stock=310, aged_stock_pct=31,
             payment_delay_days=22, payment_delay_days_3m_ago=12, csi_score=79, csi_score_3m_ago=83,
             complaints_per_100_units=1.8, top_competitor_gainer="Mahindra", top_competitor_gain_pp=2.5),
        dict(dealer_id="DLR-002", dealer_name="Sample Autos", state_name="Karnataka", units_l12m=2600,
             units_p12m=2300, target_l12m=2500, car_market_l12m=7000, car_market_p12m=6500, benchmark_share_pct=34,
             units_last6m=1350, units_last6m_prev_year=1150, car_market_last6m=3550, car_market_last6m_prev_year=3300,
             units_last3m=640, target_last3m=610, inventory_days=38, inventory_days_3m_ago=41, closing_stock=270, aged_stock_pct=6,
             payment_delay_days=4, payment_delay_days_3m_ago=5, csi_score=87, csi_score_3m_ago=86,
             complaints_per_100_units=0.8, top_competitor_gainer="Tata Motors", top_competitor_gain_pp=0.9),
    ])
