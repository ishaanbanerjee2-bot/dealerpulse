"""
Bring-your-own-data: turn a company's MONTHLY dealer file into the same inputs the app builds from VAHAN.

One row per dealer per month. Required columns:
    dealer_id, month, units_sold, target_units, local_car_market
Optional columns (each one switches on more of the app):
    dealer_name, state, zone                         -> names and the zone / state filters
    expected_share_pct                               -> your own share benchmark (else the state / zone average)
    inventory_days, aged_stock_pct, payment_delay_days, csi_score, complaints
                                                     -> inventory & payments and service pillars
    comp_<Brand> (e.g. comp_Hyundai)                 -> competitor units in the dealer's market (competition view)

At least 24 months of history are needed: the scores compare the latest 12 months with the 12 before.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import engine as E

M_REQUIRED = ["dealer_id", "month", "units_sold", "target_units", "local_car_market"]
M_TEXT = ["dealer_name", "state", "zone"]
M_OPS = ["inventory_days", "aged_stock_pct", "payment_delay_days", "csi_score", "complaints"]
M_NUMERIC = ["units_sold", "target_units", "local_car_market", "expected_share_pct"] + M_OPS
MIN_MONTHS, MAX_ROWS, MAX_DEALERS, SNAPSHOTS = 24, 250_000, 5_000, 12


def _norm(c: str) -> str:
    c = str(c).strip()
    if c.lower().startswith("comp_"):
        return "comp_" + c[5:].strip()                   # keep the brand's own capitalisation
    return re.sub(r"[\s\-]+", "_", c.lower())


def _brand(col: str) -> str:
    return col[5:].replace("_", " ").strip() or col


def _parse_month(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        if v.dropna().between(190001, 210012).all():     # 202405 style
            return pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m", errors="coerce")
        return pd.Series(pd.NaT, index=s.index)
    d = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", format="mixed", dayfirst=False)
    return d.dt.to_period("M").dt.to_timestamp()


def build_dataset(name: str, data: bytes) -> dict:
    """Uploaded monthly file -> {'hist', 'panel', 'targets', 'ops', 'meta'}. Raises E.InputError."""
    raw = E.read_table(name, data)
    if len(raw) > MAX_ROWS:
        raise E.InputError(f"File has {len(raw):,} rows; the prototype accepts up to {MAX_ROWS:,}.")
    df = raw.copy()
    df.columns = [_norm(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise E.InputError("Duplicate column names: " + ", ".join(df.columns[df.columns.duplicated()].unique()))
    missing = [c for c in M_REQUIRED if c not in df.columns]
    if missing:
        raise E.InputError("Missing required column(s): " + ", ".join(missing) +
                           ". Download the template to see the expected layout.")
    warnings: list[str] = []
    df["dealer_id"] = df["dealer_id"].astype(str).str.strip()
    df = df[~df.dealer_id.isin(["", "nan", "None"])]
    df["month"] = _parse_month(df["month"])
    if df.month.isna().any():
        bad = raw.loc[df.index[df.month.isna()], raw.columns[list(df.columns).index("month")]].astype(str).head(3)
        raise E.InputError("Could not read some 'month' values (e.g. " + ", ".join(bad) + "). Use YYYY-MM.")
    comps = [c for c in df.columns if c.startswith("comp_")]
    for c in M_NUMERIC + comps:
        if c not in df.columns:
            continue
        txt = df[c].astype(str).str.replace(",", "").str.replace("%", "").str.strip()
        v = pd.to_numeric(txt.where(~txt.isin(["", "nan", "None"])), errors="coerce")
        n_bad = int((v.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")).sum())
        n_neg = int((v < 0).sum())
        if n_bad:
            warnings.append(f"'{c}': {n_bad} non-numeric value(s) treated as missing.")
        if n_neg:
            warnings.append(f"'{c}': {n_neg} negative value(s) treated as missing.")
        df[c] = v.where(v >= 0).astype(float)
    if df.duplicated(["dealer_id", "month"]).any():
        d = df.loc[df.duplicated(["dealer_id", "month"]), ["dealer_id", "month"]].head(3)
        raise E.InputError("Each dealer can appear once per month; duplicates found, e.g. " +
                           ", ".join(f"{a} {b:%Y-%m}" for a, b in d.itertuples(index=False)))
    over = df.units_sold > df.local_car_market
    if over.any():
        raise E.InputError(f"{int(over.sum())} row(s) have units_sold above local_car_market "
                           "(the local market must include the dealer's own sales).")
    if (df.aged_stock_pct > 100).any() if "aged_stock_pct" in df else False:
        raise E.InputError("'aged_stock_pct' must be a percentage between 0 and 100.")
    if (df.csi_score > 100).any() if "csi_score" in df else False:
        raise E.InputError("'csi_score' must be on a 0-100 scale.")
    if (df.expected_share_pct > 100).any() if "expected_share_pct" in df else False:
        raise E.InputError("'expected_share_pct' must be a percentage between 0 and 100.")
    if df.dealer_id.nunique() > MAX_DEALERS:
        raise E.InputError(f"File has {df.dealer_id.nunique():,} dealers; the prototype accepts up to {MAX_DEALERS:,}.")

    months = pd.date_range(df.month.min(), df.month.max(), freq="MS")
    if len(months) < MIN_MONTHS:
        raise E.InputError(f"The file covers {len(months)} month(s); at least {MIN_MONTHS} are needed "
                           "(the latest 12 months are compared with the 12 before).")
    for c in M_TEXT:
        if c not in df.columns:
            df[c] = None
    info = df.sort_values("month").groupby("dealer_id")[M_TEXT].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else None)
    info["dealer_name"] = info.dealer_name.where(info.dealer_name.notna() & (info.dealer_name.astype(str).str.strip() != ""),
                                                 info.index.to_series())
    info["state"] = info.state.fillna("All")
    info["zone"] = info.zone.fillna("All")
    ids = sorted(df.dealer_id.unique())

    def piv(col):
        if col not in df.columns:
            return None
        return df.pivot(index="month", columns="dealer_id", values=col).reindex(index=months, columns=ids)

    U, T, CM = piv("units_sold"), piv("target_units"), piv("local_car_market")
    ops = {c: piv(c) for c in M_OPS}
    comp = {c: piv(c) for c in comps}
    exp_share = piv("expected_share_pct")

    def wsum(p, end, n, lag=0):                      # NaN unless the whole window is present
        e = end - pd.DateOffset(months=lag)
        w = p.loc[(p.index > e - pd.DateOffset(months=n)) & (p.index <= e)]
        return w.sum(min_count=n) if len(w) == n else pd.Series(np.nan, index=p.columns)

    def wmean(p, end, n, lag=0):                     # 3-month averages, 1 decimal (same convention as the pipeline)
        e = end - pd.DateOffset(months=lag)
        w = p.loc[(p.index > e - pd.DateOffset(months=n)) & (p.index <= e)]
        return w.mean().round(1)

    snaps = months[MIN_MONTHS - 1:][-SNAPSHOTS:]
    rows = []
    for t in snaps:
        f = pd.DataFrame(index=pd.Index(ids, name="dealer_id"))
        f["as_of"] = t
        f["zone"], f["state_name"] = info.zone, info.state
        f["office_code"], f["office_name"], f["territory"] = f.index, info.dealer_name, info.dealer_name
        f["units_l12m"], f["units_p12m"] = wsum(U, t, 12), wsum(U, t, 12, 12)
        f["target_l12m"] = wsum(T, t, 12)
        f["car_market_l12m"], f["car_market_p12m"] = wsum(CM, t, 12), wsum(CM, t, 12, 12)
        f["units_last6m"], f["units_last6m_prev_year"] = wsum(U, t, 6), wsum(U, t, 6, 12)
        f["car_market_last6m"], f["car_market_last6m_prev_year"] = wsum(CM, t, 6), wsum(CM, t, 6, 12)
        f["units_last3m"], f["target_last3m"] = wsum(U, t, 3), wsum(T, t, 3)
        if ops["inventory_days"] is not None:
            f["inventory_days"] = wmean(ops["inventory_days"], t, 3)
            f["inventory_days_3m_ago"] = wmean(ops["inventory_days"], t, 3, 3)
        if ops["aged_stock_pct"] is not None:
            f["aged_stock_pct"] = wmean(ops["aged_stock_pct"], t, 3)
        if ops["payment_delay_days"] is not None:
            f["payment_delay_days"] = wmean(ops["payment_delay_days"], t, 3)
            f["payment_delay_days_3m_ago"] = wmean(ops["payment_delay_days"], t, 3, 3)
        if ops["csi_score"] is not None:
            f["csi_score"] = wmean(ops["csi_score"], t, 3)
            f["csi_score_3m_ago"] = wmean(ops["csi_score"], t, 3, 3)
        if ops["complaints"] is not None:
            u3 = wsum(U, t, 3)
            f["complaints_per_100_units"] = np.round(E._safe_div(wsum(ops["complaints"], t, 3), u3.where(u3 > 0)) * 100, 2)
        # share benchmark: your own expected share if supplied, else the state (or zone) average share
        if exp_share is not None:
            f["benchmark_share_pct"] = exp_share.loc[:t].ffill().iloc[-1]
        else:
            ok = f.units_l12m.notna() & f.car_market_l12m.notna()
            for key in ("state_name", "zone"):
                g = f[ok].groupby(key)
                avg = (g.units_l12m.sum() / g.car_market_l12m.sum() * 100)
                fill = f[key].map(avg)
                f["benchmark_share_pct"] = fill if "benchmark_share_pct" not in f else f.benchmark_share_pct.fillna(fill)
            f["benchmark_share_pct"] = f.benchmark_share_pct.fillna(f.units_l12m[ok].sum() / f.car_market_l12m[ok].sum() * 100)
        if comp:
            ml, mp = f.car_market_l12m, f.car_market_p12m
            gains = pd.DataFrame({_brand(c): (wsum(p, t, 12) / ml - wsum(p, t, 12, 12) / mp) * 100 for c, p in comp.items()})
            has = gains.notna().any(axis=1)
            f["top_competitor_gainer"] = gains.idxmax(axis=1).where(has)
            f["top_competitor_gain_pp"] = gains.max(axis=1).round(2)
        rows.append(f.reset_index())
    hist = pd.concat(rows, ignore_index=True)
    complete = hist[E.REQUIRED[1:]].notna().all(axis=1) & (hist.target_l12m > 0) & (hist.car_market_l12m > 0) \
        & (hist.car_market_p12m > 0)
    hist = hist[complete].copy()
    if hist.empty:
        raise E.InputError("No dealer has 24 consecutive months of units, target and local market data — "
                           "check for gaps in the history.")
    last = hist.as_of.max()
    dropped = sorted(set(ids) - set(hist.loc[hist.as_of == last, "dealer_id"]))
    if dropped:
        warnings.append(f"{len(dropped)} dealer(s) skipped for the latest month because their last 24 months of units, "
                        f"target or market data have gaps (e.g. {', '.join(dropped[:3])}).")
    for c in E.OPTIONAL_NUMERIC:
        if c not in hist:
            hist[c] = np.nan
    for c in ("top_competitor_gainer",):
        if c not in hist:
            hist[c] = None
    # tables for charts
    long = df.rename(columns={"month": "date", "units_sold": "oem_units", "local_car_market": "car_market"})
    long = long.merge(info[["zone", "state", "dealer_name"]].rename(columns={"state": "state_name", "dealer_name": "territory"}),
                      left_on="dealer_id", right_index=True, how="left", suffixes=("_raw", ""))
    panel = long[["dealer_id", "date", "oem_units", "car_market", "zone", "state_name", "territory"] + comps] \
        .rename(columns={c: _brand(c) for c in comps}).sort_values(["dealer_id", "date"]).reset_index(drop=True)
    targets = long[["dealer_id", "date", "target_units"]].copy()
    present_ops = [c for c in M_OPS if c in df.columns and df[c].notna().any()]
    ops_tab = None
    if present_ops:
        ops_tab = long[["dealer_id", "date"] + present_ops].copy()
        for c in M_OPS:
            if c not in ops_tab:
                ops_tab[c] = np.nan
        ops_tab["retail_units"] = long.oem_units
    meta = {
        "source": name, "dealers": int(hist.loc[hist.as_of == last, "dealer_id"].nunique()),
        "months": len(months), "first": months[0], "last": months[-1],
        "ops_columns": present_ops, "competitors": [_brand(c) for c in comps],
        "benchmark": "your expected share" if exp_share is not None else "state / zone average share",
        "warnings": warnings,
    }
    return {"hist": hist.reset_index(drop=True), "panel": panel, "targets": targets, "ops": ops_tab, "meta": meta}


def template_monthly() -> pd.DataFrame:
    """Blank-ish template: 2 dealers x 24 months, showing every column."""
    months = pd.date_range("2022-06-01", periods=24, freq="MS")
    rows = []
    rng = np.random.default_rng(1)
    for d, name, st, zn, base in [("D001", "City Motors", "Maharashtra", "West", 120), ("D002", "Highway Autos", "Gujarat", "West", 80)]:
        for i, m in enumerate(months):
            u = int(base * (1 + 0.1 * np.sin(i / 2)) + rng.integers(-8, 8))
            rows.append(dict(dealer_id=d, dealer_name=name, state=st, zone=zn, month=m.strftime("%Y-%m"),
                             units_sold=u, target_units=int(base * 1.05), local_car_market=int(u / 0.38),
                             expected_share_pct=38.0, inventory_days=40 + i % 10, aged_stock_pct=8,
                             payment_delay_days=5, csi_score=84, complaints=1, comp_Hyundai=int(u * 0.45),
                             comp_Tata_Motors=int(u * 0.35)))
    return pd.DataFrame(rows)
