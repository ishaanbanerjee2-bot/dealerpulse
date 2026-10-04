"""
Step 5 - Build everything the web app needs (data/app/).

  1. Modelled monthly TARGETS (real-data derived): prior-FY actual x (1 + zone market growth, clipped 3-12%),
     phased by the zone's pre-Covid (2019) seasonality.
  2. SYNTHETIC operations layer, mechanistically linked to real retail sales:
       inventory  : OEM dispatches to target, dealer retails actual VAHAN sales -> stock builds where retail < target
       payments   : days past due rise with inventory stress + sales shortfall, offset by latent financial resilience
       service    : CSI / complaints driven by latent service quality (mildly linked to real share trend) + stock stress
       events     : ~11% of dealers get an injected emerging problem in Feb-May 2024 (hidden ground truth for testing)
  3. EXPECTED SHARE benchmark: out-of-fold gradient boosting on market structure (state, car penetration,
     2W / tractor mix, market size, growth) - a fairer benchmark than the plain state average.
  4. Dealer x month INPUT table for the last 12 snapshot months (Jun-2023..May-2024), in the same raw schema the
     app accepts from an uploaded CSV, so sample data, uploads and manual inputs run through one engine.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "data/app"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20240531)
COMPETITORS = ["Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW", "Renault"]

panel = pd.read_csv(P / "master_panel.csv", parse_dates=["date"])
dealers = pd.read_csv(P / "master_dealers.csv")
panel["fy"] = panel.date.dt.year + (panel.date.dt.month >= 4).astype(int)       # FY24 = Apr-23..Mar-24
ids = sorted(panel.dealer_id.unique())
months = sorted(panel.date.unique())
U = panel.pivot(index="date", columns="dealer_id", values="oem_units")[ids].astype(float)   # month x dealer
zone_of = panel.groupby("dealer_id").zone.first()[ids]

# ------------------------------------------------------------------ 1. targets
fy_units = panel.groupby(["dealer_id", "fy"]).oem_units.sum().unstack()
zone_mkt = panel.groupby(["zone", "fy"]).car_market.sum().unstack()
zone_growth = (zone_mkt / zone_mkt.shift(1, axis=1) - 1)
c19 = panel[panel.date.dt.year == 2019]
season = c19.groupby(["zone", c19.date.dt.month]).oem_units.sum().unstack()
season = season.div(season.sum(axis=1), axis=0)                               # zone x calendar-month weights
sim_months = [m for m in months if m >= pd.Timestamp("2022-04-01")]
T = pd.DataFrame(index=sim_months, columns=ids, dtype=float)
for m in sim_months:
    fy = m.year + (m.month >= 4)
    g = zone_growth[fy - 1].clip(0.03, 0.12).reindex(zone_of).to_numpy()      # growth seen when target was set
    annual = fy_units[fy - 1].reindex(ids).to_numpy() * (1 + g)
    T.loc[m] = annual * season.reindex(zone_of.to_numpy())[m.month].to_numpy()
T = T.round()

# ------------------------------------------------------------------ 2. synthetic operations
n, M = len(ids), len(sim_months)
R = U.loc[sim_months].to_numpy()
Tm = T.to_numpy()
alpha = rng.uniform(0.25, 0.45, n)                  # how fast the OEM corrects dispatches for excess stock
resil = rng.normal(0, 1, n)                         # latent financial resilience
# latent service quality, mildly linked to the dealer's real share trend (service drives loyalty)
share_trend = dealers.set_index("dealer_id").share_trend_pp_per_yr.reindex(ids).fillna(0).to_numpy()
z_st = (share_trend - share_trend.mean()) / share_trend.std()
quality = 0.35 * z_st + np.sqrt(1 - 0.35 ** 2) * rng.normal(0, 1, n)

# injected emerging events (Feb-May 2024), disjoint dealer sets
perm = rng.permutation(n)
ev_workshop, ev_credit, ev_push = perm[:37], perm[37:74], perm[74:102]
event = np.array([""] * n, dtype=object)
event[ev_workshop], event[ev_credit], event[ev_push] = "service_disruption", "credit_squeeze", "stock_push"
ev_start = sim_months.index(pd.Timestamp("2024-02-01"))
ramp = np.zeros(M); ramp[ev_start:] = np.linspace(0.4, 1.0, M - ev_start)

stock = np.zeros((M, n)); W = np.zeros((M, n)); doi = np.zeros((M, n)); aged = np.zeros((M, n))
dpd = np.zeros((M, n)); csi = np.zeros((M, n)); compl = np.zeros((M, n))
fy22_avg = U.loc[(U.index >= "2021-04-01") & (U.index <= "2022-03-01")].mean().to_numpy()
s_prev = 35 / 30.4 * fy22_avg * rng.lognormal(0, 0.2, n)
e_dpd, e_csi = np.zeros(n), np.zeros(n)
for t in range(M):
    push = np.ones(n)
    if t >= ev_start:
        push[ev_push] = 1.5          # material dispatch push beyond what the dealer can retail
    norm = 30 / 30.4 * Tm[t]
    w = np.maximum(Tm[t] * (1 + rng.normal(0, 0.06, n)) * push - alpha * (s_prev - norm), 0)
    # beyond ~75 days of cover dealers refuse / OEM holds billing: dispatches taper to 20%
    prev_doi = doi[t - 1] if t > 0 else np.full(n, 35.0)
    w = w * np.clip(1 - (prev_doi - 75) / 60, 0.2, 1)
    s = np.maximum(s_prev + w - R[t], 0)
    r3 = R[max(0, t - 2): t + 1].mean(axis=0)
    run_rate = np.maximum(r3, 0.5 * fy22_avg)          # robust to a freak low-retail month
    d = np.minimum(s / np.maximum(run_rate, 1) * 30.4, 240)
    a = np.clip(0.03 + 0.75 * np.maximum(0, d - 40) / np.maximum(d, 1) + rng.normal(0, 0.015, n), 0, 0.85)
    short = np.maximum(0, 1 - R[max(0, t - 2): t + 1].sum(0) / np.maximum(Tm[max(0, t - 2): t + 1].sum(0), 1))
    stress = np.clip((d - 45) / 30, 0, 2)
    e_dpd = 0.6 * e_dpd + rng.normal(0, 2.0, n)
    p_ = 3 + 10 * stress + 15 * short - 3 * resil + e_dpd
    p_[ev_credit] += ramp[t] * rng.uniform(14, 24, len(ev_credit))
    e_csi = 0.6 * e_csi + rng.normal(0, 1.2, n)
    c_ = 84 + 4.5 * quality - 2.5 * stress + e_csi
    c_[ev_workshop] -= ramp[t] * rng.uniform(7, 11, len(ev_workshop))
    lam = R[t] * 0.009 * np.exp(-0.35 * quality + 0.25 * stress)
    lam[ev_workshop] *= 1 + 1.4 * ramp[t]
    stock[t], W[t], doi[t], aged[t] = s, w, d, a
    dpd[t], csi[t], compl[t] = np.clip(p_, 0, 90), np.clip(c_, 55, 98), rng.poisson(lam)
    s_prev = s

ops = pd.DataFrame({
    "dealer_id": np.tile(ids, M), "date": np.repeat(sim_months, n),
    "target_units": Tm.ravel(), "wholesale_units": W.ravel().round(), "retail_units": R.ravel(),
    "closing_stock": stock.ravel().round(), "inventory_days": doi.ravel().round(1),
    "aged_stock_pct": (aged.ravel() * 100).round(1), "payment_delay_days": dpd.ravel().round(1),
    "csi_score": csi.ravel().round(1), "complaints": compl.ravel()})
ops.to_parquet(OUT / "ops_monthly.parquet", index=False)
pd.DataFrame({"dealer_id": ids, "injected_event": event}).to_csv(OUT / "ground_truth_events.csv", index=False)

# ------------------------------------------------------------------ 3. expected share (out-of-fold)
v = pd.read_parquet(P / "vahan_clean.parquet")
l = v[(v.date > "2023-05-01") & v.office_code.isin(dealers.office_code)]
TW = (r"Hero Motocorp|Honda Motorcycle|Tvs Motor|Bajaj Auto|Suzuki Motorcycle|Royal-Enfield|Yamaha|Ola Electric|"
      r"Ather|Greaves|Classic Legends|Okinawa|Bgauss|Lectrix|Wardwizard")
TR = r"Tractor|Swaraj|International Tractors|Tafe|Escorts|John Deere|Kubota|Cnh|Sonalika"
l = l.assign(seg=np.select([l.maker.str.contains(TW), l.maker.str.contains(TR), l.maker_group != "Other"],
                           ["2w", "tractor", "car"], "other"))
seg = l.pivot_table(index="office_code", columns="seg", values="registrations", aggfunc="sum").fillna(0)
tot = seg.sum(axis=1)
X = dealers.set_index("office_code")[["dealer_id", "state_name", "units_l12m", "car_market_l12m", "market_growth_yoy",
                                      "share_l12m"]].copy()
X["car_pen"] = (seg.car / tot).reindex(X.index)
X["tw_sh"] = (seg["2w"] / tot).reindex(X.index)
X["tractor_sh"] = (seg.tractor / tot).reindex(X.index)
X["log_car"] = np.log(X.car_market_l12m)
X = X.reset_index()
FEATS = ["car_pen", "tw_sh", "tractor_sh", "log_car", "market_growth_yoy", "state_enc"]
y = X.share_l12m.to_numpy() * 100
oof = np.zeros((5, len(X)))
for rep in range(5):
    for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
        a, b = X.iloc[tr].copy(), X.iloc[te].copy()
        enc = a.groupby("state_name").units_l12m.sum() / a.groupby("state_name").car_market_l12m.sum() * 100
        glob = a.units_l12m.sum() / a.car_market_l12m.sum() * 100
        a["state_enc"], b["state_enc"] = a.state_name.map(enc), b.state_name.map(enc).fillna(glob)
        mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20,
                                            random_state=rep).fit(a[FEATS], y[tr])
        oof[rep, te] = mdl.predict(b[FEATS])
X["expected_share_pct"] = oof.mean(axis=0).round(2)
state_share = X.groupby("state_name").apply(lambda g: g.units_l12m.sum() / g.car_market_l12m.sum() * 100)
model_card = {
    "expected_share_model": "HistGradientBoosting, 5x5-fold out-of-fold, in-fold state encoding",
    "features": FEATS,
    "cv_r2": round(r2_score(y, X.expected_share_pct), 3),
    "cv_mae_pp": round(mean_absolute_error(y, X.expected_share_pct), 2),
    "state_average_mae_pp": round(mean_absolute_error(y, X.state_name.map(state_share)), 2),
    "state_average_r2": round(r2_score(y, X.state_name.map(state_share)), 3),
}

# ------------------------------------------------------------------ 4. dealer x snapshot-month input table
cm = panel.pivot(index="date", columns="dealer_id", values="car_market")[ids].astype(float)
comp = {c: panel.pivot(index="date", columns="dealer_id", values=c)[ids].astype(float) for c in COMPETITORS}
opsw = {c: ops.pivot(index="date", columns="dealer_id", values=c)[ids] for c in
        ["target_units", "inventory_days", "aged_stock_pct", "payment_delay_days", "csi_score", "complaints",
         "closing_stock", "retail_units"]}

def wsum(df, end, n_months, lag=0):
    """sum over the n months ending `lag` months before `end`"""
    e = end - pd.DateOffset(months=lag)
    return df.loc[(df.index > e - pd.DateOffset(months=n_months)) & (df.index <= e)].sum()

def wmean(df, end, n_months, lag=0):
    e = end - pd.DateOffset(months=lag)
    return df.loc[(df.index > e - pd.DateOffset(months=n_months)) & (df.index <= e)].mean()

static = dealers.set_index("dealer_id")[["zone", "state_name", "office_code", "office_name"]].reindex(ids)


def territory_name(office, code):
    """'Rta, Faridabad' -> 'Faridabad · HR38' (registration series code keeps same-city offices distinct)"""
    import re
    s = re.sub(r"\((Up|Mh|Ka)\d+\)", "", office)
    s = re.sub(r"\b(Rto|Dto|Arto|Rta|Srto|Uo|Sdm|Dy|Rla|Sub|Office|District)\b\.?", "", s, flags=re.I)
    s = re.sub(r"\s*,\s*", ", ", s).strip(" ,-")
    s = re.sub(r"\s{2,}", " ", s).replace("-Ii", "-II").replace("-Iii", "-III")
    return f"{s or office} · {code}"


static["territory"] = [territory_name(o, c) for o, c in zip(static.office_name, static.office_code)]
exp_share = X.set_index("dealer_id").expected_share_pct.reindex(ids)
snap_months = months[-12:]
rows = []
for t in snap_months:
    t = pd.Timestamp(t)
    f = static.copy()
    f["as_of"] = t
    f["units_l12m"], f["units_p12m"] = wsum(U, t, 12), wsum(U, t, 12, 12)
    f["target_l12m"] = wsum(T, t, 12)
    f["car_market_l12m"], f["car_market_p12m"] = wsum(cm, t, 12), wsum(cm, t, 12, 12)
    f["units_last6m"], f["units_last6m_prev_year"] = wsum(U, t, 6), wsum(U, t, 6, 12)
    f["car_market_last6m"], f["car_market_last6m_prev_year"] = wsum(cm, t, 6), wsum(cm, t, 6, 12)
    f["units_last3m"], f["target_last3m"] = wsum(U, t, 3), wsum(T, t, 3)
    f["benchmark_share_pct"] = exp_share
    # stock metrics are reviewed as 3-month averages (a single month-end swings with festive retail)
    f["inventory_days"] = wmean(opsw["inventory_days"], t, 3).round(1)
    f["inventory_days_3m_ago"] = wmean(opsw["inventory_days"], t, 3, 3).round(1)
    f["closing_stock"] = opsw["closing_stock"].loc[t]
    f["aged_stock_pct"] = wmean(opsw["aged_stock_pct"], t, 3).round(1)
    f["payment_delay_days"] = wmean(opsw["payment_delay_days"], t, 3).round(1)
    f["payment_delay_days_3m_ago"] = wmean(opsw["payment_delay_days"], t, 3, 3).round(1)
    f["csi_score"] = wmean(opsw["csi_score"], t, 3).round(1)
    f["csi_score_3m_ago"] = wmean(opsw["csi_score"], t, 3, 3).round(1)
    f["complaints_per_100_units"] = (wsum(opsw["complaints"], t, 3) / wsum(opsw["retail_units"], t, 3).clip(lower=1) * 100).round(2)
    # competition: share change of each competitor, latest 12m vs prior 12m
    ml, mp = f.car_market_l12m.clip(lower=1), f.car_market_p12m.clip(lower=1)
    gains = pd.DataFrame({c: (wsum(comp[c], t, 12) / ml - wsum(comp[c], t, 12, 12) / mp) * 100 for c in COMPETITORS})
    f["top_competitor_gainer"] = gains.idxmax(axis=1)
    f["top_competitor_gain_pp"] = gains.max(axis=1).round(2)
    rows.append(f.reset_index())
inputs = pd.concat(rows, ignore_index=True)
inputs.to_parquet(OUT / "dealer_inputs_history.parquet", index=False)

# panel for charts (last 36 months) + competitor detail
panel[panel.date >= months[-36]].drop(columns=["fy"]).to_parquet(OUT / "panel_recent.parquet", index=False)
T.stack().rename("target_units").reset_index().rename(columns={"level_0": "date", "level_1": "dealer_id"}) \
    .to_parquet(OUT / "targets_monthly.parquet", index=False)

# network-level context for the insights page
car_groups = ["Maruti Suzuki"] + COMPETITORS + ["Nissan", "Ford", "Jeep-Stellantis", "BYD", "Mercedes-Benz", "BMW"]
nat = v[v.maker_group.isin(car_groups)].pivot_table(index="date", columns="maker_group", values="registrations",
                                                    aggfunc="sum").fillna(0)
nat.to_parquet(OUT / "national_brand_monthly.parquet")

# distribution sanity for the synthetic layer (latest month)
last = inputs[inputs.as_of == snap_months[-1]]
model_card["synthetic_latest_month"] = {
    c: {q: round(float(last[c].quantile(q)), 1) for q in (0.1, 0.5, 0.9)}
    for c in ["inventory_days", "aged_stock_pct", "payment_delay_days", "csi_score", "complaints_per_100_units"]}
model_card["target_achievement_l12m"] = {q: round(float((last.units_l12m / last.target_l12m).quantile(q)), 3)
                                         for q in (0.1, 0.5, 0.9)}
model_card["injected_events"] = pd.Series(event).replace("", np.nan).value_counts().to_dict()
json.dump(model_card, open(OUT / "model_card.json", "w"), indent=2)
print(json.dumps(model_card, indent=1))
print("inputs", inputs.shape, "ops", ops.shape)
