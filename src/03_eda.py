"""
Step 3 - Exploratory + explanatory analysis. Writes figures to reports/figures and key numbers to
reports/eda_stats.json (every number quoted in the report comes from this file).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

ROOT = Path(__file__).resolve().parents[1]
P, FIG = ROOT / "data/processed", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
GRAY, INK, INK2, GRID = "#b4b2aa", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "figure.dpi": 140,
})
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)

S = {}
raw = pd.read_csv(ROOT / "data/raw/vahan-vehicle-registrations-by-maker.csv", usecols=["date", "type", "registrations"],
                  parse_dates=["date"])
v = pd.read_parquet(P / "vahan_clean.parquet")
panel = pd.read_csv(P / "master_panel.csv", parse_dates=["date"])
dl = pd.read_csv(P / "master_dealers.csv")
ts = pd.read_csv(P / "ts_dealers_clean.csv")
log = pd.read_csv(P / "vahan_cleaning_log.csv")

# ------------------------------------------------ 1. data quality: raw vs cleaned national totals
rt = raw.groupby("date").registrations.sum() / 1e6
ct = v.groupby("date").registrations.sum() / 1e6
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(rt.index, rt, color=GRAY, label="Raw file")
ax.plot(ct.index, ct, color=BLUE, label="After cleaning")
ax.set_ylabel("Registrations / month (million)")
ax.set_title(f"Raw VAHAN file was inflated up to {(rt / ct).max():.0f}x by leaked national totals")
ax.annotate("Real India volume ≈ 2M/month", (ct.index[48], ct.iloc[48]), (ct.index[30], 30),
            arrowprops=dict(arrowstyle="-", color=INK2), color=INK2)
ax.legend(frameon=False)
save(fig, "01_data_quality_raw_vs_clean.png")
S["raw_rows"], S["raw_volume_m"] = len(raw), round(raw.registrations.sum() / 1e6, 1)
S["clean_rows"], S["clean_volume_m"] = len(v), round(v.registrations.sum() / 1e6, 1)
S["dropped_by_reason"] = log.groupby("reason").agg(rows=("registrations", "size"), volume=("registrations", "sum")).to_dict("index")
S["peak_inflation_x"] = round(float((rt / ct).max()), 1)

# ------------------------------------------------ 2. national OEM story
CAR = ["Maruti Suzuki", "Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW",
       "Renault", "Nissan", "Ford", "Jeep-Stellantis", "BYD", "Mercedes-Benz", "BMW"]
car = v[v.maker_group.isin(CAR)].pivot_table(index="date", columns="maker_group", values="registrations", aggfunc="sum").fillna(0)
share = car.div(car.sum(axis=1), axis=0) * 100
ms = share["Maruti Suzuki"].rolling(3).mean()
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(ms.index, ms, color=BLUE)
ax.set_ylabel("Maruti share of car registrations (%)"); ax.yaxis.set_major_formatter(mt.PercentFormatter(decimals=0))
ax.set_title("Maruti's national car share is eroding (3-month rolling)")
save(fig, "02_maruti_national_share.png")
y19 = share[share.index.year == 2019].mean(); l12 = share[share.index > share.index[-13]].mean()
S["maruti_share_2019"], S["maruti_share_l12m"] = round(y19["Maruti Suzuki"], 1), round(l12["Maruti Suzuki"], 1)
d = (l12 - y19).sort_values()
d = d[d.abs() > 0.4]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(d.index, d.values, color=[RED if x < 0 else AQUA for x in d.values], height=0.6)
for i, x in enumerate(d.values):
    ax.text(x + (0.15 if x >= 0 else -0.15), i, f"{x:+.1f}", va="center", ha="left" if x >= 0 else "right", color=INK2, fontsize=9)
ax.set_xlim(d.min() - 1.2, d.max() + 1); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("Change in car-market share, 2019 → Jun-23–May-24 (pp)")
ax.set_title("Who took Maruti's share: SUV-led brands"); ax.grid(axis="y", visible=False)
save(fig, "03_brand_share_shift.png")
S["brand_share_shift_pp"] = d.round(2).to_dict()

# ------------------------------------------------ 3. territory quadrant: sales growth vs share change
dl["quadrant"] = np.select(
    [(dl.sales_growth_yoy >= 0) & (dl.share_change_pp >= 0), (dl.sales_growth_yoy >= 0) & (dl.share_change_pp < 0),
     (dl.sales_growth_yoy < 0) & (dl.share_change_pp >= 0)],
    ["Healthy: growing & gaining share", "Hidden risk: growing but losing share", "Market-driven dip: falling but holding share"],
    "Clear decline: falling & losing share")
q = dl.quadrant.value_counts()
S["n_territories"] = len(dl)
S["quadrant_counts"] = q.to_dict()
S["pct_growing_but_losing_share"] = round(100 * q.get("Hidden risk: growing but losing share", 0) / len(dl), 1)
S["pct_growth_below_market"] = round(100 * (dl.growth_vs_market_pp < 0).mean(), 1)
S["pct_positive_growth_but_below_market"] = round(100 * ((dl.sales_growth_yoy > 0) & (dl.growth_vs_market_pp < 0)).mean(), 1)
QC = {"Healthy: growing & gaining share": AQUA, "Hidden risk: growing but losing share": YELLOW,
      "Market-driven dip: falling but holding share": BLUE, "Clear decline: falling & losing share": RED}
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for k, c in QC.items():
    g = dl[dl.quadrant == k]
    ax.scatter(g.sales_growth_yoy * 100, g.share_change_pp, s=np.clip(g.units_l12m / 40, 10, 250), color=c, alpha=0.7,
               edgecolor="#fcfcfb", linewidth=1, label=f"{k} ({len(g)})")
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(-60, 100); ax.set_ylim(-20, 20)
ax.set_xlabel("Maruti sales growth YoY (%)"); ax.set_ylabel("Change in Maruti car share (pp)")
ax.set_title(f"Sales growth alone hides share loss — {len(dl)} territories (bubble = volume)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "04_territory_quadrant.png")

# ------------------------------------------------ 4. momentum reversal: good year, bad recent half-year
rev = dl[(dl.sales_growth_yoy > 0) & (dl.growth_last6m_yoy < -0.10)]
S["n_momentum_reversal"] = len(rev)

# ------------------------------------------------ 5. headroom & concentration
S["headroom_total_units"] = int(dl.headroom_units.sum())
S["headroom_top20_share_pct"] = round(100 * dl.nlargest(20, "headroom_units").headroom_units.sum() / dl.headroom_units.sum(), 1)
top = dl.nlargest(12, "headroom_units").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(top.office_name + " (" + top.state_name.str[:12] + ")", top.headroom_units, color=BLUE, height=0.6)
for i, (x, sh, ss) in enumerate(zip(top.headroom_units, top.share_l12m, top.state_share_l12m)):
    ax.text(x + 40, i, f"share {sh:.0%} vs state {ss:.0%}", va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("Untapped units / year if territory matched its state share")
ax.set_title("Biggest under-penetrated markets"); ax.grid(axis="y", visible=False)
save(fig, "05_headroom_top.png")
srt = dl.units_l12m.sort_values(ascending=False).cumsum() / dl.units_l12m.sum()
S["pct_territories_for_50pct_volume"] = round(100 * (srt < 0.5).sum() / len(dl), 1)

# ------------------------------------------------ 6. zone view
z = dl.groupby("zone").agg(territories=("dealer_id", "size"), units=("units_l12m", "sum"), market=("car_market_l12m", "sum"),
                           units_p=("units_p12m", "sum"), hidden=("quadrant", lambda s: (s == "Hidden risk: growing but losing share").mean()),
                           decline=("quadrant", lambda s: (s == "Clear decline: falling & losing share").mean()))
z["share"] = z.units / z.market; z["growth"] = z.units / z.units_p - 1
S["zone_summary"] = z.round(3).to_dict("index")

# ------------------------------------------------ 7. seasonality (festive uplift) by zone
p = panel[panel.date.dt.year.isin([2022, 2023])].copy()
p["m"] = p.date.dt.month
zm = p.groupby(["zone", "m"]).oem_units.sum().unstack()
idx = zm.div(zm.mean(axis=1), axis=0)
S["festive_index_oct_nov"] = idx[[10, 11]].mean(axis=1).round(2).to_dict()
S["festive_index_aug_sep"] = idx[[8, 9]].mean(axis=1).round(2).to_dict()
S["peak_month_by_zone"] = idx.idxmax(axis=1).to_dict()
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for zname, c in zip(["North", "South", "West", "East"], [BLUE, ORANGE, AQUA, YELLOW]):
    ax.plot(range(1, 13), idx.loc[zname], color=c, label=zname, marker="o", ms=4)
ax.axhline(1, color=INK2, lw=0.8); ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("Monthly sales index (avg = 1)"); ax.set_title("Festive timing differs by zone: South high in Aug–Sep, West/East peak in October")
ax.legend(frameon=False, ncol=4)
save(fig, "06_seasonality_by_zone.png")

# ------------------------------------------------ 8. Telangana roster
S["ts"] = {"dealers": len(ts), "rta_offices": int(ts.rta_office.nunique()),
           "dealer_type": ts.dealer_type.value_counts().to_dict(),
           "pct_hyderabad_metro": round(100 * (ts.region_cluster == "Hyderabad metro").mean(), 1),
           "brand_inferred": int(ts.brand_inferred.notna().sum())}
t = ts.dealer_type.value_counts().iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(t.index, t.values, color=BLUE, height=0.6)
for i, x in enumerate(t.values): ax.text(x + 8, i, str(x), va="center", fontsize=9, color=INK2)
ax.set_title("Telangana RTA dealer roster: half have no identifiable brand"); ax.grid(axis="y", visible=False)
save(fig, "07_ts_dealer_types.png")

# example territories for the narrative
cols = ["dealer_id", "office_name", "state_name", "units_l12m", "sales_growth_yoy", "market_growth_yoy", "share_change_pp", "top_competitor_gainer"]
S["examples_hidden_risk"] = dl[dl.quadrant.str.startswith("Hidden")].nlargest(5, "units_l12m")[cols].round(3).to_dict("records")
S["examples_clear_decline"] = dl[dl.quadrant.str.startswith("Clear")].nlargest(5, "units_l12m")[cols].round(3).to_dict("records")
S["top_competitor_gainer_counts"] = dl[dl.share_change_pp < 0].top_competitor_gainer.value_counts().to_dict()

dl.to_csv(P / "master_dealers.csv", index=False)  # persist quadrant label
json.dump(S, open(ROOT / "reports/eda_stats.json", "w"), indent=2, default=str)
print(json.dumps(S, indent=1, default=str))
