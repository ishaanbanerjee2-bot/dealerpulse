"""
Step 2 - Build the master dataset.

Unit of analysis: a DEALER TERRITORY = Maruti Suzuki x one RTO office.
VAHAN registrations are retail sales recorded where the buyer registers the car, so an RTO's
Maruti registrations approximate the sales of the Maruti dealer(s) serving that RTO.

Outputs (data/processed/)
  master_panel.csv      territory x month (2019-01 .. 2024-05): own sales, market, competitor sales
  master_dealers.csv    one row per territory: real market KPIs for the latest 12 months
  ts_network_benchmarks.csv  network-structure facts from the Telangana roster
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OEM = "Maruti Suzuki"
COMPETITORS = ["Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW", "Renault"]
CAR_BRANDS = [OEM] + COMPETITORS + ["Nissan", "Ford", "Jeep-Stellantis", "BYD", "Mercedes-Benz", "BMW"]

ZONE = {
    "North": ["Delhi", "Haryana", "Punjab", "Himachal Pradesh", "Jammu And Kashmir", "Ladakh", "Chandigarh",
              "Uttarakhand", "Uttar Pradesh", "Rajasthan"],
    "West": ["Maharashtra", "Gujarat", "Goa", "Dadra Nagar Haveli And Daman Diu"],
    "South": ["Karnataka", "Kerala", "Tamil Nadu", "Andhra Pradesh", "Puducherry", "Andaman And Nicobar Islands"],
    "East": ["West Bengal", "Odisha", "Bihar", "Jharkhand"],
    "Central": ["Madhya Pradesh", "Chhattisgarh"],
    "North-East": ["Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"],
}
STATE_ZONE = {s: z for z, ss in ZONE.items() for s in ss}

v = pd.read_parquet(P / "vahan_clean.parquet")
months = pd.date_range(v.date.min(), v.date.max(), freq="MS")

# ---- office x month aggregates
tot = v.groupby(["office_code", "date"]).registrations.sum().rename("all_vehicles_market")
car = v[v.maker_group.isin(CAR_BRANDS)]
brand = car.pivot_table(index=["office_code", "date"], columns="maker_group", values="registrations", aggfunc="sum")
panel = brand.join(tot, how="outer").fillna(0)
panel["car_market"] = panel[[c for c in CAR_BRANDS if c in panel]].sum(axis=1)

# complete grid so missing months are explicit zeros only when the office reported that month
offices = v.groupby("office_code").agg(state_name=("state_name", "first"), office_name=("office_name", "first"))
reported = v.groupby(["office_code", "date"]).size().rename("office_reported")
grid = pd.MultiIndex.from_product([offices.index, months], names=["office_code", "date"])
panel = panel.reindex(grid).join(reported)
panel["office_reported"] = panel.office_reported.notna()
panel = panel.reset_index().merge(offices, on="office_code")

# ---- territory eligibility: a realistic dealer catchment
last12 = panel.date > months[-13]
prev12 = (panel.date > months[-25]) & ~last12
stats = panel.groupby("office_code").apply(lambda g: pd.Series({
    "l12_units": g.loc[last12[g.index], OEM].sum(),
    "p12_units": g.loc[prev12[g.index], OEM].sum(),
    "months_reported_24": g.loc[(last12 | prev12)[g.index], "office_reported"].sum(),
}))
# administrative break: an office whose WHOLE vehicle market (all brands, all vehicle types) steps up or down
# was split, merged or had work moved - a jurisdiction change, not a dealer-performance signal.
# Level = office's share of its zone's registrations (removes seasonality & macro swings); a break is a
# sustained shift vs the Jun-22..May-23 reference, over both the last 6 and last 9 months (3m alone is noisy:
# e.g. winter in hill districts).
panel["zone_tmp"] = panel.state_name.map(STATE_ZONE)
zone_tot = panel.groupby(["zone_tmp", "date"]).all_vehicles_market.transform("sum")
panel["rel_level"] = panel.all_vehicles_market / zone_tot.where(zone_tot > 0)


def level_shift(g):
    r = g.sort_values("date").rel_level.to_numpy()
    base = np.nanmedian(r[-24:-12])
    if not base or not np.isfinite(base):
        return pd.Series({"r6": np.nan, "r9": np.nan})
    return pd.Series({"r6": np.nanmedian(r[-6:]) / base, "r9": np.nanmedian(r[-9:]) / base})


shift = panel.groupby("office_code").apply(level_shift)
stats = stats.join(shift)
stats["admin_break"] = ((stats.r6 < 0.65) & (stats.r9 < 0.70)) | ((stats.r6 > 1.55) & (stats.r9 > 1.50))
panel = panel.drop(columns=["zone_tmp", "rel_level"])
# special-purpose offices (inspection units, state HQ) register fleets / special numbers, not a retail catchment
special = offices.office_name.str.contains("Viu|Head Office|State Transport Authority", case=False)
stats["special_office"] = stats.index.map(special)
eligible_pre = (stats.l12_units >= 240) & (stats.p12_units >= 120) & (stats.months_reported_24 == 24) & ~stats.special_office
print("territories excluded for administrative break:", int((stats.admin_break & eligible_pre).sum()))
stats[stats.admin_break & eligible_pre].join(offices).round(2).to_csv(P / "excluded_admin_breaks.csv")
eligible = stats[(stats.l12_units >= 240) & (stats.p12_units >= 120) & (stats.months_reported_24 == 24)
                 & ~stats.admin_break & ~stats.special_office].index
panel = panel[panel.office_code.isin(eligible)].copy()

panel["zone"] = panel.state_name.map(STATE_ZONE)
panel = panel.sort_values(["zone", "state_name", "office_code", "date"])
# pseudonymous dealer id (we do NOT attach synthetic attributes to real business names)
ids = panel[["state_name", "office_code"]].drop_duplicates().copy()
ids["dealer_id"] = "MS-" + ids.office_code.str.extract(r"^([A-Z]+)")[0] + "-" + \
                   (ids.groupby("state_name").cumcount() + 1).astype(str).str.zfill(3)
panel = panel.merge(ids[["office_code", "dealer_id"]], on="office_code")

panel = panel.rename(columns={OEM: "oem_units"})
panel["oem_car_share"] = np.where(panel.car_market > 0, panel.oem_units / panel.car_market, np.nan)
cols = ["dealer_id", "zone", "state_name", "office_code", "office_name", "date", "office_reported",
        "oem_units", "car_market", "all_vehicles_market", "oem_car_share"] + [c for c in COMPETITORS if c in panel]
panel = panel[cols]
for c in ["oem_units", "car_market", "all_vehicles_market"] + COMPETITORS:
    panel[c] = panel[c].fillna(0).astype(int)
panel.to_csv(P / "master_panel.csv", index=False)

# ---- dealer snapshot: latest 12m (Jun-23..May-24) vs prior 12m
L = panel[panel.date > months[-13]]
Pv = panel[(panel.date > months[-25]) & (panel.date <= months[-13])]
agg = lambda df: df.groupby("dealer_id")[["oem_units", "car_market", "all_vehicles_market"] + COMPETITORS].sum()
l, p = agg(L), agg(Pv)
s = panel.groupby("dealer_id")[["zone", "state_name", "office_code", "office_name"]].first()
s["units_l12m"] = l.oem_units
s["units_p12m"] = p.oem_units
s["sales_growth_yoy"] = l.oem_units / p.oem_units - 1
s["car_market_l12m"] = l.car_market
s["market_growth_yoy"] = l.car_market / p.car_market - 1
s["share_l12m"] = l.oem_units / l.car_market
s["share_p12m"] = p.oem_units / p.car_market
s["share_change_pp"] = (s.share_l12m - s.share_p12m) * 100
s["growth_vs_market_pp"] = (s.sales_growth_yoy - s.market_growth_yoy) * 100
# biggest competitor gainer in the territory (share points)
comp_pp = (l[COMPETITORS].div(l.car_market, axis=0) - p[COMPETITORS].div(p.car_market, axis=0)) * 100
s["top_competitor_gainer"] = comp_pp.idxmax(axis=1)
s["top_competitor_gain_pp"] = comp_pp.max(axis=1)
# untapped potential: units if the territory matched its STATE share, minus actual (floored at 0)
state_share = s.groupby("state_name").apply(lambda g: g.units_l12m.sum() / g.car_market_l12m.sum())
s["state_share_l12m"] = s.state_name.map(state_share)
s["headroom_units"] = (s.car_market_l12m * s.state_share_l12m - s.units_l12m).clip(lower=0).round()
# momentum: last 6 months vs same 6 months a year earlier (de-seasonalised)
m6 = panel[panel.date > months[-7]].groupby("dealer_id").oem_units.sum()
m6p = panel[(panel.date > months[-19]) & (panel.date <= months[-13])].groupby("dealer_id").oem_units.sum()
s["growth_last6m_yoy"] = m6 / m6p - 1
# volatility of monthly sales (coefficient of variation, last 24 months)
r24 = panel[panel.date > months[-25]]
s["sales_cv_24m"] = r24.groupby("dealer_id").oem_units.std() / r24.groupby("dealer_id").oem_units.mean()
# share trend slope (pp per year) over last 24 months
def slope(g):
    y = g.oem_car_share.to_numpy() * 100
    x = np.arange(len(y)) / 12
    ok = ~np.isnan(y)
    return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() > 6 else np.nan
s["share_trend_pp_per_yr"] = r24.groupby("dealer_id").apply(slope)
s = s.round(4).reset_index()
s.to_csv(P / "master_dealers.csv", index=False)

# ---- Telangana network benchmarks (structure only; no join possible - TS absent from VAHAN)
d = pd.read_csv(P / "ts_dealers_clean.csv")
bench = {
    "ts_registered_dealers": len(d),
    "ts_rta_offices": d.rta_office.nunique(),
    "dealers_per_rta_office_median": d.groupby("rta_office").size().median(),
    "share_dealers_hyderabad_metro": (d.region_cluster == "Hyderabad metro").mean(),
    "share_body_builders": (d.dealer_type == "Body builder / fabricator").mean(),
    "multi_outlet_name_groups": (d.groupby("dealerName").size() > 1).sum(),
    "dealers_in_multi_outlet_groups": d.dealerName.map(d.dealerName.value_counts()).gt(1).sum(),
}
pd.Series(bench).to_csv(P / "ts_network_benchmarks.csv", header=["value"])

print(f"territories: {s.shape[0]}  panel rows: {len(panel):,}")
print(s.groupby("zone").dealer_id.count())
print(s.describe().T.round(3).to_string())
print(pd.Series(bench))
