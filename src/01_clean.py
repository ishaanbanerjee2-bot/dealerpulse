"""
Step 1 - Clean & sanitise the two raw datasets.

Inputs  (data/raw/)
  vahan-vehicle-registrations-by-maker.csv   India Data Portal / MoRTH VAHAN, RTO x maker x month
  ts_transport_dealer_06_12_2025.csv         Telangana RTA registered-dealer roster (Open Data Telangana)

Outputs (data/processed/)
  vahan_clean.parquet          cleaned RTO x maker x month panel
  vahan_cleaning_log.csv       every dropped row with the reason (audit trail)
  ts_dealers_clean.csv         Telangana dealer roster, PII removed, brand + dealer-type tagged
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- VAHAN
v = pd.read_csv(RAW / "vahan-vehicle-registrations-by-maker.csv")
n_raw, vol_raw = len(v), v.registrations.sum()
v["date"] = pd.to_datetime(v["date"])
v = v.drop(columns=["id", "category"])  # category is constant ("Maker")

# 1. Harmonise state names (same UT spelled two ways after 2020 re-organisation)
STATE_FIX = {
    "Andaman & Nicobar Island": "Andaman And Nicobar Islands",
    "Jammu and Kashmir": "Jammu And Kashmir",
    "UT of DNH and DD": "Dadra Nagar Haveli And Daman Diu",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra Nagar Haveli And Daman Diu",
}
v["state_name"] = v["state_name"].replace(STATE_FIX)

# 2. One canonical office name per office code (a few codes carry 2 spellings)
canon = v.groupby(["office_code", "office_name"]).size().reset_index(name="n")
canon = canon.sort_values("n").drop_duplicates("office_code", keep="last").set_index("office_code").office_name
v["office_name"] = v["office_code"].map(canon)

# 3. Harmonise maker names (legal-entity renames / splits) -> brand group
MAKER_GROUP = {
    "Maruti Suzuki India Ltd": "Maruti Suzuki", "Maruti Udyog Ltd": "Maruti Suzuki",
    "Hyundai Motor India Ltd": "Hyundai",
    "Tata Motors Ltd": "Tata Motors", "Tata Motors Passenger Vehicles Ltd": "Tata Motors",
    "Tata Passenger Electric Mobility Ltd": "Tata Motors",
    "Mahindra & Mahindra Limited": "Mahindra", "Mahindra Electric Mobility Limited": "Mahindra",
    "Kia India Private Limited": "Kia", "Toyota Kirloskar Motor Pvt Ltd": "Toyota",
    "Honda Cars India Ltd": "Honda Cars", "Renault India Pvt Ltd": "Renault",
    "Nissan Motor India Pvt Ltd": "Nissan", "Mg Motor India Pvt Ltd": "MG Motor",
    "Skoda Auto Volkswagen India Pvt Ltd": "Skoda-VW", "Volkswagen India Pvt Ltd": "Skoda-VW",
    "Skoda Auto India Pvt Ltd": "Skoda-VW", "Ford India Pvt Ltd": "Ford",
    "Fca India Automobiles Private Limited": "Jeep-Stellantis", "Byd India Private Limited": "BYD",
    "Mercedes-Benz India Pvt Ltd": "Mercedes-Benz", "Bmw India Pvt Ltd": "BMW",
}
v["maker_group"] = v["type"].map(MAKER_GROUP).fillna("Other")
v = v.rename(columns={"type": "maker"})
# 3b. Source mislabel: in Madhya Pradesh 2019 Honda two-wheeler volume is filed under "Honda Cars India Ltd"
#     (~119k vs ~3k/yr in every other year). Keep the rows but move them out of the car market.
mislabel = (v.maker == "Honda Cars India Ltd") & (v.state_name == "Madhya Pradesh") & (v.date.dt.year == 2019)
v.loc[mislabel, "maker_group"] = "Other"

# 4. Exact-key duplicates (563 rows): split sub-records of the same key -> sum
v = v.groupby(["date", "state_name", "state_code", "office_name", "office_code", "maker", "maker_group"],
              as_index=False).registrations.sum()

log = []
# 5. Non-registering offices (vehicle fitness / testing centres register no new vehicles)
nonrto = v.office_name.str.contains("fitness|testing", case=False)
log.append(v[nonrto].assign(reason="non_rto_office"))
v = v[~nonrto].copy()

# 6. Contamination: national/state totals were written into ordinary office rows
#    (e.g. Maruti's national Jan-2020 total 164,202 appears in 19 different offices).
#    Baseline = office x maker median over the verified-clean years (2019, 2023-24).
clean_yrs = v.date.dt.year.isin([2019, 2023, 2024])
b_clean = v[clean_yrs].groupby(["office_code", "maker"]).registrations.median()
b_all = v.groupby(["office_code", "maker"]).registrations.median()
key = pd.MultiIndex.from_frame(v[["office_code", "maker"]])
v["baseline"] = b_clean.reindex(key).to_numpy()
v["baseline"] = v["baseline"].fillna(pd.Series(b_all.reindex(key).to_numpy(), index=v.index))
v["row_flag"] = (v.registrations > 8 * v.baseline) & (v.registrations - v.baseline > 150)

# 6b. Whole office-month is suspect if >50% of its volume is in flagged rows
fv = v.registrations.where(v.row_flag, 0)
share = fv.groupby([v.office_code, v.date]).transform("sum") / v.groupby(["office_code", "date"]).registrations.transform("sum")
v["om_flag"] = share > 0.5

# 6c. Offices contaminated in most months are unusable as a time series -> drop entirely
bad_rate = (v.row_flag | v.om_flag).groupby(v.office_code).mean()
bad_offices = bad_rate[bad_rate > 0.5].index
v["office_flag"] = v.office_code.isin(bad_offices)

drop = v.row_flag | v.om_flag | v.office_flag
reason = np.select([v.office_flag, v.om_flag, v.row_flag],
                   ["contaminated_office", "contaminated_office_month", "outlier_vs_baseline"], default="")
log.append(v[drop].assign(reason=reason[drop.to_numpy()]))
v = v[~drop].drop(columns=["baseline", "row_flag", "om_flag", "office_flag"])

pd.concat(log)[["date", "state_name", "office_code", "office_name", "maker", "registrations", "reason"]] \
  .to_csv(OUT / "vahan_cleaning_log.csv", index=False)
v.to_parquet(OUT / "vahan_clean.parquet", index=False)

print(f"VAHAN raw rows {n_raw:,} vol {vol_raw/1e6:,.1f}M  ->  clean rows {len(v):,} vol {v.registrations.sum()/1e6:,.1f}M")
print("dropped offices:", sorted(canon[bad_offices]))
print(pd.concat(log).groupby("reason").agg(rows=("registrations", "size"), vol=("registrations", "sum")))

# ---------------------------------------------------------------- Telangana dealers
d = pd.read_csv(RAW / "ts_transport_dealer_06_12_2025.csv", dtype=str)
for c in d.columns:
    d[c] = d[c].str.strip().str.replace(r"\s+", " ", regex=True).str.rstrip(",")
text = (d.dealerName + " " + d.repEmail.str.split("@").str[0]).str.upper()

BRAND_RX = {
    "Maruti Suzuki": r"MARUTI|MARUTHI|SABOO|VARUN MOTOR|NEXA", "Hyundai": r"HYUNDAI",
    "Tata Motors": r"\bTATA", "Mahindra": r"MAHINDRA", "Toyota": r"TOYOTA", "Kia": r"\bKIA\b",
    "Honda": r"HONDA", "Hero": r"\bHERO\b", "Bajaj": r"BAJAJ", "TVS": r"\bTVS\b",
    "Suzuki 2W": r"SUZUKI", "Yamaha": r"YAMAHA", "Royal Enfield": r"ENFIELD",
    "Eicher": r"EICHER", "Renault": r"RENAULT", "Nissan": r"NISSAN", "Skoda-VW": r"SKODA|VOLKSWAGEN",
    "MG Motor": r"\bMG\b", "Ather": r"ATHER", "Ola": r"\bOLA\b",
}
d["brand_inferred"] = None
for b, rx in BRAND_RX.items():
    d.loc[d.brand_inferred.isna() & text.str.contains(rx, regex=True), "brand_inferred"] = b

TYPE_RX = [  # order matters
    ("Body builder / fabricator", r"ENGINEERING|ENGG|BODY|TRAILER|WORKS|FABRICAT|INDUSTRIES|COACH"),
    ("Tractor / farm equipment", r"TRACTOR|AGRO|HARVEST|FARM"),
    ("EV dealer", r"\bEV\b|ELECTRIC|E-?MOBILITY|E BIKE"),
    ("Two-wheeler dealer", r"MOTOR ?CYCLE|BIKE|MOBIKE|SCOOT|HONDA|HERO|BAJAJ|TVS|YAMAHA|ENFIELD|SUZUKI"),
    ("Car / CV dealer", r"MARUTI|HYUNDAI|TATA|MAHINDRA|TOYOTA|KIA|CARS|RENAULT|NISSAN|SKODA|EICHER|\bMG\b"),
]
d["dealer_type"] = "Automobile dealer (brand unknown)"
for t, rx in reversed(TYPE_RX):  # first match in list wins
    d.loc[text.str.contains(rx, regex=True), "dealer_type"] = t

d["region_cluster"] = np.where(d.district.isin(["HYDERABAD", "RANGA REDDY", "MEDCHAL M-GIRI", "SANGAREDDY"]),
                               "Hyderabad metro", "Rest of Telangana")
d["dup_group_size"] = d.groupby(["dealerName", "address1"]).dealerId.transform("size")
# PII removed: personal email, phone and street address are not needed for analysis
keep = ["dealerId", "dealerName", "city", "district", "mandal", "pincode", "OfficeCd",
        "brand_inferred", "dealer_type", "region_cluster", "dup_group_size"]
# phone numbers also hide inside free-text fields in the source (e.g. "JAGTIAL 9885936666") -> scrub them
for c in ["dealerName", "city", "district", "mandal"]:
    d[c] = d[c].str.replace(r"\+?\d[\d\s-]{8,}\d", "", regex=True).str.replace(r"\S+@\S+", "", regex=True) \
               .str.replace(r"\s+", " ", regex=True).str.strip(" ,-")
d[keep].rename(columns={"OfficeCd": "rta_office"}).to_csv(OUT / "ts_dealers_clean.csv", index=False)
print(f"\nTS dealers {len(d)} | brand inferred for {d.brand_inferred.notna().sum()}")
print(d.dealer_type.value_counts())
