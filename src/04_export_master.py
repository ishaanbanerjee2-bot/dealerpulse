"""Step 4 - Package everything into one master workbook: data/master/dealer_master_dataset.xlsx"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "data/master"
OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(ROOT / "reports/eda_stats.json"))

readme = pd.DataFrame({"Item": [
    "Purpose", "Unit of analysis", "Real data (source 1)", "Real data (source 2)", "Why the two are not row-joined",
    "Period", "Latest-12m window", "OEM chosen", "Car market definition", "Territory eligibility",
    "Data cleaning", "Synthetic data", "Privacy"], "Detail": [
    "Master dataset for the 'Which dealers need management attention?' prototype (Illuminati.ai case).",
    "Dealer territory = Maruti Suzuki x one RTO office. RTO registrations are where buyers register new cars, so they proxy the sales of the dealer(s) serving that RTO.",
    "VAHAN vehicle registrations by maker (MoRTH via India Data Portal): RTO x maker x month, 2019-01 to 2024-05.",
    "Telangana RTA registered-dealer roster (Open Data Telangana, snapshot 06-12-2025): 1,246 dealers, 53 RTA offices.",
    "Telangana is absent from VAHAN for this period and the roster has no brand or sales fields, so it is used as a network-structure benchmark, not joined row-by-row.",
    "Jan-2019 to May-2024 (65 months).", "Jun-2023 to May-2024 vs Jun-2022 to May-2023.",
    "Maruti Suzuki: largest network, stable legal name across all years, and a real share-erosion story.",
    "Sum of 16 car brands. Tata Motors and Mahindra include some light commercial vehicles, so shares run a few pp below published PV shares.",
    ">=240 Maruti units in last 12m, >=120 in prior 12m, all 24 months reported, no administrative break in total registrations.",
    f"Raw {S['raw_volume_m']}M registrations -> {S['clean_volume_m']}M after removing leaked national totals, fitness-centre offices, 20 contaminated RTOs and one maker mislabel. See Cleaning_Summary.",
    "None yet. Inventory ageing, payment delays, service and complaints are OEM-internal and will be added in the next step as clearly labelled synthetic columns.",
    "Telangana roster stripped of emails, phones and street addresses. Synthetic risk attributes are never attached to real business names; dealers carry pseudonymous IDs."]})

dictionary = pd.DataFrame([
    ("dealer_id", "Pseudonymous territory ID: MS-<state code>-<nnn>"), ("zone", "Sales zone (North/South/East/West/Central/North-East)"),
    ("office_code / office_name", "VAHAN RTO code and name"), ("oem_units", "Maruti registrations in the month (sales proxy)"),
    ("car_market", "All car-brand registrations in the RTO (local market potential)"),
    ("all_vehicles_market", "All registrations incl. 2W/3W/CV/tractors (overall demand / affluence proxy)"),
    ("oem_car_share", "oem_units / car_market"), ("<Competitor columns>", "Monthly registrations of each competitor brand"),
    ("units_l12m / units_p12m", "Maruti units, latest 12m / prior 12m"), ("sales_growth_yoy", "units_l12m / units_p12m - 1"),
    ("market_growth_yoy", "Car-market growth, same windows"), ("share_change_pp", "Change in Maruti car share, percentage points"),
    ("growth_vs_market_pp", "Sales growth minus market growth (pp); <0 means under-performing the market"),
    ("top_competitor_gainer / _pp", "Competitor that gained most share in the territory"),
    ("state_share_l12m", "Maruti share for the whole state (benchmark)"),
    ("headroom_units", "Extra annual units if the territory matched its state share (floored at 0)"),
    ("growth_last6m_yoy", "Last 6 months vs same 6 months a year earlier (momentum)"),
    ("sales_cv_24m", "Coefficient of variation of monthly sales (volatility)"),
    ("share_trend_pp_per_yr", "Slope of monthly share over 24 months"), ("quadrant", "Sales-growth x share-change segment"),
], columns=["Column", "Meaning"])

clean = pd.DataFrame(S["dropped_by_reason"]).T.reset_index().rename(columns={"index": "reason"})
bench = pd.read_csv(P / "ts_network_benchmarks.csv")

with pd.ExcelWriter(OUT / "dealer_master_dataset.xlsx", engine="openpyxl") as xw:
    readme.to_excel(xw, "README", index=False)
    dictionary.to_excel(xw, "Data_Dictionary", index=False)
    pd.read_csv(P / "master_dealers.csv").to_excel(xw, "Dealer_Snapshot", index=False)
    pd.read_csv(P / "master_panel.csv").to_excel(xw, "Monthly_Panel", index=False)
    pd.read_csv(P / "ts_dealers_clean.csv").to_excel(xw, "TS_Dealer_Roster", index=False)
    bench.to_excel(xw, "TS_Benchmarks", index=False)
    clean.to_excel(xw, "Cleaning_Summary", index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in col[:200]) + 2))
        ws.freeze_panes = "A2"
print("written", OUT / "dealer_master_dataset.xlsx")
