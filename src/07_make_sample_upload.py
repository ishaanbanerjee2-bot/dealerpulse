"""
Step 7 - A realistic sample of the monthly file a manufacturer would upload ("Your data" mode).
150 territories x 36 months (Jun-2021..May-2024), built from the same data as the sample network, so uploading
it must reproduce the sample network's health scores for those dealers (tested in tests/test_ingest.py).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data/app"
panel = pd.read_parquet(A / "panel_recent.parquet")
ops = pd.read_parquet(A / "ops_monthly.parquet")
tg = pd.read_parquet(A / "targets_monthly.parquet")
hist = pd.read_parquet(A / "dealer_inputs_history.parquet")
last = hist[hist.as_of == hist.as_of.max()]
# a spread of territories: every 6th dealer by priority-neutral id order
pick = sorted(last.dealer_id)[::6][:150]
info = last.set_index("dealer_id")[["territory", "state_name", "zone", "benchmark_share_pct"]]
comps = ["Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW", "Renault"]
p = panel[panel.dealer_id.isin(pick)].merge(tg, on=["dealer_id", "date"], how="left") \
    .merge(ops[["dealer_id", "date", "inventory_days", "aged_stock_pct", "payment_delay_days", "csi_score", "complaints"]],
           on=["dealer_id", "date"], how="left")
out = pd.DataFrame({
    "dealer_id": p.dealer_id, "dealer_name": p.dealer_id.map(info.territory), "state": p.dealer_id.map(info.state_name),
    "zone": p.dealer_id.map(info.zone), "month": p.date.dt.strftime("%Y-%m"), "units_sold": p.oem_units,
    "target_units": p.target_units, "local_car_market": p.car_market,
    "expected_share_pct": p.dealer_id.map(info.benchmark_share_pct).round(2),
    "inventory_days": p.inventory_days, "aged_stock_pct": p.aged_stock_pct, "payment_delay_days": p.payment_delay_days,
    "csi_score": p.csi_score, "complaints": p.complaints,
    **{"comp_" + c: p[c] for c in comps},
}).sort_values(["dealer_id", "month"])
out.to_csv(A / "sample_monthly_upload.csv", index=False)
print(out.shape, out.month.min(), out.month.max(), round((A / "sample_monthly_upload.csv").stat().st_size / 1e6, 2), "MB")
