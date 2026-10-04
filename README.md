# DealerPulse — which dealers need management attention?

A Streamlit prototype for the Illuminati.ai case. It scores **909 Maruti Suzuki dealer territories across India**,
ranks the ones that need a regional manager's attention, explains **why**, and recommends **what to do**.

* **Real data:** sales, growth, market potential and competition come from VAHAN vehicle registrations (MoRTH via
  the India Data Portal), RTO × brand × month, Jan-2019 to May-2024.
* **Synthetic data:** inventory, payment delays, service CSI and complaints are OEM-internal, so they are
  modelled. Each one is linked to the real sales data and clearly labelled in the app.

**Live app:** https://dealerpulse.streamlit.app · **Code:** https://github.com/ishaanbanerjee2-bot/dealerpulse

## Run it locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. `requirements.txt` holds only what the app needs to run;
`requirements-dev.txt` adds the data pipeline and test tools.

## Deploy (Streamlit Community Cloud)

The app needs only the files in this repository: it loads `data/app/` (about 2 MB). On share.streamlit.io:

* **Repository:** this repo
* **Branch:** `main`
* **Main file path:** `app/streamlit_app.py`
* **Advanced settings → Python:** 3.12. The test suite passes on Python 3.12 with the versions the cloud installs.

## Raw data (not in the repo)

To rebuild everything from scratch, place the two source files in `data/raw/`:

* `vahan-vehicle-registrations-by-maker.csv`: India Data Portal → VAHAN Vehicle Registrations → by Maker
  (240 MB, too large for GitHub).
* `ts_transport_dealer_06_12_2025.csv`: Open Data Telangana → RTA vehicle dealers. This file is kept out of the repo
  because the original contains personal emails and phone numbers; the cleaned copy without them is in
  `data/processed/`.

## Use your own dealer data

The app isn't tied to the sample. In the sidebar, switch **Data → Your data** and upload a monthly dealer file
(CSV or Excel). Every view then runs on that network with the same engine: the Attention Board, deep-dives, trends,
what-if, Network Insights and Method.

* **Format:** one row per dealer per month, at least 24 months.
  * **Required:** `dealer_id, month, units_sold, target_units, local_car_market`.
  * **Optional, each one unlocks more:**
    * `dealer_name, state, zone`
    * `expected_share_pct`
    * `inventory_days, aged_stock_pct, payment_delay_days, csi_score, complaints`
    * `comp_<Brand>` competitor columns
* **Missing columns:** any column you don't supply is skipped, and the health score is re-weighted over what you do
  supply.
* **Checks:** the file is validated row by row, with clear messages for gaps, duplicates, impossible values
  and bad dates.
* **Downloads:** the landing page offers a template and a 150-dealer sample file
  (`data/app/sample_monthly_upload.csv`). Uploading the sample reproduces the built-in scores exactly (tested), so
  sample data and company data go through one code path.
* **One-off checks:** to score a few dealers without monthly history, use **Quick Assess** (form or snapshot CSV).

## What's in the app

| View | What a regional manager does there |
|---|---|
| **Attention Board** | Pick a zone or state and see KPIs, then a ranked attention list (risk × business size) and a risk map. Clicking a row opens the dealer. |
| **Dealer Deep-Dive** | Health score and its 4 pillars, why the dealer is flagged (with evidence), actions with owner and deadline, evidence charts and a what-if simulator. |
| **Quick Assess** | Type in one dealer's numbers, or upload a CSV/Excel file of many, and get a score, diagnosis and action plan from the same engine. |
| **Network Insights** | Who is taking share, seasonality by zone, zone scorecard, competitor threat map and where attention concentrates. |
| **Method & Data** | How the score works, the data sources (real vs synthetic), validation evidence, assumptions and limitations. |

The sidebar holds the inputs: scope (zone/state), a "data as of" slider to rewind and watch concerns emerge, focus
filters, and the scoring weights.

## How the score works

1. **Inputs → 14 metrics**, grouped into 4 pillars:
   * Sales & market share (real)
   * Recent momentum (real)
   * Inventory & payments (synthetic)
   * Service & complaints (synthetic)
2. **Metric scores.** Each metric scores 0 at a red line and 100 at a good level, linear in between. The pillars are
   weighted into a **health score** (weights are adjustable).
3. **Alerts.** These are rule-based and each carries its evidence.
   * **Tier** = health score, capped by alert severity.
   * **Priority** = risk × business size.
   * **Root cause:** when weak sales co-occur with stock/credit stress, the diagnosis is "Sales slump → stock stress".
4. **Accuracy safeguards** (all evidence is in `data/app/validation.json`):
   * Share changes must pass a quasi-binomial significance test. Raw share loss over-flags by about 5×.
   * Noisy rates for small dealers are shrunk by n/(n+K) (empirical Bayes).
   * Each dealer is compared with the share **expected for its market type**: out-of-fold gradient boosting,
     R² 0.42, which beats the state average.
   * An ML early-warning model was tested and **rejected**: out-of-time AUC 0.62 vs 0.65 for a one-line rule.
   * Injected synthetic events are caught at 82–97% recall.

## Repository

```
app/            streamlit_app.py (UI) · engine.py (scoring, alerts, actions) · ingest.py (your-data uploads) · charts.py · ui.py
src/            01_clean → 02_build_master → 03_eda → 04_export_master → 05_build_app_data → 06_validate → 07_make_sample_upload
data/raw/       the two source files (VAHAN by maker; Telangana RTA dealer roster)
data/processed/ cleaned data, cleaning log, exclusions
data/master/    dealer_master_dataset.xlsx (README, dictionary, snapshot, monthly panel)
data/app/       everything the app loads
reports/        01_EDA_and_Data_Report.md + figures
tests/          128 tests: engine & upload (75), headless UI stress (19), data integrity & accuracy (15), your-data mode (18)
```

To rebuild everything from the raw files:

```bash
for s in 01_clean 02_build_master 03_eda 04_export_master 05_build_app_data 06_validate 07_make_sample_upload; do .venv/bin/python src/$s.py; done
```

To run the tests:

```bash
.venv/bin/python -m pytest tests -q
```

## 1.5-minute demo path

1. **Problem (15 s).** A Maruti territory can grow sales and still be in trouble. 21% of territories grew sales while
   losing share to SUV brands, and stock and credit stress stay invisible in sales reports.
2. **Attention Board (25 s).** Walk through:
   * "242 of 909 need attention; 142 became concerns in the last 6 months."
   * The risk map's bottom-right quadrant: growing but losing share.
   * Click the #1 dealer.
3. **Deep-Dive (30 s).** Show:
   * The root cause in one sentence (share lost to Tata, which then drove stock to 117 days and overdues to 32 days).
   * Actions with owner and deadline.
   * In the what-if, cut stock and overdues and watch the tier improve.
   * Optionally, rewind "data as of" to show the concern emerging.
4. **Value (15 s).**
   * Managers spend their time on about 27% of dealers, ranked by risk × size.
   * They get the reason and the playbook, not just a red light.
   * About ₹6,100 cr of revenue gap vs target is made visible.
5. **Limitation (5 s).** Operating data (inventory, payments, service) is synthetic. A manufacturer replaces it by
   uploading its own monthly DMS/ERP extract under **Data → Your data**, and the whole app runs on it.

## Assumptions & limitations

* **A dealer is a territory.** RTO registrations stand in for dealer sales; one RTO can be served by several dealers.
  Each "dealer" is a Maruti × RTO territory with a pseudonymous ID. Synthetic attributes are never attached to real
  business names.
* **Targets are modelled:** prior fiscal-year sales × (1 + zone market growth, capped at 3–12%), phased by the
  zone's 2019 seasonality.
* **Revenue per unit** is ₹6.32 lakh (Maruti FY24 net sales ÷ volume).
* **Car market definition.** It covers 16 car brands. Tata and Mahindra include some LCVs, so share levels run a
  few points below published PV share. Trends are unaffected.
* **Coverage.** Data ends May-2024. Telangana isn't covered by VAHAN in this period, so the Telangana dealer roster is
  used only as a network-structure benchmark.
