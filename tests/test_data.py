"""Data integrity + independent accuracy checks: every number the app shows is recomputed here from source tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import engine as E  # noqa: E402

P, A = ROOT / "data/processed", ROOT / "data/app"
PANEL = pd.read_csv(P / "master_panel.csv", parse_dates=["date"])
HIST = pd.read_parquet(A / "dealer_inputs_history.parquet")
OPS = pd.read_parquet(A / "ops_monthly.parquet")
TGT = pd.read_parquet(A / "targets_monthly.parquet")
COMPETITORS = ["Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota", "Honda Cars", "MG Motor", "Skoda-VW", "Renault"]


@pytest.fixture(scope="module")
def vahan():
    return pd.read_parquet(P / "vahan_clean.parquet")


# ------------------------------------------------------------------ cleaned VAHAN reproduces known reality
def test_monthly_national_volume_is_realistic(vahan):
    t = vahan.groupby("date").registrations.sum() / 1e6
    lockdown = t.index.isin(pd.to_datetime(["2020-04-01", "2020-05-01", "2021-05-01"]))
    assert t[lockdown].max() < 0.8, "Covid lockdown months must show the collapse"
    assert t[~lockdown].between(0.9, 3.2).all(), t[~lockdown].describe()


def test_known_brand_facts(vahan):
    car = vahan[vahan.maker_group != "Other"].pivot_table(index="date", columns="maker_group", values="registrations",
                                                          aggfunc="sum").fillna(0)
    share = car.div(car.sum(axis=1), axis=0) * 100
    assert share[share.index.year == 2019]["Maruti Suzuki"].mean() == pytest.approx(41.0, abs=0.2)
    assert share[share.index > "2023-05-01"]["Maruti Suzuki"].mean() == pytest.approx(35.9, abs=0.2)
    assert car.loc[car.index.year == 2019, "Kia"].sum() == 0, "Kia launched in India in Aug-2019 registrations"
    assert car.loc[car.index >= "2023-01-01", "Ford"].sum() < 500, "Ford exited India in 2021-22"
    m19 = car.loc[car.index.year == 2019, "Maruti Suzuki"].sum()
    assert 1.3e6 < m19 < 1.7e6, "Maruti retails ~1.5M cars a year"


def test_no_leaked_totals_or_bad_offices_survive(vahan):
    assert vahan.registrations.max() < 60000, "no single RTO-month-maker row should hold a national total"
    bad = {"AR22", "AP102", "KL71", "PB88", "PB43"}                    # known contaminated offices
    assert not bad & set(vahan.office_code.unique())
    assert not vahan.office_name.str.contains("Fitness|Testing", case=False).any()
    assert not vahan.duplicated(["date", "office_code", "maker"]).any()


# ------------------------------------------------------------------ master panel integrity
def test_panel_integrity():
    p = PANEL
    assert p.groupby("dealer_id").date.nunique().eq(65).all(), "every territory has all 65 months"
    assert (p.oem_units <= p.car_market).all() and (p.car_market <= p.all_vehicles_market).all()
    assert (p[["oem_units", "car_market", "all_vehicles_market"]] >= 0).all().all()
    assert p.groupby("dealer_id").office_code.nunique().eq(1).all()
    assert p.groupby("office_code").dealer_id.nunique().eq(1).all()
    excluded = pd.read_csv(P / "excluded_admin_breaks.csv").office_code
    assert not set(excluded) & set(p.office_code), "jurisdiction-break territories must be excluded"
    assert not p.office_name.str.contains("Viu|Head Office|State Transport Authority", case=False).any()


def test_territory_names_unique_and_ids_consistent():
    last = HIST[HIST.as_of == HIST.as_of.max()]
    assert last.territory.is_unique and last.dealer_id.is_unique and len(last) == PANEL.dealer_id.nunique()
    for df in (OPS, TGT):
        assert set(df.dealer_id) == set(PANEL.dealer_id)
    assert HIST.groupby("as_of").size().nunique() == 1


# ------------------------------------------------------------------ app inputs recomputed independently
def _window(end, n, lag=0):
    end = pd.Timestamp(end) - pd.DateOffset(months=lag)
    return pd.date_range(end - pd.DateOffset(months=n - 1), end, freq="MS")


SAMPLE = sorted(PANEL.dealer_id.unique())[::37]


@pytest.mark.parametrize("as_of", ["2023-06-01", "2023-12-01", "2024-05-01"])
def test_history_inputs_match_independent_recompute(as_of):
    h = HIST[HIST.as_of == as_of].set_index("dealer_id")
    for d in SAMPLE:
        g = PANEL[PANEL.dealer_id == d].set_index("date")
        t = TGT[TGT.dealer_id == d].set_index("date").target_units
        o = OPS[OPS.dealer_id == d].set_index("date")
        exp = {
            "units_l12m": g.oem_units.reindex(_window(as_of, 12)).sum(),
            "units_p12m": g.oem_units.reindex(_window(as_of, 12, 12)).sum(),
            "car_market_l12m": g.car_market.reindex(_window(as_of, 12)).sum(),
            "car_market_p12m": g.car_market.reindex(_window(as_of, 12, 12)).sum(),
            "units_last6m": g.oem_units.reindex(_window(as_of, 6)).sum(),
            "units_last6m_prev_year": g.oem_units.reindex(_window(as_of, 6, 12)).sum(),
            "units_last3m": g.oem_units.reindex(_window(as_of, 3)).sum(),
            "target_l12m": t.reindex(_window(as_of, 12)).sum(),
            "target_last3m": t.reindex(_window(as_of, 3)).sum(),
            "inventory_days": o.inventory_days.reindex(_window(as_of, 3)).mean(),
            "payment_delay_days": o.payment_delay_days.reindex(_window(as_of, 3)).mean(),
            "csi_score": o.csi_score.reindex(_window(as_of, 3)).mean(),
        }
        for k, v in exp.items():
            assert h.loc[d, k] == pytest.approx(v, abs=0.06), (as_of, d, k)
        cur, prv = g.reindex(_window(as_of, 12)), g.reindex(_window(as_of, 12, 12))
        gains = {c: cur[c].sum() / cur.car_market.sum() - prv[c].sum() / prv.car_market.sum() for c in COMPETITORS}
        assert h.loc[d, "top_competitor_gainer"] == max(gains, key=gains.get)


def test_retail_in_ops_is_the_real_vahan_series():
    m = OPS.merge(PANEL[["dealer_id", "date", "oem_units"]], on=["dealer_id", "date"])
    assert (m.retail_units == m.oem_units).all()


def test_metric_formulas_by_hand():
    last = HIST[HIST.as_of == HIST.as_of.max()].iloc[:50]
    m = E.derive_metrics(last)
    for _, r in m.iterrows():
        pl, pp = r.units_l12m / r.car_market_l12m, r.units_p12m / r.car_market_p12m
        se = np.sqrt(E.SHARE_DISPERSION * (pl * (1 - pl) / r.car_market_l12m + pp * (1 - pp) / r.car_market_p12m))
        assert r.share_z == pytest.approx(np.clip((pl - pp) / se, -6, 6), abs=1e-9)
        assert r.share_change_pp == pytest.approx((pl - pp) * 100)
        assert r.target_ach_pct == pytest.approx(r.units_l12m / r.target_l12m * 100)
        assert r.revenue_gap_cr == pytest.approx(max(r.target_l12m - r.units_l12m, 0) * E.REVENUE_PER_UNIT_LAKH / 100)
        assert r.excess_stock_units == pytest.approx(max(r.inventory_days - E.NORM_STOCK_DAYS, 0) / 30.4 * r.units_l12m / 12)


def test_targets_are_reasonable():
    last = HIST[HIST.as_of == HIST.as_of.max()]
    ach = last.units_l12m / last.target_l12m
    assert 0.9 < ach.median() < 1.05 and (ach.between(0.3, 3)).all()


# ------------------------------------------------------------------ synthetic layer is bounded & realistic
def test_synthetic_bounds():
    assert OPS.inventory_days.between(0, 240).all() and OPS.payment_delay_days.between(0, 90).all()
    assert OPS.csi_score.between(55, 98).all() and (OPS.complaints >= 0).all()
    last = OPS[OPS.date == OPS.date.max()]
    assert 30 <= last.inventory_days.median() <= 55, "calibrated to FADA-reported dealer stock levels"


def test_injected_events_only_after_feb_2024():
    gt = pd.read_csv(A / "ground_truth_events.csv").fillna("")
    credit = gt[gt.injected_event == "credit_squeeze"].dealer_id
    o = OPS[OPS.dealer_id.isin(credit)].groupby("date").payment_delay_days.median()
    base = OPS[~OPS.dealer_id.isin(credit)].groupby("date").payment_delay_days.median()
    assert (o.loc[:"2024-01-01"] - base.loc[:"2024-01-01"]).abs().max() < 4
    assert (o.loc["2024-04-01":] - base.loc["2024-04-01":]).min() > 8


# ------------------------------------------------------------------ privacy & deliverables
def test_telangana_roster_has_no_pii():
    ts = pd.read_csv(P / "ts_dealers_clean.csv", dtype=str)
    assert len(ts) == 1246
    assert not {"repEmail", "contactPhone", "address1"} & set(ts.columns)
    assert not ts.apply(lambda c: c.str.contains("@", na=False)).any().any()
    assert not ts.drop(columns=["pincode"]).apply(lambda c: c.str.contains(r"\d{10}", na=False)).any().any(), \
        "phone numbers hidden in free-text fields must be scrubbed"


def test_published_workbook_has_no_pii():
    x = pd.ExcelFile(ROOT / "data/master/dealer_master_dataset.xlsx")
    for sh in x.sheet_names:
        t = pd.read_excel(x, sh, dtype=str)
        assert not t.apply(lambda c: c.str.contains(r"@|\b[6-9]\d{9}\b", na=False, regex=True)).any().any(), sh


def test_master_workbook():
    x = pd.ExcelFile(ROOT / "data/master/dealer_master_dataset.xlsx")
    assert {"README", "Data_Dictionary", "Dealer_Snapshot", "Monthly_Panel", "TS_Dealer_Roster"} <= set(x.sheet_names)
    assert len(pd.read_excel(x, "Dealer_Snapshot")) == PANEL.dealer_id.nunique()
