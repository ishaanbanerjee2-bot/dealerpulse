"""Unit + property tests for the scoring engine (no Streamlit)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import engine as E  # noqa: E402
import ui as U  # noqa: E402

HIST = pd.read_parquet(ROOT / "data/app/dealer_inputs_history.parquet")
LATEST = HIST[HIST.as_of == HIST.as_of.max()]


def one(**over):
    """A complete, ordinary dealer row with overrides."""
    row = E.template_frame().iloc[1].to_dict()
    row.update(over)
    return pd.DataFrame([row])


def score(df, **kw):
    clean, _ = E.validate_inputs(df)
    return E.score_dealers(clean, volume_ref=LATEST.units_l12m, **kw)


@pytest.fixture(scope="module")
def network():
    return E.score_dealers(LATEST, volume_ref=LATEST.units_l12m)


# ------------------------------------------------------------------ rules & weights
@pytest.mark.parametrize("rule", E.RULES, ids=lambda r: r.metric)
def test_rule_score_bounded_and_directional(rule):
    xs = np.linspace(min(rule.bad, rule.good) - 100, max(rule.bad, rule.good) + 100, 401)
    s = E.rule_score(xs, rule)
    assert np.nanmin(s) >= 0 and np.nanmax(s) <= 100
    assert E.rule_score([rule.bad], rule)[0] == pytest.approx(0)
    assert E.rule_score([rule.good], rule)[0] == pytest.approx(100)
    diffs = np.diff(s) if rule.good > rule.bad else -np.diff(s)
    assert (diffs >= -1e-9).all(), "score must move monotonically from red line to good level"
    assert np.isnan(E.rule_score([np.nan], rule)[0])


def test_rule_weights_sum_to_one_per_pillar():
    for p in E.PILLARS:
        assert sum(r.weight for r in E.RULES if r.pillar == p) == pytest.approx(1.0)


@pytest.mark.parametrize("w", [None, {}, {k: 0 for k in E.PILLARS}, {k: -5 for k in E.PILLARS},
                               {"sales": float("inf")}, {"sales": 10}])
def test_normalise_weights_always_valid(w):
    n = E.normalise_weights(w)
    assert set(n) == set(E.PILLARS) and sum(n.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in n.values())


def test_single_pillar_weight_equals_that_pillar(network):
    s = E.score_dealers(LATEST, {"financial": 1, "sales": 0, "momentum": 0, "customer": 0}, LATEST.units_l12m)
    np.testing.assert_allclose(s.health_score, s.pillar_financial, atol=1e-9)


# ------------------------------------------------------------------ validation
def test_missing_required_column():
    with pytest.raises(E.InputError, match="Missing required"):
        E.validate_inputs(one().drop(columns=["target_l12m"]))


def test_empty_frame():
    with pytest.raises(E.InputError):
        E.validate_inputs(pd.DataFrame())


def test_duplicate_ids():
    with pytest.raises(E.InputError, match="Duplicate"):
        E.validate_inputs(pd.concat([one(), one()]))


def test_blank_id():
    with pytest.raises(E.InputError, match="dealer_id"):
        E.validate_inputs(one(dealer_id="  "))


def test_market_smaller_than_units_rejected():
    with pytest.raises(E.InputError, match="car_market"):
        E.validate_inputs(one(car_market_l12m=100, units_l12m=500))


def test_required_non_numeric_rejected():
    with pytest.raises(E.InputError, match="units_p12m"):
        E.validate_inputs(one(units_p12m="abc"))


def test_optional_non_numeric_and_negative_become_missing_with_warning():
    clean, warns = E.validate_inputs(one(inventory_days="lots", payment_delay_days=-4))
    assert np.isnan(clean.inventory_days.iloc[0]) and np.isnan(clean.payment_delay_days.iloc[0])
    assert any("inventory_days" in w for w in warns) and any("payment_delay_days" in w for w in warns)


def test_percent_and_csi_ranges():
    with pytest.raises(E.InputError):
        E.validate_inputs(one(aged_stock_pct=140))
    with pytest.raises(E.InputError):
        E.validate_inputs(one(csi_score=850))


def test_formatted_numbers_accepted():
    clean, warns = E.validate_inputs(one(units_l12m="2,600", aged_stock_pct="6%"))
    assert clean.units_l12m.iloc[0] == 2600 and clean.aged_stock_pct.iloc[0] == 6 and not warns


def test_column_names_normalised():
    df = one()
    df.columns = [c.upper().replace("_", " ") for c in df.columns]
    clean, _ = E.validate_inputs(df)
    assert "units_l12m" in clean


# ------------------------------------------------------------------ upload parsing
def _csv(df, sep=",", enc="utf-8"):
    return df.to_csv(index=False, sep=sep).encode(enc)


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_read_table_delimiters(sep):
    df = E.read_table("x.csv", _csv(E.template_frame(), sep))
    assert len(df) == 2 and "units_l12m" in df.columns


def test_read_table_latin1_and_bom():
    t = E.template_frame().assign(dealer_name=["Société Motors", "Ünited Autos"])
    assert len(E.read_table("a.csv", _csv(t, enc="latin-1"))) == 2
    assert E.read_table("b.csv", b"\xef\xbb\xbf" + _csv(t)).columns[0] == "dealer_id"


def test_read_table_excel():
    buf = io.BytesIO()
    E.template_frame().to_excel(buf, index=False)
    assert len(E.read_table("t.xlsx", buf.getvalue())) == 2


@pytest.mark.parametrize("data", [b"", b"\n\n", b"just some text without structure"])
def test_read_table_garbage(data):
    with pytest.raises(E.InputError):
        df = E.read_table("bad.csv", data)
        E.validate_inputs(df)


def test_read_table_corrupt_excel():
    with pytest.raises(E.InputError):
        E.read_table("bad.xlsx", b"PK\x03\x04 not really a workbook")


def test_header_only_csv():
    with pytest.raises(E.InputError):
        E.validate_inputs(E.read_table("h.csv", (",".join(E.REQUIRED) + "\n").encode()))


def test_upload_roundtrip_scores():
    raw = E.read_table("t.csv", _csv(E.template_frame()))
    res = score(raw)
    assert res.tier.isin(E.TIERS).all() and res.health_score.between(0, 100).all()


# ------------------------------------------------------------------ scoring behaviour
def test_minimal_required_columns_only():
    df = one()[E.REQUIRED]
    res = score(df).iloc[0]
    assert 0 <= res.health_score <= 100
    assert res.data_coverage < 1
    assert res.tier in E.TIERS


def test_zero_units_both_years_no_crash():
    res = score(one(units_l12m=0, units_p12m=0, units_last6m=0, units_last6m_prev_year=0, units_last3m=0)).iloc[0]
    assert 0 <= res.health_score <= 100 and res.tier in E.TIERS


def test_extreme_values_are_bounded_and_critical():
    res = score(one(inventory_days=10000, aged_stock_pct=100, payment_delay_days=1000, csi_score=0,
                    complaints_per_100_units=999, units_l12m=10, target_l12m=100000)).iloc[0]
    assert 0 <= res.health_score <= 100
    assert res.tier == "Critical"


def test_template_rows_behave_sensibly():
    res = score(E.template_frame()).set_index("dealer_id")
    bad, good = res.loc["DLR-001"], res.loc["DLR-002"]
    assert bad.tier in ("Critical", "Watch") and good.tier in ("Strong", "Stable")
    assert {a["code"] for a in bad.alerts} >= {"INVENTORY", "PAYMENT"}
    assert not [a for a in good.alerts if a["severity"] == "high"]
    assert bad.health_score < good.health_score


def test_improving_any_synthetic_metric_never_lowers_score():
    base = score(E.template_frame().iloc[[0]]).iloc[0].health_score
    for col, better in [("inventory_days", 30), ("aged_stock_pct", 2), ("payment_delay_days", 1), ("csi_score", 92),
                        ("complaints_per_100_units", 0.2)]:
        t = E.template_frame().iloc[[0]].copy()
        t[col] = better
        assert score(t).iloc[0].health_score >= base - 1e-9, col


def test_market_dip_gets_support_not_blame():
    # sales -10%, market -12%, share held -> market-driven, support action
    res = score(one(units_l12m=900, units_p12m=1000, car_market_l12m=2640, car_market_p12m=3000,
                    target_l12m=1050)).iloc[0]
    codes = {a["code"] for a in res.alerts}
    assert "MARKET_DIP" in codes and "SHARE_LOSS" not in codes
    below = [a for a in res.alerts if a["code"] == "BELOW_TARGET"]
    assert all(a["action_type"] == "Support" for a in below)


def test_share_loss_needs_statistical_significance():
    # same -3 pp share drop: not significant in a tiny market, significant in a big one
    small = score(one(units_l12m=105, units_p12m=120, car_market_l12m=300, car_market_p12m=312,
                      target_l12m=110)).iloc[0]
    big = score(one(units_l12m=10500, units_p12m=12000, car_market_l12m=30000, car_market_p12m=31200,
                    target_l12m=11000)).iloc[0]
    assert "SHARE_LOSS" not in {a["code"] for a in small.alerts}
    assert "SHARE_LOSS" in {a["code"] for a in big.alerts}


def test_reliability_shrinkage_dampens_small_dealers():
    m = E.derive_metrics(E.validate_inputs(pd.concat([
        one(dealer_id="S", units_l12m=80, units_p12m=100, car_market_l12m=1000, car_market_p12m=1000),
        one(dealer_id="L", units_l12m=8000, units_p12m=10000, car_market_l12m=100000, car_market_p12m=100000)]))[0])
    s, l = m.set_index("dealer_id").loc["S"], m.set_index("dealer_id").loc["L"]
    assert s.growth_vs_market_pp == pytest.approx(l.growth_vs_market_pp)
    assert abs(s.growth_vs_market_adj) < abs(l.growth_vs_market_adj)


def test_root_cause_diagnosis():
    res = score(E.template_frame().iloc[[0]]).iloc[0]
    codes = {a["code"] for a in res.alerts if a["severity"] in ("high", "medium")}
    if codes & E.SALES_CODES and codes & E.FIN_CODES:
        assert res.primary_issue == E.ROOT_CAUSE
        assert "Root cause" in E.summary_sentence(res)


# ------------------------------------------------------------------ whole-network invariants
def test_network_invariants(network):
    s = network
    assert len(s) == LATEST.dealer_id.nunique()
    assert s.health_score.between(0, 100).all() and s.health_score.notna().all()
    assert s.tier.isin(E.TIERS).all()
    assert sorted(s.priority_rank) == list(range(1, len(s) + 1))
    for p in E.PILLARS:
        assert s[f"pillar_{p}"].between(0, 100).all()
    assert (s.data_coverage > 0.999).all(), "sample network has every input"


def test_attention_dealers_always_have_an_explanation(network):
    att = network[network.tier.isin(["Critical", "Watch"])]
    assert (att.primary_issue != "On track").all()
    for _, r in att.iterrows():
        assert E.summary_sentence(r)


def test_every_alert_is_actionable(network):
    for fl in network.alerts:
        for a in fl:
            assert a["action"] and a["owner"] and a["horizon"] and a["evidence"]
            assert a["action_type"] in ("Corrective", "Support", "Grow")
            assert "nan" not in a["evidence"].lower()


def test_tier_distribution_is_usable(network):
    share = network.tier.isin(["Critical", "Watch"]).mean()
    assert 0.10 <= share <= 0.40, f"attention share {share:.0%} would be useless to a manager"
    assert (network.tier == "Critical").mean() <= 0.20


def test_ground_truth_recall(network):
    gt = pd.read_csv(ROOT / "data/app/ground_truth_events.csv").fillna("")
    s = network.merge(gt, on="dealer_id")
    want = {"service_disruption": {"SERVICE", "COMPLAINTS"}, "credit_squeeze": {"PAYMENT"}, "stock_push": {"INVENTORY"}}
    for ev, codes in want.items():
        g = s[s.injected_event == ev]
        recall = g.alerts.apply(lambda fl: any(f["code"] in codes for f in fl)).mean()
        assert recall >= 0.8, f"{ev} recall {recall:.0%}"


def test_history_scores_every_month():
    h = E.score_history(HIST, volume_ref=LATEST.units_l12m)
    assert h.as_of.nunique() == HIST.as_of.nunique()
    assert h.health_score.between(0, 100).all()


def test_scoring_is_deterministic():
    a = E.score_dealers(LATEST, volume_ref=LATEST.units_l12m)
    b = E.score_dealers(LATEST.sample(frac=1, random_state=1), volume_ref=LATEST.units_l12m)
    pd.testing.assert_series_equal(a.set_index("dealer_id").health_score.sort_index(),
                                   b.set_index("dealer_id").health_score.sort_index())


# ------------------------------------------------------------------ ui safety
def test_html_is_escaped():
    out = U.md_bold_to_html("<script>alert(1)</script> **bold**")
    assert "<script>" not in out and "<b>bold</b>" in out
    assert "<img" not in U.esc('<img src=x onerror=alert(1)>')


# ------------------------------------------------------------------ regressions fixed in the final audit
def test_ok_accepts_every_real_number_type():
    assert all(E._ok(x) for x in [np.int64(3), 3, 2.5, np.float32(1.5), np.uint8(7)])
    assert not any(E._ok(x) for x in [True, np.bool_(True), np.nan, None, "5", np.inf, -np.inf])


def test_integer_inputs_still_raise_alerts():
    row = E.template_frame().iloc[[0]].copy()
    row[["inventory_days", "payment_delay_days", "csi_score"]] = np.array([[100, 30, 65]], dtype=np.int64)
    for df in (row, E.validate_inputs(row)[0]):           # what-if path (unvalidated) and upload path
        codes = {a["code"] for a in E.score_dealers(df).iloc[0].alerts}
        assert {"INVENTORY", "PAYMENT", "SERVICE"} <= codes


def test_zero_target_rejected():
    with pytest.raises(E.InputError, match="target_l12m"):
        E.validate_inputs(one(target_l12m=0))


def test_excess_stock_matches_displayed_cover():
    m = E.derive_metrics(E.validate_inputs(one(inventory_days=95, units_l12m=1216, closing_stock=10))[0]).iloc[0]
    assert m.excess_stock_units == pytest.approx((95 - 35) / 30.4 * 1216 / 12)
    m2 = E.derive_metrics(E.validate_inputs(one(inventory_days=np.nan, closing_stock=304, units_l12m=1216))[0]).iloc[0]
    assert m2.excess_stock_units == pytest.approx((304 * 30.4 / (1216 / 12) - 35) / 30.4 * 1216 / 12)


# ------------------------------------------------------------------ upload screen (the exact function the UI calls)
def _up(df_or_bytes, name="f.csv", **kw):
    data = df_or_bytes if isinstance(df_or_bytes, bytes) else _csv(df_or_bytes)
    return E.score_upload(name, data, volume_ref=LATEST.units_l12m, **kw)


def test_upload_happy_path_and_columns():
    out, warns = _up(E.template_frame())
    assert list(out.columns) == ["Priority", "Dealer", "Tier", "Health", "Main issue", "Action",
                                 "Top recommended action", "Data coverage"]
    assert out.Dealer.tolist() == ["Example Motors", "Sample Autos"] and not warns
    assert out.Tier.isin(E.TIERS).all() and out["Top recommended action"].str.len().gt(10).all()


def test_upload_real_network_export_roundtrip():
    """A manager exports our own network inputs and re-uploads them: scores must match the app exactly."""
    raw = LATEST.drop(columns=["as_of"])
    out, _ = _up(raw.head(300))
    ref = E.score_dealers(LATEST.head(300), volume_ref=LATEST.units_l12m).set_index("territory").health_score.round(0)
    got = out.set_index("Dealer").Health
    pd.testing.assert_series_equal(got.sort_index(), ref.sort_index(), check_names=False)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["car_market_p12m"]), "Missing required"),
    (lambda d: d.assign(units_l12m=["", "x"]), "units_l12m"),
    (lambda d: d.assign(dealer_id=["A", "A"]), "Duplicate"),
    (lambda d: d.assign(car_market_l12m=[1, 1]), "car_market"),
    (lambda d: d.assign(target_l12m=[0, 5]), "target_l12m"),
    (lambda d: d.assign(csi_score=[200, 80]), "CSI"),
])
def test_upload_rejections(mutate, msg):
    with pytest.raises(E.InputError, match=msg):
        _up(mutate(E.template_frame()))


def test_upload_hostile_but_usable_content():
    t = E.template_frame()
    t["dealer_name"] = ['<script>alert("x")</script>', "   "]
    t["inventory_days"] = ["lots", -5]
    t["extra_unknown_column"] = "ignored"
    out, warns = _up(t)
    assert out.Dealer.tolist()[1] == "DLR-002", "blank names fall back to the dealer id"
    assert len(warns) == 2                                    # non-numeric + negative
    assert out.Health.between(0, 100).all()


def test_upload_limits_and_garbage():
    big = pd.concat([E.template_frame().iloc[[1]]] * 5001, ignore_index=True)
    big["dealer_id"] = [f"D{i}" for i in range(len(big))]
    with pytest.raises(E.InputError, match="up to"):
        _up(big)
    for name, data in [("x.csv", b""), ("x.csv", b"\x00\x01\x02\xff"), ("x.xlsx", b"not excel"), ("x.csv", b"a;b\n1;2")]:
        with pytest.raises(E.InputError):
            _up(data, name)


def test_upload_scale_performance():
    import time
    big = pd.concat([LATEST.drop(columns=["as_of"])] * 5, ignore_index=True).head(4500)
    big["dealer_id"] = [f"D{i}" for i in range(len(big))]
    t0 = time.time()
    out, _ = _up(big)
    assert len(out) == 4500 and time.time() - t0 < 15
