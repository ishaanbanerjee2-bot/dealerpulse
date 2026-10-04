"""End-to-end UI stress tests with Streamlit AppTest (headless)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apptest_utils import new_app, goto, problems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = pd.read_parquet(ROOT / "data/app/dealer_inputs_history.parquet")
VIEWS = ["Attention Board", "Dealer Deep-Dive", "Quick Assess", "Network Insights", "Method & Data"]


@pytest.fixture(scope="module")
def at():
    a = new_app()
    assert not problems(a)
    return a


def _texts(a):
    return " ".join(m.value for m in a.markdown)


def test_every_view_renders(at):
    for v in VIEWS:
        goto(at, v)
        assert not problems(at), (v, problems(at))
    goto(at, "Attention Board")
    assert "Which dealers need attention?" in _texts(at)


def test_every_zone_and_state_on_board_and_deep_dive(at):
    zones = ["All India"] + sorted(HIST.zone.unique())
    for z in zones:
        at.selectbox(key="zone").set_value(z).run()
        states = [o for o in at.selectbox(key="state").options]
        for s in states:                       # every state in every zone
            at.selectbox(key="state").set_value(s).run()
            for v in ("Attention Board", "Dealer Deep-Dive", "Network Insights"):
                goto(at, v)
                assert not problems(at), (z, s, v, problems(at))
    at.selectbox(key="zone").set_value("All India").run()
    at.selectbox(key="state").set_value("All states").run()


def test_state_resets_when_zone_changes(at):
    at.selectbox(key="zone").set_value("South").run()
    at.selectbox(key="state").set_value("Kerala").run()
    at.selectbox(key="zone").set_value("North").run()
    assert at.selectbox(key="state").value == "All states"
    assert not problems(at)
    at.selectbox(key="zone").set_value("All India").run()


def test_every_as_of_month(at):
    goto(at, "Attention Board")
    months = [pd.Timestamp(m) for m in sorted(HIST.as_of.unique())][-6:]
    for m in months:
        for v in ("Attention Board", "Dealer Deep-Dive", "Network Insights"):
            at.session_state["as_of"] = m
            goto(at, v)
            assert not problems(at), (m, v, problems(at))
    at.session_state["as_of"] = months[-1]
    at.run()


def test_deep_dive_many_dealers(at):
    goto(at, "Dealer Deep-Dive")
    latest = HIST[HIST.as_of == HIST.as_of.max()].sort_values("units_l12m")
    ids = (list(latest.dealer_id.head(15)) + list(latest.dealer_id.tail(15)) +
           list(latest.dealer_id.sample(70, random_state=7)))
    for d in dict.fromkeys(ids):
        at.session_state["dealer_id"] = d
        at.run()
        assert not problems(at), (d, problems(at))
        assert at.selectbox(key="dealer_id").value == d


def test_prev_next_buttons_cycle(at):
    goto(at, "Dealer Deep-Dive")
    first = at.selectbox(key="dealer_id").value
    at.button[0].click().run()          # Prev from #1 wraps to the last dealer
    assert at.selectbox(key="dealer_id").value != first
    at.button[1].click().run()          # Next returns
    assert at.selectbox(key="dealer_id").value == first
    assert not problems(at)


def test_what_if_sliders_extremes(at):
    goto(at, "Dealer Deep-Dive")
    sliders = [s for s in at.slider if str(s.key).startswith("wi_")]
    assert sliders, "what-if sliders should exist"
    for s in sliders:
        for v in (s.min, s.max):
            s.set_value(v)
        at.run()
        assert not problems(at), (s.key, problems(at))


@pytest.mark.parametrize("weights", [
    {"sales": 0, "momentum": 0, "financial": 0, "customer": 0},   # all zero -> default + warning
    {"sales": 60, "momentum": 0, "financial": 0, "customer": 0},
    {"sales": 0, "momentum": 0, "financial": 60, "customer": 0},
    {"sales": 60, "momentum": 60, "financial": 60, "customer": 60},
])
def test_weight_extremes(weights):
    a = new_app()
    for p, v in weights.items():
        a.slider(key=f"w_{p}").set_value(v)
    a.run()
    for v in ("Attention Board", "Dealer Deep-Dive", "Method & Data"):
        goto(a, v)
        assert not problems(a), (weights, v, problems(a))
    if sum(weights.values()) == 0:
        assert any("All weights are zero" in w.value for w in a.warning)


def test_focus_filters_that_match_nothing(at):
    goto(at, "Attention Board")
    at.multiselect(key="issues").set_value(["Market-driven dip"]).run()
    at.text_input(key="board_search").set_value("zzzz-no-such-dealer").run()
    assert not problems(at)
    at.text_input(key="board_search").set_value("").run()
    at.multiselect(key="issues").set_value([]).run()
    assert not problems(at)


def test_search_special_characters(at):
    goto(at, "Attention Board")
    for q in ["(", "[a-z]+", "*", "·", "\\", "MS-RJ"]:
        at.text_input(key="board_search").set_value(q).run()
        assert not problems(at), (q, problems(at))
    at.text_input(key="board_search").set_value("").run()


def _qa(a):
    return {n.label: n for n in a.number_input}


def test_quick_assess_default_and_submit(at):
    goto(at, "Quick Assess")
    assert not problems(at)
    assert "Manual assessment" in _texts(at)
    submit = [b for b in at.button if "Assess dealer" in str(b.label)]
    assert submit, "form submit button should render"
    submit[0].click().run()
    assert not problems(at)


def test_quick_assess_invalid_market_shows_message_not_crash(at):
    goto(at, "Quick Assess")
    ni = _qa(at)
    ni["Units sold · last 12m"].set_value(400000)
    ni["Local car market · last 12m"].set_value(1000)
    at.run()
    assert not at.exception
    assert any("car_market" in e.value for e in at.error)
    ni = _qa(at)
    ni["Units sold · last 12m"].set_value(1450)
    ni["Local car market · last 12m"].set_value(4200)
    at.run()
    assert not problems(at)


def test_quick_assess_extremes(at):
    goto(at, "Quick Assess")
    for label, v in [("Inventory cover (days)", 365.0), ("Payment delay (avg days past due)", 180.0),
                     ("Service CSI (0–100)", 0.0), ("Complaints per 100 units", 50.0), ("Units sold · last 3m", 0)]:
        _qa(at)[label].set_value(v)
        at.run()
        assert not problems(at), (label, problems(at))
    assert "Critical" in _texts(at)


# ------------------------------------------------------------------ regressions fixed in the final audit
def _metric(a, label):
    return next(m for m in a.metric if m.label == label)


def test_what_if_untouched_equals_current_score(at):
    goto(at, "Dealer Deep-Dive")
    latest = HIST[HIST.as_of == HIST.as_of.max()]
    # include the highest-stock dealers, whose values sit outside the default slider ranges
    ids = list(latest.nlargest(5, "inventory_days").dealer_id) + list(latest.dealer_id.sample(20, random_state=3))
    for d in ids:
        at.session_state["dealer_id"] = d
        at.run()
        assert not problems(at)
        assert _metric(at, "Health after changes").delta == "+0 pts", d
        assert _metric(at, "Tier after changes").delta == "unchanged", d


def test_health_trend_never_shows_months_after_as_of(at):
    import json
    goto(at, "Dealer Deep-Dive")
    for m in [pd.Timestamp("2023-12-01"), pd.Timestamp("2024-02-01")]:
        at.session_state["as_of"] = m
        at.run()
        assert not problems(at)
        found = False
        for el in at.get("plotly_chart"):
            spec = json.loads(el.proto.spec)
            for tr in spec.get("data", []):
                if "Health %{y:.0f}" in str(tr.get("hovertemplate", "")):
                    found = True
                    assert max(pd.to_datetime(tr["x"])) <= m
        assert found, "health trend chart should render"
    at.session_state["as_of"] = pd.Timestamp(sorted(HIST.as_of.unique())[-1])
    at.run()


def test_labels_are_accurate(at):
    goto(at, "Attention Board")
    assert "MODELLED" in " ".join(m.value for m in at.markdown), "revenue gap uses a modelled target"
    goto(at, "Method & Data")
    txt = " ".join(m.value for m in at.markdown)
    assert "20 RTOs contaminated" in txt and "16 territories whose jurisdiction changed" in txt
