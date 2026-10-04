"""'Your data' mode: monthly-file intake (unit tests) and the full app running on uploaded data (AppTest)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))
import engine as E  # noqa: E402
import ingest as I  # noqa: E402
from apptest_utils import new_app, goto, problems  # noqa: E402

SAMPLE = (ROOT / "data/app/sample_monthly_upload.csv").read_bytes()
HIST = pd.read_parquet(ROOT / "data/app/dealer_inputs_history.parquet")


def csv(df, **kw):
    return df.to_csv(index=False, **kw).encode("utf-8")


def sample_df():
    return pd.read_csv(io.BytesIO(SAMPLE))


def minimal(n_dealers=3, months=24, start="2022-06"):
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_dealers):
        for m in pd.date_range(start, periods=months, freq="MS"):
            u = int(rng.integers(40, 120))
            rows.append(dict(dealer_id=f"X{d}", month=m.strftime("%Y-%m"), units_sold=u, target_units=90,
                             local_car_market=u * 3))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ accuracy: same engine, same answer
def test_sample_upload_reproduces_builtin_scores_exactly():
    ds = I.build_dataset("s.csv", SAMPLE)
    assert ds["meta"]["dealers"] == 150 and not ds["meta"]["warnings"]
    for t, g in ds["hist"].groupby("as_of"):
        a = E.score_dealers(g).set_index("dealer_id")
        b = E.score_dealers(HIST[HIST.as_of == t]).set_index("dealer_id").loc[a.index]
        assert (a.health_score - b.health_score).abs().max() < 1e-9, t
        assert (a.tier == b.tier).all()
        assert all(sorted(x["code"] for x in a.loc[d].alerts) == sorted(x["code"] for x in b.loc[d].alerts) for d in a.index)


def test_excel_and_semicolon_and_month_formats_give_same_result():
    base = I.build_dataset("a.csv", SAMPLE)["hist"]
    df = sample_df()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    variants = {
        "xlsx": ("a.xlsx", buf.getvalue()),
        "semicolon": ("a.csv", csv(df, sep=";")),
        "yyyymm int": ("a.csv", csv(df.assign(month=df.month.str.replace("-", "").astype(int)))),
        "first-of-month": ("a.csv", csv(df.assign(month=df.month + "-01"))),
        "Mon-YYYY": ("a.csv", csv(df.assign(month=pd.to_datetime(df.month).dt.strftime("%b-%Y")))),
        "upper-case headers": ("a.csv", csv(df.rename(columns=lambda c: c.upper() if not c.startswith("comp_") else c))),
        "shuffled rows": ("a.csv", csv(df.sample(frac=1, random_state=4))),
    }
    for label, (name, data) in variants.items():
        h = I.build_dataset(name, data)["hist"]
        pd.testing.assert_series_equal(
            h.set_index(["as_of", "dealer_id"]).units_l12m.sort_index(),
            base.set_index(["as_of", "dealer_id"]).units_l12m.sort_index(), obj=label)


# ------------------------------------------------------------------ minimal / partial files
def test_minimal_file_required_columns_only():
    ds = I.build_dataset("m.csv", csv(minimal()))
    h = ds["hist"]
    assert h.as_of.nunique() == 1, "exactly 24 months -> one scoring date"
    assert ds["ops"] is None and ds["meta"]["competitors"] == [] and ds["meta"]["ops_columns"] == []
    s = E.score_dealers(h)
    assert s.health_score.between(0, 100).all() and s.tier.isin(E.TIERS).all()
    assert (s.data_coverage < 1).all()
    assert set(h.zone) == {"All"} and set(h.state_name) == {"All"}


def test_gappy_dealer_is_skipped_with_warning():
    df = minimal(n_dealers=3, months=30)
    df = df[~((df.dealer_id == "X1") & (df.month == "2024-10"))]        # hole in the last 24 months
    ds = I.build_dataset("g.csv", csv(df))
    last = ds["hist"][ds["hist"].as_of == ds["hist"].as_of.max()]
    assert "X1" not in set(last.dealer_id) and len(last) == 2
    assert any("skipped" in w for w in ds["meta"]["warnings"])


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["target_units"]), "Missing required"),
    (lambda d: d[d.month < d.month.max()], "at least 24"),                  # 23 months
    (lambda d: pd.concat([d, d.head(1)]), "once per month"),
    (lambda d: d.assign(units_sold=d.local_car_market + 1), "above local_car_market"),
    (lambda d: d.assign(month="not a date"), "month"),
    (lambda d: d.assign(target_units=0), "24 consecutive months"),
])
def test_rejections(mutate, msg):
    with pytest.raises(E.InputError, match=msg):
        I.build_dataset("r.csv", csv(mutate(minimal())))


def test_hostile_values_become_warnings_not_crashes():
    df = minimal().astype({"units_sold": object})
    df.loc[0, "units_sold"] = "abc"                 # text in a number column
    df["inventory_days"] = -3
    df["dealer_name"] = "<b>bold</b>"
    ds = I.build_dataset("h.csv", csv(df))
    w = " ".join(ds["meta"]["warnings"])
    assert "non-numeric" in w and "negative" in w
    assert "skipped" in w, "the dealer with a now-missing month is skipped, not mis-scored"


def test_template_is_a_valid_upload():
    ds = I.build_dataset("t.csv", csv(I.template_monthly()))
    assert ds["meta"]["dealers"] == 2 and ds["meta"]["competitors"] == ["Hyundai", "Tata Motors"]


# ------------------------------------------------------------------ the whole app on uploaded data
VIEWS = ["Attention Board", "Dealer Deep-Dive", "Quick Assess", "Network Insights", "Method & Data"]


def _upload(at, name, data):
    at.session_state["data_source"] = "Your data"
    at.session_state["byod_file"] = (name, data)
    at.run()
    return at


def _txt(at):
    return " ".join(m.value for m in at.markdown)


def test_app_landing_page_without_a_file():
    at = new_app()
    at.session_state["data_source"] = "Your data"
    at.run()
    assert not problems(at) and "Bring your own dealer data" in _txt(at)


@pytest.mark.parametrize("label, name, data", [
    ("full sample", "sample.csv", SAMPLE),
    ("minimal 24 months", "minimal.csv", csv(minimal())),
    ("minimal 30 months", "minimal30.csv", csv(minimal(months=30))),
])
def test_app_every_view_on_uploaded_data(label, name, data):
    at = _upload(new_app(), name, data)
    assert not problems(at), (label, problems(at))
    assert "YOUR DATA" in _txt(at) and "SYNTHETIC" not in _txt(at), "uploads must never be labelled synthetic"
    for v in VIEWS:
        goto(at, v)
        assert not problems(at), (label, v, problems(at))
    goto(at, "Dealer Deep-Dive")
    for d in at.selectbox(key="dealer_id").options[:8]:
        at.session_state["dealer_id"] = d
        at.run()
        assert not problems(at), (label, d, problems(at))
        assert next(m for m in at.metric if m.label == "Health after changes").delta == "+0 pts"


def test_app_bad_upload_shows_message_and_recovers():
    at = _upload(new_app(), "bad.csv", b"dealer_id,month\nA,2024-01\n")
    assert not at.exception
    assert any("Missing required" in e.value for e in at.error)
    at.session_state["byod_file"] = ("sample.csv", SAMPLE)
    at.run()
    assert not problems(at)


def test_app_switch_between_sources_keeps_working():
    at = _upload(new_app(), "sample.csv", SAMPLE)
    goto(at, "Dealer Deep-Dive")
    at.session_state["data_source"] = "Sample network"
    at.run()
    assert not problems(at)
    at.session_state["data_source"] = "Your data"
    at.run()
    assert not problems(at), "the uploaded file is remembered when switching back"
