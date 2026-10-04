"""
Step 6 - Validation evidence shown on the app's Method page (data/app/validation.json).

  A. Real-data backtest: do the market signals the engine uses persist? (pooled origins Jun-22..May-23, 12m ahead)
  B. ML early-warning test: can a model predict a >1pp share loss in the next 12 months better than a simple rule?
     (train origins Dec-21..May-22, test origins Jan-23..May-23 - out of time)
  C. Synthetic ground truth: does the engine catch the injected emerging events?
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import engine as E  # noqa: E402

p = pd.read_csv(ROOT / "data/processed/master_panel.csv", parse_dates=["date"])
st = p.groupby("dealer_id").state_name.first()


def win(t_end, n=12):
    m = (p.date > t_end - pd.DateOffset(months=n)) & (p.date <= t_end)
    return p[m].groupby("dealer_id")[["oem_units", "car_market", "all_vehicles_market"]].sum()


def origin_frame(t0):
    a, b, fut = win(t0), win(t0 - pd.DateOffset(months=12)), win(t0 + pd.DateOffset(months=12))
    a6, b6 = win(t0, 6), win(t0 - pd.DateOffset(months=12), 6)
    sa, sb, sf = [x.oem_units / x.car_market for x in (a, b, fut)]
    se = np.sqrt(E.SHARE_DISPERSION * (sa * (1 - sa) / a.car_market + sb * (1 - sb) / b.car_market))
    stsh = a.groupby(st).oem_units.sum() / a.groupby(st).car_market.sum()
    fst = fut.groupby(st).oem_units.sum() / fut.groupby(st).car_market.sum()
    return pd.DataFrame({
        "t0": t0, "d1": (sa - sb) * 100, "z": (sa - sb) / se, "gap": (sa - st.map(stsh)) * 100, "share": sa * 100,
        "gvm": ((a.oem_units / b.oem_units) - (a.car_market / b.car_market)) * 100,
        "mkt_g": (a.car_market / b.car_market - 1) * 100, "mom6": (a6.oem_units / b6.oem_units - 1) * 100,
        "mom6_mkt": (a6.car_market / b6.car_market - 1) * 100, "logvol": np.log(a.oem_units),
        "carpen": a.car_market / a.all_vehicles_market, "fut": (sf - sa) * 100, "fut_gap": (sf - st.map(fst)) * 100,
    }).replace([np.inf, -np.inf], np.nan).dropna()


out = {}
# ---------------- A. persistence backtest
D = pd.concat([origin_frame(t) for t in pd.date_range("2022-06-01", "2023-05-01", freq="MS")])
sig_loss = D.z <= -1.96
out["backtest"] = {
    "origins": "monthly, Jun-2022 to May-2023 (12 origins), outcome = next 12 months",
    "n_observations": int(len(D)),
    "future_share_change_after_significant_loss_pp": round(float(D[sig_loss].fut.mean()), 2),
    "future_share_change_others_pp": round(float(D[~sig_loss].fut.mean()), 2),
    "future_share_change_after_any_loss_pp": round(float(D[D.d1 < 0].fut.mean()), 2),
    "future_share_change_after_gain_pp": round(float(D[D.d1 >= 0].fut.mean()), 2),
    "pct_still_below_state_after_12m_if_5pp_below": round(float((D[D.gap < -5].fut_gap < 0).mean() * 100), 1),
    "spearman_gap_persistence": round(float(D.gap.corr(D.fut_gap, method="spearman")), 3),
}

# ---------------- B. ML early-warning (out of time)
F = ["d1", "z", "gap", "share", "gvm", "mkt_g", "mom6", "mom6_mkt", "logvol", "carpen"]
A = pd.concat([origin_frame(t) for t in pd.date_range("2021-12-01", "2023-05-01", freq="MS")])
A["y"] = (A.fut < -1.0).astype(int)
tr, te = A[A.t0 <= "2022-05-01"], A[A.t0 >= "2023-01-01"]
res = {}
for name, mdl in {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
                  "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                      min_samples_leaf=50, random_state=0)}.items():
    mdl.fit(tr[F], tr.y)
    res[name] = round(float(roc_auc_score(te.y, mdl.predict_proba(te[F])[:, 1])), 3)
res["Rule: current share level"] = round(float(roc_auc_score(te.y, te.share)), 3)
res["Rule: last-12m share loss"] = round(float(roc_auc_score(te.y, -te.d1)), 3)
out["ml_early_warning"] = {"test_auc": res, "train_base_rate": round(float(tr.y.mean()), 3),
                           "test_base_rate": round(float(te.y.mean()), 3),
                           "decision": "Not deployed: no gain over a one-variable rule out of time; base rate shifted "
                                       "with Maruti's 2023-24 SUV recovery, so probabilities would be mis-calibrated."}

# ---------------- C. synthetic ground-truth recall
h = pd.read_parquet(ROOT / "data/app/dealer_inputs_history.parquet")
last = h[h.as_of == h.as_of.max()]
s = E.score_dealers(last, volume_ref=last.units_l12m)
gt = pd.read_csv(ROOT / "data/app/ground_truth_events.csv").fillna("")
s = s.merge(gt, on="dealer_id")
want = {"service_disruption": {"SERVICE", "COMPLAINTS"}, "credit_squeeze": {"PAYMENT"}, "stock_push": {"INVENTORY"}}
rec = {}
for ev, codes in want.items():
    g = s[s.injected_event == ev]
    rec[ev] = {"n": int(len(g)),
               "recall_pct": round(float(g.alerts.apply(lambda fl: any(f["code"] in codes for f in fl)).mean() * 100), 1),
               "in_attention_tiers_pct": round(float(g.tier.isin(["Critical", "Watch"]).mean() * 100), 1)}
out["ground_truth_recall"] = rec

# ---------------- D. significance filter effect
m = E.derive_metrics(last)
hidden = (m.sales_growth_pct >= 0) & (m.share_change_pp < 0)
out["significance"] = {
    "pct_share_changes_significant": round(float((m.share_z.abs() > 1.96).mean() * 100), 1),
    "hidden_risk_raw": int(hidden.sum()),
    "hidden_risk_significant": int((hidden & (m.share_z <= -1.96)).sum()),
}
out["expected_share_model"] = json.load(open(ROOT / "data/app/model_card.json"))

# ---------------- E. what was excluded during cleaning (shown on the Method page)
log = pd.read_csv(ROOT / "data/processed/vahan_cleaning_log.csv", usecols=["office_code", "office_name", "reason"])
clean_offices = pd.read_parquet(ROOT / "data/processed/vahan_clean.parquet", columns=["office_code", "office_name"]).drop_duplicates()
out["exclusions"] = {
    "contaminated_offices": int(log.loc[log.reason == "contaminated_office", "office_code"].nunique()),
    "fitness_centres": int(log.loc[log.reason == "non_rto_office", "office_code"].nunique()),
    "special_offices": int(clean_offices.office_name.str.contains("Viu|Head Office|State Transport Authority", case=False).sum()),
    "jurisdiction_breaks": int(len(pd.read_csv(ROOT / "data/processed/excluded_admin_breaks.csv"))),
}
out["tier_mix_latest"] = s.tier.value_counts().to_dict()
json.dump(out, open(ROOT / "data/app/validation.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "expected_share_model"}, indent=1))
