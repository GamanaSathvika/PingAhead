"""
train.py  -  attack-onset forecasting (attack_in_next_60s) on windowed Argus flows.

What this script does
---------------------
1. Builds episode masks per scenario:
     in_episode    : windows inside an attack episode (attack runs <= MAX_GAP apart are merged)
     post_episode  : the POST_WINDOWS windows after an episode (lag/rolling features still
                     contain attack traffic, so they are not "quiet" traffic)
     tail_row      : last HORIZON windows of a file (look-ahead label is incomplete)
     quiet         : the rows used for training AND evaluation (see build_masks)
2. Leave-one-scenario-out (LOSO) evaluation with a NESTED threshold:
     for each held-out scenario, the alert threshold is chosen using only the other
     scenarios (inner LOSO), then applied once to the held-out scenario.
   So FA/hr, episode recall and lead time are genuinely held-out numbers.
3. Reports episode recall (overall + per phase + Wilson 95% CI), lead time, and false
   alarms as ALERT EVENTS per hour (consecutive alert windows = 1 event) and per window.
4. Optional feature ablation (--ablate) to test dependence on response-ratio / volume features.
5. Trains the dashboard demo model (hold out scenario-5 and scenario-8 by default), picks
   its threshold from inner LOSO on the training scenarios only, and writes simulation CSVs.

Usage
-----
    python3 src/train.py                 # nested LOSO + demo export
    python3 src/train.py --ablate        # additionally run the feature ablation
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

WINDOW_SEC = 10
HORIZON = 6          # forecast horizon in windows (60 s)
MAX_GAP = 6          # attack runs separated by <= this many benign windows are one episode
POST_WINDOWS = 6     # windows after an episode whose lag/rolling features are contaminated
TARGET = "attack_in_next_60s"
BASELINE_COL = "no_resp_ratio_roll6_mean"   # simple rule baseline score
PHASE_LABELS = {0: "phase 1", 1: "phase 2", 2: "phase 3"}

# Never features: identifiers, labels, and anything computed from labels.
FORBIDDEN_COLS = ["attack_flow_ratio"]
META_AND_LABEL_COLS = [
    "StartTime", "scenario", "is_attack", "stage_id", TARGET,
    "in_episode", "episode_id", "phase", "is_pre_episode_window", "pre_for_episode",
    "post_episode", "tail_row", "quiet",
] + FORBIDDEN_COLS

# Ablation groups (prefix match on feature names).
RESPONSE_PREFIXES = ("no_resp", "state_", "DstPkts", "DstBytes", "src_bytes_ratio",
                     "dMeanPktSz", "PCRatio", "TcpRtt", "SynAck", "Loss")
SCALE_PREFIXES = ("TotBytes", "TotPkts", "SrcBytes", "DstBytes", "SrcPkts", "DstPkts",
                  "Rate", "flow_count", "max_flows_per_1s")


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
def build_masks(df, max_gap=MAX_GAP, horizon=HORIZON, post=POST_WINDOWS):
    """df must be sorted by (scenario, StartTime) with a clean RangeIndex."""
    df = df.copy()
    n = len(df)
    in_ep = np.zeros(n, int)
    ep_id = np.full(n, -1)
    phase = np.full(n, -1)
    pre = np.zeros(n, int)
    pre_for = np.full(n, -1)
    post_m = np.zeros(n, int)
    tail = np.zeros(n, int)
    atk_all = df["is_attack"].to_numpy()
    next_ep = 0

    for _, idx in df.groupby("scenario", sort=False).indices.items():
        idx = np.sort(idx)
        L = len(idx)
        tail[idx[max(0, L - horizon):]] = 1
        pos = np.where(atk_all[idx] == 1)[0]
        if len(pos) == 0:
            continue
        runs = np.split(pos, np.where(np.diff(pos) > max_gap + 1)[0] + 1)
        spans = []
        for k, r in enumerate(runs):
            a, b = int(r[0]), int(r[-1])
            in_ep[idx[a:b + 1]] = 1
            ep_id[idx[a:b + 1]] = next_ep
            phase[idx[a:b + 1]] = k
            spans.append((a, b, next_ep))
            next_ep += 1
        for a, b, e in spans:                       # in_ep is complete for this scenario now
            for p in range(max(0, a - horizon), a):
                if in_ep[idx[p]] == 0:
                    pre[idx[p]] = 1
                    pre_for[idx[p]] = e
            for p in range(b + 1, min(L, b + 1 + post)):
                if in_ep[idx[p]] == 0:
                    post_m[idx[p]] = 1

    df["in_episode"] = in_ep
    df["episode_id"] = ep_id
    df["phase"] = phase
    df["is_pre_episode_window"] = pre
    df["pre_for_episode"] = pre_for
    df["post_episode"] = post_m
    df["tail_row"] = tail
    # quiet = outside episodes, not in the contaminated post-episode zone (unless it is a genuine
    # pre-onset positive), and with a complete look-ahead label.
    df["quiet"] = ((in_ep == 0) & (tail == 0) &
                   ((post_m == 0) | (df[TARGET].to_numpy() == 1))).astype(int)
    return df


# --------------------------------------------------------------------------------------
# Alerts, thresholds, metrics
# --------------------------------------------------------------------------------------
def count_alerts(scores, quiet, y, scen, thr):
    """Merge consecutive alert windows (within a scenario, on quiet rows) into alert EVENTS.
    Returns (false_events, true_events, false_alert_windows). A true event contains >= 1
    genuine pre-onset window (y == 1)."""
    alert = (scores >= thr) & (quiet == 1)
    if not alert.any():
        return 0, 0, 0
    prev = np.r_[False, alert[:-1]]
    same = np.r_[False, scen[1:] == scen[:-1]]
    start = alert & ~(prev & same)
    run_id = np.where(alert, np.cumsum(start), 0)
    n_runs = int(start.sum())
    true_runs = len(np.unique(run_id[alert & (y == 1)]))
    false_windows = int((alert & (y == 0)).sum())
    return n_runs - true_runs, true_runs, false_windows


def choose_threshold(scores, quiet, y, scen, target_events_per_hr):
    """Lowest threshold whose false alert EVENTS/hr stays <= target on the given rows.
    Scans downward from the strictest threshold and stops at the first violation."""
    quiet = quiet.astype(int)
    hours = quiet.sum() * WINDOW_SEC / 3600.0
    neg = scores[(quiet == 1) & (y == 0)]
    cand = np.unique(np.quantile(neg, 1 - np.logspace(-4, -0.3, 150)))[::-1]
    chosen = None
    for t in cand:
        fe, _, _ = count_alerts(scores, quiet, y, scen, t)
        if fe / hours <= target_events_per_hr:
            chosen = t
        else:
            break
    if chosen is None:
        chosen = float(neg.max()) + 1e-9
    return float(chosen)


def evaluate_scenario(g, scores, thr):
    """Episode-level evaluation of one scenario at a fixed threshold."""
    quiet = g["quiet"].to_numpy()
    y = g[TARGET].to_numpy()
    fe, te, fw = count_alerts(scores, quiet, y, g["scenario"].to_numpy(), thr)
    ep_col = g["episode_id"].to_numpy()
    pre_for = g["pre_for_episode"].to_numpy()
    phase = g["phase"].to_numpy()
    scen = g["scenario"].iloc[0]
    eps = []
    for e in np.unique(ep_col[ep_col >= 0]):
        onset = int(np.where(ep_col == e)[0][0])
        hit = np.where((pre_for == e) & (quiet == 1) & (scores >= thr))[0]
        eps.append(dict(
            scenario=scen, episode_id=int(e), phase=int(phase[onset]),
            detected=int(len(hit) > 0),
            lead_sec=float((onset - hit[0]) * WINDOW_SEC) if len(hit) else np.nan,
            threshold=thr))
    stats = dict(false_events=fe, true_events=te, false_windows=fw,
                 hours=quiet.sum() * WINDOW_SEC / 3600.0)
    return eps, stats


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def fit_model(X, y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=spw, eval_metric="logloss", tree_method="hist",
        random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def inner_oof(df, scens, feats):
    """LOSO scores restricted to `scens`: each scenario scored by a model trained on the
    other scenarios of `scens` (quiet rows only). Returns an array aligned to df rows."""
    out = np.zeros(len(df))
    scen_arr = df["scenario"].to_numpy()
    for k in scens:
        rest = [s for s in scens if s != k]
        tr = df[df["scenario"].isin(rest) & (df["quiet"] == 1)]
        model = fit_model(tr[feats], tr[TARGET])
        m = scen_arr == k
        out[m] = model.predict_proba(df.loc[m, feats])[:, 1]
    return out


# --------------------------------------------------------------------------------------
# Nested LOSO
# --------------------------------------------------------------------------------------
def run_nested_loso(df, feats, target_fa):
    scenarios = sorted(df["scenario"].unique())
    scen_arr = df["scenario"].to_numpy()
    quiet = df["quiet"].to_numpy()
    y = df[TARGET].to_numpy()
    base_score = df[BASELINE_COL].to_numpy(float)
    xgb_score = np.zeros(len(df))
    thr_x, thr_b = {}, {}

    for s in scenarios:
        others = [x for x in scenarios if x != s]
        print(f"  outer fold: hold out {s} ...", flush=True)
        tr = df[df["scenario"].isin(others) & (df["quiet"] == 1)]
        model = fit_model(tr[feats], tr[TARGET])
        m_s = scen_arr == s
        xgb_score[m_s] = model.predict_proba(df.loc[m_s, feats])[:, 1]

        inner = inner_oof(df, others, feats)
        m_o = np.isin(scen_arr, others)
        thr_x[s] = choose_threshold(inner[m_o], quiet[m_o], y[m_o], scen_arr[m_o], target_fa)
        thr_b[s] = choose_threshold(base_score[m_o], quiet[m_o], y[m_o], scen_arr[m_o], target_fa)
    return scenarios, xgb_score, base_score, thr_x, thr_b


def summarize(name, eps_all, stats_all):
    n = len(eps_all)
    k = int(sum(e["detected"] for e in eps_all))
    lo, hi = wilson(k, n)
    leads = [e["lead_sec"] for e in eps_all if e["detected"]]
    hours = sum(s["hours"] for s in stats_all)
    fe = sum(s["false_events"] for s in stats_all)
    fw = sum(s["false_windows"] for s in stats_all)
    print(f"{name:14s} | warned {k:2d}/{n:2d} ({100 * k / max(1, n):4.1f}%, 95% CI "
          f"{100 * lo:.0f}-{100 * hi:.0f}%) | mean lead {np.mean(leads) if leads else 0:5.1f}s | "
          f"false alert events/hr {fe / hours:5.2f} | false alert windows/hr {fw / hours:5.2f}")
    for ph in sorted({e["phase"] for e in eps_all}):
        sub = [e for e in eps_all if e["phase"] == ph]
        kk = sum(e["detected"] for e in sub)
        print(f"{'':14s}   {PHASE_LABELS.get(ph, f'phase {ph + 1}')}: {kk}/{len(sub)}")


def report_nested(df, scenarios, xgb_score, base_score, thr_x, thr_b, target_fa):
    q = df["quiet"].to_numpy() == 1
    y = df[TARGET].to_numpy()
    scen_arr = df["scenario"].to_numpy()

    print(f"\n{'Scenario':22s} | {'ROC':>6s} | {'PR':>6s} | {'bROC':>6s} | {'bPR':>6s} | "
          f"{'thr(xgb)':>8s} | {'thr(rule)':>9s}")
    print("-" * 86)
    for s in scenarios:
        m = (scen_arr == s) & q
        print(f"{s:22s} | {roc_auc_score(y[m], xgb_score[m]):6.4f} | "
              f"{average_precision_score(y[m], xgb_score[m]):6.4f} | "
              f"{roc_auc_score(y[m], base_score[m]):6.4f} | "
              f"{average_precision_score(y[m], base_score[m]):6.4f} | "
              f"{thr_x[s]:8.4f} | {thr_b[s]:9.4f}")
    print(f"{'pooled':22s} | {roc_auc_score(y[q], xgb_score[q]):6.4f} | "
          f"{average_precision_score(y[q], xgb_score[q]):6.4f} | "
          f"{roc_auc_score(y[q], base_score[q]):6.4f} | "
          f"{average_precision_score(y[q], base_score[q]):6.4f} |")

    results = {"xgboost": ([], []), "rule baseline": ([], [])}
    rows = []
    for s in scenarios:
        g = df[df["scenario"] == s]
        m = (scen_arr == s)
        for name, sc, th in (("xgboost", xgb_score, thr_x[s]),
                             ("rule baseline", base_score, thr_b[s])):
            eps, st = evaluate_scenario(g, sc[m], th)
            results[name][0].extend(eps)
            results[name][1].append(st)
            rows.extend({**e, "method": name} for e in eps)

    print("\n" + "=" * 86)
    print(f"NESTED LOSO EPISODE SUMMARY  (threshold per fold chosen on the other scenarios, "
          f"budget <= {target_fa:.1f} false alert events/hr)")
    print("=" * 86)
    for name, (eps, st) in results.items():
        summarize(name, eps, st)

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/episode_results.csv", index=False)
    print("\nPer-episode results -> reports/episode_results.csv")


# --------------------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------------------
def run_ablation(df, feats):
    variants = {
        "full": feats,
        "no_response": [c for c in feats if not c.startswith(RESPONSE_PREFIXES)],
        "no_scale": [c for c in feats if not c.startswith(SCALE_PREFIXES)],
        "no_response_no_scale": [c for c in feats if not c.startswith(RESPONSE_PREFIXES + SCALE_PREFIXES)],
    }
    scenarios = sorted(df["scenario"].unique())
    scen_arr = df["scenario"].to_numpy()
    q = df["quiet"].to_numpy() == 1
    y = df[TARGET].to_numpy()
    print("\n" + "=" * 86)
    print("FEATURE ABLATION (plain LOSO, quiet rows)   PR-AUC per held-out scenario")
    print("=" * 86)
    header = f"{'variant':22s} | {'#feat':>5s} | " + " ".join(f"{s.split('-')[1]:>5s}" for s in scenarios) + " | pooled PR | pooled ROC"
    print(header)
    for name, fs in variants.items():
        print(f"  running {name} ({len(fs)} features) ...", flush=True)
        sc = inner_oof(df, scenarios, fs)
        prs = [average_precision_score(y[(scen_arr == s) & q], sc[(scen_arr == s) & q]) for s in scenarios]
        print(f"{name:22s} | {len(fs):5d} | " + " ".join(f"{p:5.2f}" for p in prs) +
              f" | {average_precision_score(y[q], sc[q]):9.4f} | {roc_auc_score(y[q], sc[q]):10.4f}")


# --------------------------------------------------------------------------------------
# Dashboard demo
# --------------------------------------------------------------------------------------
def build_demo(df, feats, holdouts, target_fa):
    scenarios = sorted(df["scenario"].unique())
    train_scens = [s for s in scenarios if s not in holdouts]
    scen_arr = df["scenario"].to_numpy()
    quiet = df["quiet"].to_numpy()
    y = df[TARGET].to_numpy()
    print(f"\nDemo model: train on {len(train_scens)} scenarios, hold out {holdouts}", flush=True)

    inner = inner_oof(df, train_scens, feats)
    m_tr = np.isin(scen_arr, train_scens)
    thr = choose_threshold(inner[m_tr], quiet[m_tr], y[m_tr], scen_arr[m_tr], target_fa)
    base_thr = choose_threshold(df[BASELINE_COL].to_numpy(float)[m_tr], quiet[m_tr], y[m_tr],
                                scen_arr[m_tr], target_fa)

    tr = df[df["scenario"].isin(train_scens) & (df["quiet"] == 1)]
    model = fit_model(tr[feats], tr[TARGET])

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("Top 12 features (check for lab fingerprints):")
    for f, v in imp.head(12).items():
        print(f"  {f:32s}: {v:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_pingahead.pkl")
    with open("models/model_config.json", "w") as f:
        json.dump({
            "feature_cols": feats,
            "threshold": thr,
            "rule_baseline_threshold": base_thr,
            "target_false_alert_events_per_hr": target_fa,
            "threshold_source": "inner LOSO on training scenarios only",
            "holdout_scenarios": holdouts,
            "scores_are_probabilities": False,
            "note": "Model is trained on quiet rows only; scores inside attacks are not meaningful.",
        }, f, indent=2)

    sim_dir = "data/processed/03_simulation"
    os.makedirs(sim_dir, exist_ok=True)
    print(f"\nDemo threshold = {thr:.4f}")
    for h in holdouts:
        g = df[df["scenario"] == h].copy().reset_index(drop=True)
        g["pred_prob"] = model.predict_proba(g[feats])[:, 1]
        # SIMULATION ONLY: in_episode / post_episode come from labels. A live system would not
        # know them. They let the dashboard show ground truth and hide forecasts during attacks.
        g["forecast_valid"] = ((g["in_episode"] == 0) & (g["post_episode"] == 0)).astype(int)
        g["forecast_alert"] = ((g["pred_prob"] >= thr) & (g["forecast_valid"] == 1)).astype(int)
        g["ui_state"] = np.select(
            [g["in_episode"] == 1, g["post_episode"] == 1, g["forecast_alert"] == 1],
            ["ATTACK", "RECOVERING", "WARNING"], default="NORMAL")
        eps, st = evaluate_scenario(g, g["pred_prob"].to_numpy(), thr)
        warned = sum(e["detected"] for e in eps)
        print(f"  {h}: episodes warned {warned}/{len(eps)} | false alert events {st['false_events']} "
              f"over {st['hours']:.1f} quiet h | alert windows shown: {int(g['forecast_alert'].sum())}")
        out = os.path.join(sim_dir, f"sim_{h}.csv")
        g.to_csv(out, index=False)
        print(f"    saved {out} ({len(g)} rows)")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/02_windowed/all_scenarios_windowed.csv")
    ap.add_argument("--target-fa", type=float, default=2.0, help="false alert events per hour")
    ap.add_argument("--holdout", nargs="+", default=["scenario-5-disrupt", "scenario-8-authtest"])
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Missing {args.data}. Run python3 src/windows.py first.")
    df = pd.read_csv(args.data)
    df["StartTime"] = pd.to_datetime(df["StartTime"])
    df = df.sort_values(["scenario", "StartTime"], kind="stable").reset_index(drop=True)
    df = build_masks(df)

    feats = [c for c in df.columns if c not in META_AND_LABEL_COLS]
    present = [c for c in FORBIDDEN_COLS if c in df.columns]
    if present:
        print(f"Note: excluded label-derived columns from features: {present}")
    assert not set(feats) & set(META_AND_LABEL_COLS)
    assert BASELINE_COL in feats

    q = df["quiet"] == 1
    n_pos = int(((df["quiet"] == 1) & (df[TARGET] == 1)).sum())
    n_pre = int(((df["quiet"] == 1) & (df["is_pre_episode_window"] == 1)).sum())
    print(f"{len(df):,} rows | {int(q.sum()):,} quiet rows ({q.sum() * WINDOW_SEC / 3600:.1f} h) | "
          f"{int(df['in_episode'].sum()):,} in-episode | {int(df['post_episode'].sum())} post-episode | "
          f"{int(df['tail_row'].sum())} tail | {df['episode_id'].max() + 1} episodes | "
          f"{n_pos} quiet positives ({n_pre} flagged pre-onset) | {len(feats)} features")
    if n_pos != n_pre:
        print("WARNING: quiet positives != pre-onset windows; check label/episode definitions.")

    print("\nNested leave-one-scenario-out evaluation (this fits ~64 small models) ...")
    scenarios, xs, bs, tx, tb = run_nested_loso(df, feats, args.target_fa)
    report_nested(df, scenarios, xs, bs, tx, tb, args.target_fa)

    if args.ablate:
        run_ablation(df, feats)

    build_demo(df, feats, args.holdout, args.target_fa)


if __name__ == "__main__":
    main()