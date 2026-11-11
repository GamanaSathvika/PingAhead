"""
train.py  -  attack-onset forecasting on multi-resolution windowed Argus flows (5s, 10s, 30s, 60s).
Time-anchored to a fixed 60-second forecast horizon across all resolutions.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

# Fixed time parameters (in seconds)
TARGET_LEAD_SEC = 60
MAX_GAP_SEC = 60
POST_SEC = 60

TARGET = "attack_in_next_60s"
BASELINE_COL = "no_resp_ratio_roll_mean"   # simple rule baseline score (60 s rolling mean; see windows.py)
PHASE_LABELS = {0: "phase 1", 1: "phase 2", 2: "phase 3"}

FORBIDDEN_COLS = ["attack_flow_ratio"]
META_AND_LABEL_COLS = [
    "StartTime", "scenario", "is_attack", "stage_id", TARGET,
    "in_episode", "episode_id", "phase", "is_pre_episode_window", "pre_for_episode",
    "post_episode", "tail_row", "quiet",
] + FORBIDDEN_COLS

RESPONSE_PREFIXES = ("no_resp", "state_", "DstPkts", "DstBytes", "src_bytes_ratio",
                     "dMeanPktSz", "PCRatio", "TcpRtt", "SynAck", "Loss")
SCALE_PREFIXES = ("TotBytes", "TotPkts", "SrcBytes", "DstBytes", "SrcPkts", "DstPkts",
                  "Rate", "flow_count", "max_flows_per_1s")


# --------------------------------------------------------------------------------------
# Masks (now accepts dynamic resolution parameters)
# --------------------------------------------------------------------------------------
def build_masks(df, horizon, max_gap, post):
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
        for a, b, e in spans:
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
    df["quiet"] = ((in_ep == 0) & (tail == 0) &
                   ((post_m == 0) | (df[TARGET].to_numpy() == 1))).astype(int)
    return df


# --------------------------------------------------------------------------------------
# Alerts, thresholds, metrics
# --------------------------------------------------------------------------------------
def count_alerts(scores, quiet, y, scen, thr, window_sec):
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


def choose_threshold(scores, quiet, y, scen, target_events_per_hr, window_sec):
    quiet = quiet.astype(int)
    hours = quiet.sum() * window_sec / 3600.0
    neg = scores[(quiet == 1) & (y == 0)]
    cand = np.unique(np.quantile(neg, 1 - np.logspace(-4, -0.3, 150)))[::-1]
    chosen = None
    for t in cand:
        fe, _, _ = count_alerts(scores, quiet, y, scen, t, window_sec)
        if fe / hours <= target_events_per_hr:
            chosen = t
        else:
            break
    if chosen is None:
        chosen = float(neg.max()) + 1e-9
    return float(chosen)


def evaluate_scenario(g, scores, thr, window_sec, horizon):
    quiet = g["quiet"].to_numpy()
    y = g[TARGET].to_numpy()
    fe, te, fw = count_alerts(scores, quiet, y, g["scenario"].to_numpy(), thr, window_sec)
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
            # features of a window are only complete when it ends, so lead = gap to onset minus one
            # window. Onset is the START of the first attack window, so this is a lower bound.
            lead_sec=float((onset - hit[0]) * window_sec - window_sec) if len(hit) else np.nan,
            threshold=thr))
    stats = dict(scenario=scen, false_events=fe, true_events=te, false_windows=fw,
                 hours=quiet.sum() * window_sec / 3600.0)
    return eps, stats


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d



def sanity_check(df, window_sec, horizon, res_name):
    """Refuse to run if the file is not really at this resolution or the label horizon is off."""
    bad = []
    for scen, g in df.groupby("scenario", sort=False):
        step = g["StartTime"].diff().dropna().dt.total_seconds()
        if not (step == window_sec).all():
            bad.append(f"{scen}: window step(s) {sorted(step.unique())[:3]} != {window_sec}s")
    if bad:
        raise ValueError(f"[{res_name}] file is not on a {window_sec}s grid, so it was not windowed at this "
                         "resolution:\n  " + "\n  ".join(bad))
    eps = df[df["episode_id"] >= 0].groupby("scenario")["episode_id"].nunique()
    print(f"  episodes per scenario: {eps.to_dict()}")
    if (eps != 3).any():
        print("  WARNING: expected 3 episodes per scenario; check max_gap / merged attack runs")
    n_pos = int(((df["quiet"] == 1) & (df[TARGET] == 1)).sum())
    n_pre = int(((df["quiet"] == 1) & (df["is_pre_episode_window"] == 1)).sum())
    print(f"  rows {len(df):,} | quiet {int(df['quiet'].sum()):,} | quiet positives {n_pos} | "
          f"pre-onset windows {n_pre}")
    if n_pos != n_pre:
        raise ValueError(f"[{res_name}] label look-ahead does not match horizon={horizon} windows; "
                         "rebuild the windowed file with windows.py at this resolution")

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
def run_nested_loso(df, feats, target_fa, window_sec):
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
        thr_x[s] = choose_threshold(inner[m_o], quiet[m_o], y[m_o], scen_arr[m_o], target_fa, window_sec)
        thr_b[s] = choose_threshold(base_score[m_o], quiet[m_o], y[m_o], scen_arr[m_o], target_fa, window_sec)
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


def report_nested(df, scenarios, xgb_score, base_score, thr_x, thr_b, target_fa, res_name, window_sec, horizon):
    q = df["quiet"].to_numpy() == 1
    y = df[TARGET].to_numpy()
    scen_arr = df["scenario"].to_numpy()

    results = {"xgboost": ([], []), "rule baseline": ([], [])}
    rows = []
    stat_rows = []
    for s in scenarios:
        g = df[df["scenario"] == s]
        m = (scen_arr == s)
        for name, sc, th in (("xgboost", xgb_score, thr_x[s]),
                             ("rule baseline", base_score, thr_b[s])):
            eps, st = evaluate_scenario(g, sc[m], th, window_sec, horizon)
            results[name][0].extend(eps)
            results[name][1].append(st)
            rows.extend({**e, "method": name} for e in eps)
            stat_rows.append({**st, "method": name, "threshold": th})

    print(f"\nNESTED LOSO EPISODE SUMMARY [{res_name}]")
    for name, (eps, st) in results.items():
        summarize(name, eps, st)

    out_dir = os.path.join("reports", res_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "episode_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    fa_path = os.path.join(out_dir, "fa_summary.csv")
    pd.DataFrame(stat_rows).to_csv(fa_path, index=False)
    print(f"Per-episode results -> {out_path}\nFalse-alert summary -> {fa_path}")


# --------------------------------------------------------------------------------------
# Dashboard demo
# --------------------------------------------------------------------------------------
def build_demo(df, feats, holdouts, target_fa, res_name, window_sec, horizon):
    scenarios = sorted(df["scenario"].unique())
    train_scens = [s for s in scenarios if s not in holdouts]
    scen_arr = df["scenario"].to_numpy()
    quiet = df["quiet"].to_numpy()
    y = df[TARGET].to_numpy()
    print(f"\n[{res_name}] Demo model: train on {len(train_scens)} scenarios, hold out {holdouts}")

    inner = inner_oof(df, train_scens, feats)
    m_tr = np.isin(scen_arr, train_scens)
    thr = choose_threshold(inner[m_tr], quiet[m_tr], y[m_tr], scen_arr[m_tr], target_fa, window_sec)
    base_thr = choose_threshold(df[BASELINE_COL].to_numpy(float)[m_tr], quiet[m_tr], y[m_tr],
                                scen_arr[m_tr], target_fa, window_sec)

    tr = df[df["scenario"].isin(train_scens) & (df["quiet"] == 1)]
    model = fit_model(tr[feats], tr[TARGET])

    model_dir = os.path.join("models", res_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "xgb_pingahead.pkl"))
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump({
            "feature_cols": feats,
            "threshold": thr,
            "rule_baseline_threshold": base_thr,
            "target_false_alert_events_per_hr": target_fa,
            "resolution": res_name,
            "window_sec": window_sec,
            "horizon_steps": horizon,
            "threshold_source": "inner LOSO on training scenarios only",
            "holdout_scenarios": holdouts,
        }, f, indent=2)

    sim_dir = os.path.join("data", "processed", "03_simulation", res_name)
    os.makedirs(sim_dir, exist_ok=True)
    for h in holdouts:
        g = df[df["scenario"] == h].copy().reset_index(drop=True)
        g["pred_prob"] = model.predict_proba(g[feats])[:, 1]
        g["forecast_valid"] = ((g["in_episode"] == 0) & (g["post_episode"] == 0)).astype(int)
        g["forecast_alert"] = ((g["pred_prob"] >= thr) & (g["forecast_valid"] == 1)).astype(int)
        g["ui_state"] = np.select(
            [g["in_episode"] == 1, g["post_episode"] == 1, g["forecast_alert"] == 1],
            ["ATTACK", "RECOVERING", "WARNING"], default="NORMAL")
        out = os.path.join(sim_dir, f"sim_{h}.csv")
        g.to_csv(out, index=False)
        print(f"  saved simulation -> {out}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fa", type=float, default=2.0)
    ap.add_argument("--holdout", nargs="+", default=["scenario-5-disrupt", "scenario-8-authtest"])
    ap.add_argument("--res", nargs="+", default=None, help="subset of resolutions, e.g. --res 10s")
    args = ap.parse_args()

    resolutions = [
        ("5s", 5, "data/processed/02_windowed/5s/all_scenarios_windowed.csv"),
        ("10s", 10, "data/processed/02_windowed/10s/all_scenarios_windowed.csv"),
        ("30s", 30, "data/processed/02_windowed/30s/all_scenarios_windowed.csv"),
        ("60s", 60, "data/processed/02_windowed/60s/all_scenarios_windowed.csv"),
    ]

    for res_name, window_sec, data_path in resolutions:
        if args.res and res_name not in args.res:
            continue
        if not os.path.exists(data_path):
            print(f"Skipping {res_name}: {data_path} not found.")
            continue

        # Dynamically scale horizon, max gap, and post windows based on physical time (60s target)
        horizon = max(1, TARGET_LEAD_SEC // window_sec)
        max_gap = max(1, MAX_GAP_SEC // window_sec)
        post_windows = max(1, POST_SEC // window_sec)

        print("\n" + "#" * 86)
        print(f"PROCESSING RESOLUTION: {res_name} ({window_sec}s windows) | Horizon: {horizon} steps ({horizon * window_sec}s lead time)")
        print("#" * 86)

        df = pd.read_csv(data_path)
        df["StartTime"] = pd.to_datetime(df["StartTime"])
        df = df.sort_values(["scenario", "StartTime"], kind="stable").reset_index(drop=True)
        df = build_masks(df, horizon=horizon, max_gap=max_gap, post=post_windows)

        sanity_check(df, window_sec, horizon, res_name)

        feats = [c for c in df.columns if c not in META_AND_LABEL_COLS]
        assert BASELINE_COL in feats

        scenarios, xs, bs, tx, tb = run_nested_loso(df, feats, args.target_fa, window_sec)
        report_nested(df, scenarios, xs, bs, tx, tb, args.target_fa, res_name, window_sec, horizon)
        build_demo(df, feats, args.holdout, args.target_fa, res_name, window_sec, horizon)


if __name__ == "__main__":
    main()