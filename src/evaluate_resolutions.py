"""
evaluate_resolutions.py - compare nested-LOSO results across 5s, 10s, 30s and 60s windows.
Reads reports/<res>/episode_results.csv and reports/<res>/fa_summary.csv written by train_multires.py.
"""
import os

import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    resolutions = ["5s", "10s", "30s", "60s"]
    print("=" * 104)
    print(f"{'RESOLUTION COMPARISON (60s horizon, nested leave-one-scenario-out)':^104}")
    print("=" * 104)
    print(f"{'Res':<5} | {'Method':<13} | {'Episodes':>8} | {'Recall (95% CI)':<22} | "
          f"{'Mean lead (s)':>13} | {'False events/hr':>15} | {'False windows/hr':>16}")
    print("-" * 104)

    for res in resolutions:
        ep_path = os.path.join("reports", res, "episode_results.csv")
        fa_path = os.path.join("reports", res, "fa_summary.csv")
        if not (os.path.exists(ep_path) and os.path.exists(fa_path)):
            print(f"{res:<5} | not run yet (need {ep_path} and {fa_path})")
            continue
        eps = pd.read_csv(ep_path)
        fa = pd.read_csv(fa_path)
        for method in ["xgboost", "rule baseline"]:
            e = eps[eps["method"] == method]
            f = fa[fa["method"] == method]
            n, k = len(e), int(e["detected"].sum())
            lo, hi = wilson(k, n)
            lead = e.loc[e["detected"] == 1, "lead_sec"].mean()
            hours = f["hours"].sum()
            print(f"{res:<5} | {method:<13} | {n:>8d} | "
                  f"{k:2d}/{n:2d} {100 * k / max(1, n):5.1f}% ({100 * lo:2.0f}-{100 * hi:3.0f}%) | "
                  f"{(0 if np.isnan(lead) else lead):13.1f} | "
                  f"{f['false_events'].sum() / hours:15.2f} | {f['false_windows'].sum() / hours:16.2f}")
        print("-" * 104)

    print("Notes: lead = gap from the end of the first alerting window to the start of the first attack\n"
          "window (a lower bound). Compare rows only if every resolution has 24 episodes.")


if __name__ == "__main__":
    main()