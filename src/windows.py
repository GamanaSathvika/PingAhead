"""
windows.py - aggregate cleaned Argus flows into fixed time windows at a chosen resolution.

Everything that depends on time is defined in SECONDS and converted to window steps, so
5s / 10s / 30s / 60s runs see the same physical horizon and (as far as the grid allows)
the same physical history:

    forecast horizon : 60 s   -> label attack_in_next_60s looks ceil(60 / window) windows ahead
    lags             : 10, 20, 30, 60 s
    diffs            : 10, 30 s
    rolling span     : 60 s (at least 2 windows so a std exists)

Lag / diff names carry the real number of seconds (e.g. flow_count_lag_30s). Rolling columns
are named *_roll_mean / *_roll_std regardless of resolution.
Steps that collapse to the same value at coarse resolutions are de-duplicated.

Usage:
    python3 src/windows.py                       # all resolutions: 5 10 30 60
    python3 src/windows.py --res 10              # one resolution
    python3 src/windows.py --res 10 --input-dir data/processed/01_cleaned
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

HORIZON_SEC = 60
LAG_SEC = (10, 20, 30, 60)
DIFF_SEC = (10, 30)
ROLL_SEC = 60
DEFAULT_RES = (5, 10, 30, 60)


def to_steps(seconds, window_sec, minimum=1):
    return sorted({max(minimum, int(round(s / window_sec))) for s in seconds})


def create_time_windows(input_file, output_file, window_sec):
    window_size = f"{window_sec}s"
    forecast_windows = max(1, math.ceil(HORIZON_SEC / window_sec))
    lag_steps = to_steps(LAG_SEC, window_sec)
    diff_steps = to_steps(DIFF_SEC, window_sec)
    roll_steps = max(2, int(round(ROLL_SEC / window_sec)))

    base_name = os.path.basename(input_file)
    scenario_name = base_name.replace("clean_", "").replace(".csv", "")
    print(f"Windowing: {base_name} (scenario={scenario_name}, window={window_size}, "
          f"horizon={forecast_windows} windows)")

    df = pd.read_csv(input_file, low_memory=False)
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    df = df.dropna(subset=['StartTime']).sort_values(by='StartTime')
    df['window_start'] = df['StartTime'].dt.floor(window_size)

    df['no_resp_flow'] = (df['DstPkts'] == 0).astype(int)

    src_col = 'SrcAddr' if 'SrcAddr' in df.columns else ('Ssaddr' if 'Ssaddr' in df.columns else None)
    dst_col = 'DstAddr' if 'DstAddr' in df.columns else ('Sdaddr' if 'Sdaddr' in df.columns else None)

    # 1-second burstiness inside each window
    df['sec_bucket'] = df['StartTime'].dt.floor('1s')
    burst_1s = (df.groupby(['window_start', 'sec_bucket']).size()
                  .groupby('window_start').max().rename('max_flows_per_1s'))

    g = df.groupby('window_start')
    agg_spec = {
        'is_attack': ('is_attack', 'max'),          # labels / metadata only
        'stage_id': ('stage_id', 'max'),
        'flow_count': ('Dur', 'size'),
        'unique_dports': ('Dport', 'nunique'),
        'Dur_mean': ('Dur', 'mean'), 'Dur_std': ('Dur', 'std'), 'Dur_max': ('Dur', 'max'),
        'TotPkts_sum': ('TotPkts', 'sum'), 'TotPkts_mean': ('TotPkts', 'mean'),
        'TotPkts_std': ('TotPkts', 'std'), 'TotPkts_max': ('TotPkts', 'max'),
        'TotBytes_sum': ('TotBytes', 'sum'), 'TotBytes_mean': ('TotBytes', 'mean'),
        'TotBytes_std': ('TotBytes', 'std'), 'TotBytes_max': ('TotBytes', 'max'),
        'SrcBytes_sum': ('SrcBytes', 'sum'), 'DstBytes_sum': ('DstBytes', 'sum'),
        'SrcPkts_sum': ('SrcPkts', 'sum'), 'DstPkts_sum': ('DstPkts', 'sum'),
        'no_resp_ratio': ('no_resp_flow', 'mean'),
        'PCRatio_mean': ('PCRatio', 'mean'),
        'Rate_mean': ('Rate', 'mean'), 'Rate_max': ('Rate', 'max'), 'Rate_std': ('Rate', 'std'),
        'SIntPkt_mean': ('SIntPkt', 'mean'), 'SIntPkt_std': ('SIntPkt', 'std'),
        'SrcJitter_mean': ('SrcJitter', 'mean'),
        'TcpRtt_mean': ('TcpRtt', 'mean'), 'TcpRtt_max': ('TcpRtt', 'max'),
        'SynAck_mean': ('SynAck', 'mean'),
        'sMeanPktSz_mean': ('sMeanPktSz', 'mean'), 'dMeanPktSz_mean': ('dMeanPktSz', 'mean'),
        'Loss_sum': ('Loss', 'sum'),
    }
    if src_col:
        agg_spec['unique_src'] = (src_col, 'nunique')
    if dst_col:
        agg_spec['unique_dst'] = (dst_col, 'nunique')

    win = g.agg(**agg_spec).join(burst_1s)

    # Categorical shares
    proto_series = df['Proto'].astype(str).str.lower()
    proto_filtered = proto_series.where(proto_series.isin(['tcp', 'udp']), 'other')
    proto_share = pd.crosstab(df['window_start'], proto_filtered, normalize='index').add_prefix('proto_')
    for p in ['proto_tcp', 'proto_udp', 'proto_other']:
        if p not in proto_share.columns:
            proto_share[p] = 0.0

    top_states = ['CON', 'REQ', 'RST', 'INT', 'FIN']
    state_series = df['State'].astype(str).str.upper()
    state_filtered = state_series.where(state_series.isin(top_states), 'OTHER')
    state_share = pd.crosstab(df['window_start'], state_filtered, normalize='index').add_prefix('state_')
    for s in [f'state_{x}' for x in top_states + ['OTHER']]:
        if s not in state_share.columns:
            state_share[s] = 0.0

    win = win.join(proto_share).join(state_share)

    # Continuous grid: empty windows stay as explicit zero-traffic rows
    full_idx = pd.date_range(start=win.index.min(), end=win.index.max(), freq=window_size, name='StartTime')
    win = win.reindex(full_idx).fillna(0.0).reset_index()
    win.insert(1, 'scenario', scenario_name)

    # Ratios
    fc = win['flow_count'].replace(0, np.nan)
    tb = win['TotBytes_sum'].replace(0, np.nan)
    win['dport_per_flow'] = (win['unique_dports'] / fc).fillna(0.0)
    if 'unique_dst' in win.columns:
        win['dst_per_flow'] = (win['unique_dst'] / fc).fillna(0.0)
        win['dport_per_dst'] = (win['unique_dports'] / win['unique_dst'].replace(0, np.nan)).fillna(0.0)
    if 'unique_src' in win.columns:
        win['src_per_flow'] = (win['unique_src'] / fc).fillna(0.0)
    win['src_bytes_ratio'] = (win['SrcBytes_sum'] / tb).fillna(0.0)

    # Temporal context (past + current windows only), spans defined in seconds
    lag_cols = ['flow_count', 'unique_dports', 'TotBytes_sum', 'TotPkts_sum',
                'no_resp_ratio', 'state_REQ', 'state_RST', 'dport_per_flow', 'max_flows_per_1s']
    if 'unique_dst' in win.columns:
        lag_cols.append('unique_dst')

    new_cols = {}
    for col in lag_cols:
        for k in lag_steps:
            new_cols[f'{col}_lag_{k * window_sec}s'] = win[col].shift(k)
        for k in diff_steps:
            new_cols[f'{col}_diff_{k * window_sec}s'] = win[col].diff(k)
        r = win[col].rolling(window=roll_steps, min_periods=1)
        new_cols[f'{col}_roll_mean'] = r.mean()
        new_cols[f'{col}_roll_std'] = r.std().fillna(0.0)
    win = pd.concat([win, pd.DataFrame(new_cols)], axis=1)

    # Target: attack anywhere in the next HORIZON_SEC (needs the full look-ahead to exist)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=forecast_windows)
    win['attack_in_next_60s'] = (
        win['is_attack'].shift(-1).rolling(window=indexer, min_periods=forecast_windows).max()
    )

    # Drop warm-up rows explicitly, then the tail rows whose look-ahead is incomplete
    warmup = max(max(lag_steps), max(diff_steps), roll_steps - 1)
    win = win.iloc[warmup:].dropna().reset_index(drop=True)

    win['is_attack'] = win['is_attack'].astype(int)
    win['stage_id'] = win['stage_id'].astype(int)
    win['attack_in_next_60s'] = win['attack_in_next_60s'].astype(int)

    step = win['StartTime'].diff().dropna().dt.total_seconds()
    if not (step == window_sec).all():
        raise ValueError(f"{scenario_name}: output is not on a {window_sec}s grid")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    win.to_csv(output_file, index=False)
    print(f"  -> Saved {os.path.basename(output_file)} | Shape: {win.shape} | "
          f"{win['StartTime'].min()} to {win['StartTime'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", type=int, default=list(DEFAULT_RES),
                    help="window sizes in seconds (default: 5 10 30 60)")
    ap.add_argument("--input-dir", default=None,
                    help="cleaned CSV folder; default data/processed/01_cleaned/<res>s "
                         "(falls back to data/processed/01_cleaned)")
    ap.add_argument("--output-root", default="data/processed/02_windowed")
    args = ap.parse_args()

    for ws in args.res:
        res = f"{ws}s"
        in_dir = args.input_dir
        if in_dir is None:
            in_dir = os.path.join("data/processed/01_cleaned", res)
            if not os.path.isdir(in_dir):
                in_dir = "data/processed/01_cleaned"
                print(f"[{res}] no per-resolution folder; using {in_dir}")
        out_dir = os.path.join(args.output_root, res)

        files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
        if not files:
            print(f"[{res}] no cleaned CSVs found in {in_dir}")
            continue

        print(f"\n=== Resolution {res}: input {in_dir} -> {out_dir} ===")
        all_dfs = []
        for f in files:
            out_path = os.path.join(out_dir, os.path.basename(f).replace("clean_", "windowed_"))
            create_time_windows(f, out_path, ws)
            all_dfs.append(pd.read_csv(out_path))

        combined_path = os.path.join(out_dir, "all_scenarios_windowed.csv")
        pd.concat(all_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"Combined dataset saved to {combined_path}")


if __name__ == "__main__":
    main()