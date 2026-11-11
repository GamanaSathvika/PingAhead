import pandas as pd
import numpy as np
import os
import glob

def create_time_windows(input_file, output_file, window_size='10s', forecast_windows=6):
    base_name = os.path.basename(input_file)
    scenario_name = base_name.replace("clean_", "").replace(".csv", "")
    print(f"Windowing: {base_name} (scenario={scenario_name})")
    
    df = pd.read_csv(input_file, low_memory=False)
    
    # 1. Convert timestamp strings to datetime objects and floor to window bucket
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    df = df.dropna(subset=['StartTime']).sort_values(by='StartTime')
    df['window_start'] = df['StartTime'].dt.floor(window_size)
    
    # Helper flags at flow level before grouping
    df['no_resp_flow'] = (df['DstPkts'] == 0).astype(int)
    
    src_col = 'SrcAddr' if 'SrcAddr' in df.columns else ('Ssaddr' if 'Ssaddr' in df.columns else None)
    dst_col = 'DstAddr' if 'DstAddr' in df.columns else ('Sdaddr' if 'Sdaddr' in df.columns else None)

    # 2. Sub-bucket 1-second burstiness
    df['sec_bucket'] = df['StartTime'].dt.floor('1s')
    burst_1s = df.groupby(['window_start', 'sec_bucket']).size().groupby('window_start').max().rename('max_flows_per_1s')

    # 3. Core Numeric & Cardinality Aggregations
    g = df.groupby('window_start')
    
    agg_spec = {
        # Labels / Metadata ONLY (Excluded from training features)
        'is_attack': ('is_attack', 'max'),
        'stage_id': ('stage_id', 'max'),
        
        # Volume & Diversity
        'flow_count': ('Dur', 'size'),
        'unique_dports': ('Dport', 'nunique'),
        
        # Duration dispersion
        'Dur_mean': ('Dur', 'mean'),
        'Dur_std': ('Dur', 'std'),
        'Dur_max': ('Dur', 'max'),
        
        # Packet & Byte totals + burstiness
        'TotPkts_sum': ('TotPkts', 'sum'),
        'TotPkts_mean': ('TotPkts', 'mean'),
        'TotPkts_std': ('TotPkts', 'std'),
        'TotPkts_max': ('TotPkts', 'max'),
        'TotBytes_sum': ('TotBytes', 'sum'),
        'TotBytes_mean': ('TotBytes', 'mean'),
        'TotBytes_std': ('TotBytes', 'std'),
        'TotBytes_max': ('TotBytes', 'max'),
        
        # Directional asymmetry
        'SrcBytes_sum': ('SrcBytes', 'sum'),
        'DstBytes_sum': ('DstBytes', 'sum'),
        'SrcPkts_sum': ('SrcPkts', 'sum'),
        'DstPkts_sum': ('DstPkts', 'sum'),
        'no_resp_ratio': ('no_resp_flow', 'mean'),
        'PCRatio_mean': ('PCRatio', 'mean'),
        
        # Rates & Inter-packet timing
        'Rate_mean': ('Rate', 'mean'),
        'Rate_max': ('Rate', 'max'),
        'Rate_std': ('Rate', 'std'),
        'SIntPkt_mean': ('SIntPkt', 'mean'),
        'SIntPkt_std': ('SIntPkt', 'std'),
        'SrcJitter_mean': ('SrcJitter', 'mean'),
        
        # TCP Handshake & Packet Size
        'TcpRtt_mean': ('TcpRtt', 'mean'),
        'TcpRtt_max': ('TcpRtt', 'max'),
        'SynAck_mean': ('SynAck', 'mean'),
        'sMeanPktSz_mean': ('sMeanPktSz', 'mean'),
        'dMeanPktSz_mean': ('dMeanPktSz', 'mean'),
        'Loss_sum': ('Loss', 'sum'),
    }
    
    if src_col:
        agg_spec['unique_src'] = (src_col, 'nunique')
    if dst_col:
        agg_spec['unique_dst'] = (dst_col, 'nunique')

    win = g.agg(**agg_spec).join(burst_1s)

    # 4. Categorical Shares
    top_protos = ['tcp', 'udp']
    proto_series = df['Proto'].astype(str).str.lower()
    proto_filtered = proto_series.where(proto_series.isin(top_protos), 'other')
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

    # 5. Reindex to continuous 10s grid so empty windows stay as explicit 0-traffic rows
    full_idx = pd.date_range(start=win.index.min(), end=win.index.max(), freq=window_size, name='StartTime')
    win = win.reindex(full_idx).fillna(0.0).reset_index()
    
    # Insert scenario identifier column right after StartTime
    win.insert(1, 'scenario', scenario_name)

    # 6. Cardinality & Asymmetry Ratios
    fc = win['flow_count'].replace(0, np.nan)
    tb = win['TotBytes_sum'].replace(0, np.nan)
    
    win['dport_per_flow'] = (win['unique_dports'] / fc).fillna(0.0)
    if 'unique_dst' in win.columns:
        win['dst_per_flow'] = (win['unique_dst'] / fc).fillna(0.0)
        win['dport_per_dst'] = (win['unique_dports'] / win['unique_dst'].replace(0, np.nan)).fillna(0.0)
    if 'unique_src' in win.columns:
        win['src_per_flow'] = (win['unique_src'] / fc).fillna(0.0)
        
    win['src_bytes_ratio'] = (win['SrcBytes_sum'] / tb).fillna(0.0)

    # 7. Temporal Context (Past + Current windows only)
    lag_cols = [
        'flow_count', 'unique_dports', 'TotBytes_sum', 'TotPkts_sum',
        'no_resp_ratio', 'state_REQ', 'state_RST', 'dport_per_flow', 'max_flows_per_1s'
    ]
    if 'unique_dst' in win.columns:
        lag_cols.append('unique_dst')

    for col in lag_cols:
        for l in [1, 2, 3, 6]:
            win[f'{col}_lag_{l}'] = win[col].shift(l)
        win[f'{col}_diff_1'] = win[col].diff(1)
        win[f'{col}_diff_3'] = win[col].diff(3)
        win[f'{col}_roll6_mean'] = win[col].rolling(window=6, min_periods=1).mean()
        win[f'{col}_roll6_std'] = win[col].rolling(window=6, min_periods=1).std().fillna(0.0)

    # 8. The PingAhead Target: Require ALL 6 future windows to exist (min_periods=forecast_windows)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=forecast_windows)
    win['attack_in_next_60s'] = (
        win['is_attack']
        .shift(-1)
        .rolling(window=indexer, min_periods=forecast_windows)
        .max()
    )
    
    # 9. Drop warm-up rows (first 6 where lag_6 is NaN) AND tail rows (last 6 where lookahead is NaN)
    win = win.dropna().reset_index(drop=True)
        
    win['is_attack'] = win['is_attack'].astype(int)
    win['stage_id'] = win['stage_id'].astype(int)
    win['attack_in_next_60s'] = win['attack_in_next_60s'].astype(int)

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    win.to_csv(output_file, index=False)
    print(f"  -> Saved {os.path.basename(output_file)} | Shape: {win.shape} | {win['StartTime'].min()} to {win['StartTime'].max()}")

if __name__ == "__main__":
    input_dir = "data/processed/01_cleaned"
    output_dir = "data/processed/02_windowed"
    
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    
    if not files:
        print(f"No cleaned CSVs found in {input_dir}")
    else:
        all_dfs = []
        for f in files:
            out_path = os.path.join(output_dir, os.path.basename(f).replace("clean_", "windowed_"))
            create_time_windows(f, out_path)
            all_dfs.append(pd.read_csv(out_path))
            
        # Also save a combined file for easy cross-validation & inspection
        combined_path = os.path.join(output_dir, "all_scenarios_windowed.csv")
        pd.concat(all_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\nPhase 2 Complete. Combined dataset saved to {combined_path}")