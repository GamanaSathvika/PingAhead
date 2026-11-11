import pandas as pd
import numpy as np
import os
import glob

# Keep SrcAddr and DstAddr temporarily for Phase 2 unique host counting!
SHORTCUT_COLUMNS = [
    'SrcMac', 'DstMac', 'SrcOui', 'DstOui', 
    'Sport', 'FlowID', 'Rank', 'Seq', 'sIpId', 'dIpId',
    'LastTime', 'AutoId'
]

DEAD_COLUMNS = [
    'sCo', 'dCo', 'sMpls', 'dMpls', 'sAS', 'dAS', 'iAS',
    'NStrok', 'sNStrok', 'dNStrok', 'SIntDist', 'SIntActDist', 
    'SIntIdlDist', 'DIntDist', 'DIntActDist', 'DIntIdlDist', 
    'srcUdata', 'dstUdata', 'sVlan', 'dVlan', 'sVid', 'dVid', 
    'sVpri', 'dVpri', 'SRange', 'ERange', 'Inode'
]

def clean_and_prepare(input_file, output_file):
    print(f"Processing: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file, low_memory=False)
    
    # 1. Parse timestamps (handles both epoch numbers and datetime strings)
    if pd.api.types.is_numeric_dtype(df['StartTime']):
        df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s', errors='coerce')
    else:
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
        
    df = df.dropna(subset=['StartTime']).sort_values(by='StartTime').reset_index(drop=True)

    # 2. Extract Labels (0 = Normal, 1 = Recon, 2 = Attack)
    df['stage_id'] = df['CategoryLabel'].astype(str).str.lower().map({
        'benign': 0,
        'recon': 1
    }).fillna(2).astype(int)
    
    df['is_attack'] = df['BinaryLabel'].astype(int)

    # Drop raw label columns to prevent leakage
    df = df.drop(columns=['BinaryLabel', 'CategoryLabel', 'SubCategoryLabel', 'Label'], errors='ignore')

    # 3. Drop Shortcuts and Dead Columns
    cols_to_drop = [c for c in (SHORTCUT_COLUMNS + DEAD_COLUMNS) if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 4. Clean NaNs and infinite numbers
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns
    df[cat_cols] = df[cat_cols].fillna('none')

    # 5. Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"  -> Saved {os.path.basename(output_file)} | Shape: {df.shape}")

if __name__ == "__main__":
    input_dir = "data/raw/scenarios-10-sec"
    output_dir = "data/processed/01_cleaned"
    
    scenario_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    
    if not scenario_files:
        print(f"No CSVs found in {input_dir}")
    else:
        for file_path in scenario_files:
            file_name = os.path.basename(file_path)
            out_path = os.path.join(output_dir, f"clean_{file_name}")
            clean_and_prepare(file_path, out_path)
        print("\nPhase 1 Data Cleaning Complete.")