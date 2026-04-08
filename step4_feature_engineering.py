# ============================================================
# STEP 4 — FEATURE ENGINEERING
# ============================================================
import pandas as pd
import numpy as np
from config import (TARGET_COL, DISTANCE_COL, CATEGORICAL_COLS,
                    DATETIME_COL, DATE_COLS, COLS_TO_DROP_ALWAYS)

df = pd.read_csv("data/cleaned_data.csv", low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1. PRICE_PER_KM (if distance col exists) ──────────────
# This becomes the encoding target for mean target encoding
if DISTANCE_COL in df.columns and TARGET_COL in df.columns:
    # Avoid division by zero
    df['PRICE_PER_KM'] = df[TARGET_COL] / df[DISTANCE_COL].replace(0, np.nan)
    print(f"\nCreated PRICE_PER_KM from {TARGET_COL} / {DISTANCE_COL}")
else:
    # Fall back to using target directly for mean encoding
    df['PRICE_PER_KM'] = df[TARGET_COL]
    print(f"\nNo distance column found — using {TARGET_COL} directly for encoding")

# ── 2. Build RELATION feature ──────────────────────────────
# Combine origin + destination + vehicle if all present
origin_col = next((c for c in df.columns
                   if 'origin' in c.lower() and
                   any(x in c.lower() for x in ['country','state','region','city'])), None)
dest_col   = next((c for c in df.columns
                   if ('dest' in c.lower() or 'destination' in c.lower()) and
                   any(x in c.lower() for x in ['country','state','region','city'])), None)
vehicle_col = 'VEHICLE_TYPE' if 'VEHICLE_TYPE' in df.columns else None

if origin_col and dest_col:
    if vehicle_col:
        df['RELATION'] = (df[origin_col].astype(str) + '_' +
                          df[dest_col].astype(str) + '_' +
                          df[vehicle_col].astype(str))
    else:
        df['RELATION'] = (df[origin_col].astype(str) + '_' +
                          df[dest_col].astype(str))
    print(f"Created RELATION from {origin_col} + {dest_col}" +
          (f" + {vehicle_col}" if vehicle_col else ""))
    CATEGORICAL_COLS_WITH_RELATION = CATEGORICAL_COLS + ['RELATION']
else:
    CATEGORICAL_COLS_WITH_RELATION = CATEGORICAL_COLS

# ── 3. Mean PRICE_PER_KM per categorical column ───────────
print("\nCreating mean target encoded features:")
encoded_cols = []
for col in CATEGORICAL_COLS_WITH_RELATION:
    if col in df.columns:
        new_col = f'{col}_MEAN_PRICE_PER_KM'
        df[new_col] = df.groupby(col)['PRICE_PER_KM'].transform('mean')
        encoded_cols.append(new_col)
        print(f"  Created → {new_col}")
    else:
        print(f"  Skipped (not found): {col}")

# ── 4. Time features from DATETIME_COL ────────────────────
time_feature_cols = []
if DATETIME_COL and DATETIME_COL in df.columns:
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
    df['ENTRY_DAY']       = df[DATETIME_COL].dt.day
    df['ENTRY_WEEKDAY']   = df[DATETIME_COL].dt.weekday
    df['ENTRY_MONTH']     = df[DATETIME_COL].dt.month
    df['ENTRY_YEAR']      = df[DATETIME_COL].dt.year
    df['ENTRY_DAYOFYEAR'] = df[DATETIME_COL].dt.dayofyear
    time_features = ['ENTRY_DAY','ENTRY_WEEKDAY','ENTRY_MONTH',
                     'ENTRY_YEAR','ENTRY_DAYOFYEAR']
    print(f"\nExtracted time features from {DATETIME_COL}")
    for feat in time_features:
        new_col = f'{feat}_MEAN_PRICE_PER_KM'
        df[new_col] = df.groupby(feat)['PRICE_PER_KM'].transform('mean')
        time_feature_cols.append(new_col)
        print(f"  Created → {new_col}")
else:
    print("\nNo datetime column — skipping time features")
    time_features = []

# ── 5. Drop raw/unnecessary columns ───────────────────────
cols_to_drop = []

# Drop original categoricals (replaced by encoded versions)
for col in CATEGORICAL_COLS_WITH_RELATION:
    if col in df.columns:
        cols_to_drop.append(col)

# Drop date columns
for col in DATE_COLS:
    if col in df.columns:
        cols_to_drop.append(col)

# Drop datetime column
if DATETIME_COL and DATETIME_COL in df.columns:
    cols_to_drop.append(DATETIME_COL)

# Drop time features (replaced by mean encoded versions)
for feat in time_features:
    if feat in df.columns:
        cols_to_drop.append(feat)

# Drop always-drop columns (IDs etc)
for col in COLS_TO_DROP_ALWAYS:
    if col in df.columns:
        cols_to_drop.append(col)

# Remove duplicates from drop list
cols_to_drop = list(set(cols_to_drop))
df.drop(columns=cols_to_drop, inplace=True)
print(f"\nDropped {len(cols_to_drop)} raw/unnecessary columns")
print(f"Dataset now has {df.shape[1]} columns")

# ── 6. Safety net — fill any remaining nulls ──────────────
null_cols = [c for c in df.columns
             if df[c].isnull().sum() > 0 and df[c].dtype in ['float64','int64']]
if null_cols:
    print(f"\nFilling nulls in: {null_cols}")
    for col in null_cols:
        df[col] = df[col].fillna(df[col].mean())
else:
    print("\nNo nulls found — dataset is clean")

print(f"\nFinal shape: {df.shape}")
df.to_csv("data/processed_data.csv", index=False)
print("Saved → data/processed_data.csv")
print("Step 4 complete — Ready for Step 5!")