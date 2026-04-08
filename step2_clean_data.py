# ============================================================
# STEP 2 — CLEAN DATA
# ============================================================
import pandas as pd
from pathlib import Path
from config import DATASET_FILENAME, DATE_COLS, DATETIME_COL

df = pd.read_csv(Path("data") / DATASET_FILENAME)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Convert date columns (strip time, keep date only)
for col in DATE_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        print(f"  Cleaned date column: {col}")
    else:
        print(f"  Skipped (not found): {col}")

# Convert datetime column (keep full datetime for time feature extraction)
if DATETIME_COL and DATETIME_COL in df.columns:
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
    print(f"  Cleaned datetime column: {DATETIME_COL}")
elif DATETIME_COL:
    print(f"  Datetime column '{DATETIME_COL}' not found — skipping")

# Verify
print("\nSample of cleaned date columns:")
available_date_cols = [c for c in DATE_COLS if c in df.columns]
if available_date_cols:
    print(df[available_date_cols].sample(min(5, len(df))))

# Null check after conversion
print("\nNull values after cleaning:")
cols_to_check = [c for c in DATE_COLS + ([DATETIME_COL] if DATETIME_COL else [])
                 if c in df.columns]
print(df[cols_to_check].isnull().sum() if cols_to_check else "No date cols to check")

df.to_csv("data/cleaned_data.csv", index=False)
print("\nSaved → data/cleaned_data.csv")
print("Step 2 complete — Ready for Step 3!")