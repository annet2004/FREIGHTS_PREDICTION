# ============================================================
# STEP 1 — VIEW RAW DATA
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATASET_FILENAME, TARGET_COL

DATA_PATH = Path("data") / DATASET_FILENAME
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print(f"Dataset  : {DATASET_FILENAME}")
print(f"Rows     : {df.shape[0]:,}")
print(f"Columns  : {df.shape[1]}")
print(f"Target   : {TARGET_COL}")
print("=" * 55)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
df.info()

if TARGET_COL in df.columns:
    df[TARGET_COL].plot(figsize=(12, 4), title=f"{TARGET_COL} — Raw Series",
                        color='#1F4E79')
    plt.xlabel("Row Index")
    plt.ylabel(TARGET_COL)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/target_raw_series.png", dpi=150)
    plt.show()
    print("Plot saved → plots/target_raw_series.png")
else:
    print(f"Warning: Target column '{TARGET_COL}' not found. Check config.py")

print("\nStep 1 complete — Ready for Step 2!")