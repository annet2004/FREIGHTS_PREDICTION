# ============================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COL, APP_CURRENCY

df = pd.read_csv("data/cleaned_data.csv", low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Missing values ─────────────────────────────────────────
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Feature": missing.index,
                            "Missing_count": missing.values,
                            "Missing_%": missing_pct.values})
print("\nMissing values overview:")
print(missing_df[missing_df["Missing_%"] > 0].head(15).to_string(index=False))
if missing_df["Missing_%"].max() == 0:
    print("No missing values found.")

# Plot missing values if any exist
if missing_df["Missing_%"].max() > 0:
    top_missing = missing_df[missing_df["Missing_%"] > 0].head(12)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Missing_%", y="Feature", data=top_missing, palette="Reds_r")
    plt.title("Features by Percentage of Missing Values")
    plt.tight_layout()
    plt.savefig("plots/missing_values.png", dpi=150)
    plt.show()

# ── Target distribution ────────────────────────────────────
if TARGET_COL not in df.columns:
    print(f"ERROR: Target column '{TARGET_COL}' not found. Check config.py")
    exit()

mean_val    = df[TARGET_COL].mean()
median_val  = df[TARGET_COL].median()
quantile_95 = df[TARGET_COL].quantile(0.95)

plt.figure(figsize=(10, 6))
bins = np.arange(0, df[TARGET_COL].max() + (df[TARGET_COL].max() / 50),
                 df[TARGET_COL].max() / 50)
sns.histplot(df[TARGET_COL].dropna(), bins=bins, kde=True, color='#2E86AB')
plt.axvline(mean_val,    color='red',    linestyle='--',
            label=f"Mean {APP_CURRENCY}{mean_val:,.2f}")
plt.axvline(median_val,  color='green',  linestyle='--',
            label=f"Median {APP_CURRENCY}{median_val:,.2f}")
plt.axvline(quantile_95, color='purple', linestyle='--',
            label=f"95th pct {APP_CURRENCY}{quantile_95:,.2f}")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.title(f"Distribution of {TARGET_COL}")
plt.xlabel(TARGET_COL)
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("plots/target_distribution.png", dpi=150)
plt.show()

print(f"\n{TARGET_COL} Statistics:")
print(f"  Mean   : {APP_CURRENCY}{mean_val:,.2f}")
print(f"  Median : {APP_CURRENCY}{median_val:,.2f}")
print(f"  95th   : {APP_CURRENCY}{quantile_95:,.2f}")
print(f"  Min    : {APP_CURRENCY}{df[TARGET_COL].min():,.2f}")
print(f"  Max    : {APP_CURRENCY}{df[TARGET_COL].max():,.2f}")
print(f"  Skew   : {df[TARGET_COL].skew():.3f}")

missing_df.to_csv("data/missing_values_overview.csv", index=False)
print("\nSaved → plots/target_distribution.png")
print("Step 3 complete — Ready for Step 4!")