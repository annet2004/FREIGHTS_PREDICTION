# ============================================================
# STEP 5 — CORRELATION ANALYSIS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COL, CORRELATION_THRESHOLD, BLACKLIST_FEATURES

df = pd.read_csv("data/processed_data.csv", low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

numerical = df.select_dtypes(include=[np.number])
print(f"Numeric columns: {numerical.shape[1]}")

# ── Correlation with target ────────────────────────────────
if TARGET_COL not in numerical.columns:
    print(f"ERROR: '{TARGET_COL}' not found. Check config.py")
    exit()

corr_matrix   = numerical.corr()
corr_with_target = corr_matrix[TARGET_COL].abs().sort_values(ascending=False)

print(f"\nAll features and their correlation with {TARGET_COL}:")
print(corr_with_target.to_string())

# ── Plot top 15 ───────────────────────────────────────────
top15 = corr_with_target.drop(TARGET_COL).head(15)

plt.figure(figsize=(10, 7))
colors = ['#1F4E79' if v > 0.3 else '#2E86AB' if v > 0.1 else '#64B5F6'
          for v in top15.values]
plt.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
plt.axvline(0.3,  color='green',  linestyle='--', alpha=0.6, label='Strong (>0.30)')
plt.axvline(0.1,  color='orange', linestyle='--', alpha=0.6, label='Moderate (>0.10)')
plt.axvline(CORRELATION_THRESHOLD, color='red', linestyle='--', alpha=0.6,
            label=f'Threshold (>{CORRELATION_THRESHOLD})')
plt.xlabel(f"Absolute Correlation with {TARGET_COL}")
plt.title(f"Feature Correlation with {TARGET_COL}")
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/correlation_with_target.png", dpi=150)
plt.show()
print("Plot saved → plots/correlation_with_target.png")

# ── Heatmap ───────────────────────────────────────────────
plt.figure(figsize=(16, 13))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.3)
plt.title("Full Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap_full.png", dpi=150)
plt.show()

# ── Filter features ───────────────────────────────────────
selected_features = corr_with_target[
    (corr_with_target > CORRELATION_THRESHOLD) &
    (~corr_with_target.index.isin(BLACKLIST_FEATURES))
].index.tolist()

print(f"\nFeatures passing threshold {CORRELATION_THRESHOLD}: {len(selected_features)}")
print("\nSelected features going into GA:")
for f in selected_features:
    print(f"  → {f}  (corr={corr_with_target[f]:.4f})")

# ── Threshold comparison ──────────────────────────────────
for t in [0.1, 0.05, 0.03]:
    count = len(corr_with_target[
        (corr_with_target > t) &
        (~corr_with_target.index.isin(BLACKLIST_FEATURES))
    ])
    marker = " ← current" if t == CORRELATION_THRESHOLD else ""
    print(f"  Threshold {t} → {count} features{marker}")

pd.Series(selected_features, name='feature').to_csv(
    "data/selected_features.csv", index=False)
print("\nSaved → data/selected_features.csv")
print("Step 5 complete — Ready for Step 6!")