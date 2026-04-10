# ============================================================
# STEP 5 — CORRELATION ANALYSIS (MULTICOLLINEARITY FILTER)
# ============================================================
# WHAT WE DO:
#   Remove features that carry DUPLICATE information — i.e.
#   pairs of features that are almost perfectly correlated
#   with EACH OTHER (multicollinearity).
#
# WHY THIS APPROACH (not filtering by correlation to target):
#   Filtering features by low correlation to target (e.g. < 0.05)
#   is too aggressive — a feature might look weak individually
#   but be powerful in COMBINATION with others. The GA in Step 6
#   handles real feature selection based on actual model performance.
#
#   What we DO filter here is MULTICOLLINEARITY — when two features
#   say almost the same thing (e.g. WEIGHT_KG and NUM_PALLETS have
#   corr=0.9996). Keeping both just adds noise without adding information.
#   We keep the one with higher correlation to the target.
#
# THRESHOLD: 0.92 — pairs above this are considered duplicates.
# All remaining features pass to the GA for selection.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COL, BLACKLIST_FEATURES, MULTICOLLINEARITY_THRESHOLD

df = pd.read_csv("data/processed_data.csv", low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1. Select numeric columns only ────────────────────────
numerical = df.select_dtypes(include=[np.number])
print(f"Numeric columns: {numerical.shape[1]}")

# Remove blacklisted columns before analysis
cols_to_analyse = [c for c in numerical.columns
                   if c not in BLACKLIST_FEATURES]
num_clean = numerical[cols_to_analyse]
print(f"Columns after removing blacklist: {num_clean.shape[1]}")

# ── 2. Correlation of each feature with TARGET ─────────────
corr_with_target = num_clean.corrwith(df[TARGET_COL]).abs()
print(f"\nCorrelation of features with {TARGET_COL}:")
print(corr_with_target.sort_values(ascending=False).to_string())

# ── 3. Find multicollinear pairs ───────────────────────────
# Build upper triangle of feature-feature correlation matrix
corr_matrix = num_clean.corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > MULTICOLLINEARITY_THRESHOLD:
            high_corr_pairs.append((col, row, round(val, 4)))

high_corr_pairs.sort(key=lambda x: -x[2])
print(f"\nPairs with correlation > {MULTICOLLINEARITY_THRESHOLD} "
      f"(multicollinear): {len(high_corr_pairs)}")
for f1, f2, v in high_corr_pairs:
    keep   = f1 if corr_with_target.get(f1,0) >= corr_with_target.get(f2,0) else f2
    remove = f2 if keep == f1 else f1
    print(f"  {f1} ↔ {f2} = {v}  →  keep: {keep}  drop: {remove}")

# ── 4. Decide which to drop ───────────────────────────────
# From each multicollinear pair keep the one with higher
# correlation to target. Drop the other.
cols_to_drop = set()
for f1, f2, v in high_corr_pairs:
    if f1 in cols_to_drop or f2 in cols_to_drop:
        continue  # already handled
    corr_f1 = corr_with_target.get(f1, 0)
    corr_f2 = corr_with_target.get(f2, 0)
    drop = f2 if corr_f1 >= corr_f2 else f1
    cols_to_drop.add(drop)

print(f"\nColumns dropped (multicollinear duplicates): {len(cols_to_drop)}")
for c in sorted(cols_to_drop):
    print(f"  ✗ {c}")

# ── 5. Final feature list ──────────────────────────────────
# All remaining features — not dropped, not blacklisted
selected_features = [c for c in num_clean.columns
                     if c not in cols_to_drop]

print(f"\nFeatures passing to GA (Step 6): {len(selected_features)}")
for f in selected_features:
    corr_val = corr_with_target.get(f, 0)
    print(f"  → {f:<45} (corr with target = {corr_val:.4f})")

# ── 6. Plot — correlation with target ─────────────────────
corr_filtered = corr_with_target[selected_features].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
colors = ['#1F4E79' if v > 0.3 else '#2E86AB' if v > 0.1 else '#64B5F6'
          for v in corr_filtered.values]
plt.barh(corr_filtered.index[::-1], corr_filtered.values[::-1],
         color=colors[::-1])
plt.xlabel(f"Absolute Correlation with {TARGET_COL}")
plt.title(f"Features Passed to GA After Multicollinearity Filter\n"
          f"(removed pairs with feature-feature corr > {MULTICOLLINEARITY_THRESHOLD})")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/correlation_with_target.png", dpi=150)
plt.show()
print("Plot saved → plots/correlation_with_target.png")

# ── 7. Full heatmap ───────────────────────────────────────
plt.figure(figsize=(16, 13))
sns.heatmap(num_clean[selected_features].corr(),
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.3)
plt.title("Feature-Feature Correlation Heatmap (After Multicollinearity Filter)")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap_full.png", dpi=150)
plt.show()
print("Plot saved → plots/correlation_heatmap_full.png")

# ── 8. Save ───────────────────────────────────────────────
pd.Series(selected_features, name='feature').to_csv(
    "data/selected_features.csv", index=False)

print(f"\nSummary:")
print(f"  Started with   : {num_clean.shape[1]} features")
print(f"  Dropped        : {len(cols_to_drop)} (multicollinear duplicates)")
print(f"  Passing to GA  : {len(selected_features)} features")
print("\nSaved → data/selected_features.csv")
print("Step 5 complete — Ready for Step 6!")