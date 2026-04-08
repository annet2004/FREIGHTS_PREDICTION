# ============================================================
# STEP 8 — ABLATION STUDY
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from config import TARGET_COL, DISTANCE_COL, APP_CURRENCY

df = pd.read_csv("data/processed_data.csv", low_memory=False)
with open("data/best_hyperparameters.json") as f:
    BEST_PARAMS = json.load(f)
with open("data/best_features.json") as f:
    FEATURES = json.load(f)

BEST_PARAMS['random_state'] = 42

print(f"Features : {FEATURES}")
print(f"Target   : {TARGET_COL}")
print(f"Baseline : {DISTANCE_COL}")

X_full = df[FEATURES].values
y      = df[TARGET_COL].values

# Use distance col if available, else first feature
dist_col_in_df = DISTANCE_COL if DISTANCE_COL in df.columns else FEATURES[0]
X_base = df[[dist_col_in_df]].values

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("linreg",  LinearRegression())
])

prep = ColumnTransformer(
    transformers=[("num", SimpleImputer(strategy="median"),
                   list(range(len(FEATURES))))],
    remainder="drop"
)
full_model = Pipeline([
    ("prep", prep),
    ("gbr",  GradientBoostingRegressor(**BEST_PARAMS))
])

tscv = TimeSeriesSplit(n_splits=5)
mape_base, mape_full = [], []

print(f"\n{'='*55}")
print("ABLATION STUDY — Baseline vs Full Model")
print(f"{'='*55}")

for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
    baseline.fit(X_base[tr], y[tr])
    mb = mean_absolute_percentage_error(y[te], baseline.predict(X_base[te]))
    mape_base.append(mb)

    full_model.fit(X_full[tr], y[tr])
    mf = mean_absolute_percentage_error(y[te], full_model.predict(X_full[te]))
    mape_full.append(mf)
    print(f"  Fold {fold}: Baseline={mb*100:.2f}%   Full Model={mf*100:.2f}%")

mean_base = np.mean(mape_base) * 100
mean_full = np.mean(mape_full) * 100
improvement = ((mean_base - mean_full) / mean_base) * 100

print(f"\n{'='*55}")
print(f"  Baseline MAPE  : {mean_base:.2f}%")
print(f"  Full Model MAPE: {mean_full:.2f}%")
print(f"  Improvement    : {improvement:.1f}% error reduction ✅")

# Plot
labels = [f"Baseline\n({dist_col_in_df} only)", "Full Model\n(GA + tuned GBR)"]
values = [mean_base, mean_full]
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=['#90CAF9','#1F4E79'], edgecolor='black')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f"{val:.2f}%", ha='center', fontsize=12, fontweight='bold')
plt.ylim(0, max(values) * 1.2)
plt.ylabel("MAPE (%)")
plt.title("Ablation: Baseline vs Full Model")
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/ablation_comparison.png", dpi=150)
plt.show()

results = {
    "baseline_mape_mean"  : round(mean_base, 4),
    "full_model_mape_mean": round(mean_full, 4),
    "improvement_pct"     : round(improvement, 2),
    "baseline_col"        : dist_col_in_df,
    "features_used"       : FEATURES,
    "fold_results": [
        {"fold": i+1,
         "baseline_mape": round(b*100, 4),
         "full_mape"    : round(f*100, 4)}
        for i, (b, f) in enumerate(zip(mape_base, mape_full))
    ]
}
with open("data/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved → data/ablation_results.json")
print("Step 8 complete — Ready for Step 9!")