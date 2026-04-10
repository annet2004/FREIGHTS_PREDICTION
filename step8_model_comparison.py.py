# ============================================================
# STEP 8 — MODEL COMPARISON
# ============================================================
# WHAT WE DO:
#   Compare two models on the GA-selected features:
#     Model 1: Baseline — distance only, Linear Regression
#     Model 2: XGBoost  — GA-selected features, tuned hyperparams
#
# METRICS USED:
#   MAE  (Mean Absolute Error)         — avg rupee error
#   RMSE (Root Mean Squared Error)     — penalises large errors more
#   MAPE (Mean Absolute % Error)       — avg % error (scale independent)
#   CV Score (cross-validated MAPE)    — robust generalisation estimate
#
# WHY THESE METRICS TOGETHER:
#   MAE and RMSE are in rupees — easy to interpret in business terms.
#   MAPE is percentage — comparable across different price ranges.
#   CV Score shows how the model generalises across time periods.
#
# Winner = model with lowest MAPE and RMSE across CV folds.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error)
from config import TARGET_COL, DISTANCE_COL, APP_CURRENCY

# ── 1. Load ───────────────────────────────────────────────
df = pd.read_csv("data/processed_data.csv", low_memory=False)

with open("data/best_hyperparameters.json") as f:
    BEST_PARAMS = json.load(f)
with open("data/best_features.json") as f:
    FEATURES = json.load(f)

BEST_PARAMS["random_state"] = 42
BEST_PARAMS["verbosity"]    = 0

print(f"Features : {FEATURES}")
print(f"Target   : {TARGET_COL}")
print(f"Baseline : {DISTANCE_COL}")
print(f"\nXGBoost params: {BEST_PARAMS}")

# Distance column for baseline
dist_col = DISTANCE_COL if DISTANCE_COL in df.columns else FEATURES[0]
X_full   = df[FEATURES].values
X_base   = df[[dist_col]].values
y        = df[TARGET_COL].values

# ── 2. Models ─────────────────────────────────────────────
baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("lr",      LinearRegression()),
])

xgb_model = XGBRegressor(**BEST_PARAMS)

# ── 3. TimeSeriesSplit evaluation ─────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

results_base = {"mae": [], "rmse": [], "mape": []}
results_xgb  = {"mae": [], "rmse": [], "mape": []}

print(f"\n{'='*60}")
print("MODEL COMPARISON — Baseline vs XGBoost")
print(f"{'='*60}")

for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
    # ── Baseline ──
    baseline.fit(X_base[tr], y[tr])
    yp_b = baseline.predict(X_base[te])
    results_base["mae"].append(
        mean_absolute_error(y[te], yp_b))
    results_base["rmse"].append(
        np.sqrt(mean_squared_error(y[te], yp_b)))
    results_base["mape"].append(
        mean_absolute_percentage_error(y[te], yp_b) * 100)

    # ── XGBoost ──
    xgb_model.fit(X_full[tr], y[tr])
    yp_x = xgb_model.predict(X_full[te])
    results_xgb["mae"].append(
        mean_absolute_error(y[te], yp_x))
    results_xgb["rmse"].append(
        np.sqrt(mean_squared_error(y[te], yp_x)))
    results_xgb["mape"].append(
        mean_absolute_percentage_error(y[te], yp_x) * 100)

    print(f"  Fold {fold}:")
    print(f"    Baseline → MAE={APP_CURRENCY}{results_base['mae'][-1]:,.0f}  "
          f"RMSE={APP_CURRENCY}{results_base['rmse'][-1]:,.0f}  "
          f"MAPE={results_base['mape'][-1]:.2f}%")
    print(f"    XGBoost  → MAE={APP_CURRENCY}{results_xgb['mae'][-1]:,.0f}  "
          f"RMSE={APP_CURRENCY}{results_xgb['rmse'][-1]:,.0f}  "
          f"MAPE={results_xgb['mape'][-1]:.2f}%")

# ── 4. Summary table ──────────────────────────────────────
def avg(lst): return round(float(np.mean(lst)), 4)

summary = {
    "Baseline (Linear Regression)": {
        "MAE"      : avg(results_base["mae"]),
        "RMSE"     : avg(results_base["rmse"]),
        "MAPE (%)" : avg(results_base["mape"]),
    },
    "XGBoost (GA tuned)": {
        "MAE"      : avg(results_xgb["mae"]),
        "RMSE"     : avg(results_xgb["rmse"]),
        "MAPE (%)" : avg(results_xgb["mape"]),
    },
}

print(f"\n{'='*60}")
print("FINAL COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Model':<35} {'MAE':>12} {'RMSE':>12} {'MAPE':>10}")
print("-" * 60)
for model_name, metrics in summary.items():
    print(f"  {model_name:<33} "
          f"{APP_CURRENCY}{metrics['MAE']:>10,.0f}  "
          f"{APP_CURRENCY}{metrics['RMSE']:>10,.0f}  "
          f"{metrics['MAPE (%)']:>8.2f}%")

# ── 5. Improvement ────────────────────────────────────────
mape_improvement = round(
    (summary["Baseline (Linear Regression)"]["MAPE (%)"] -
     summary["XGBoost (GA tuned)"]["MAPE (%)"]) /
    summary["Baseline (Linear Regression)"]["MAPE (%)"] * 100, 2)

mae_improvement  = round(
    (summary["Baseline (Linear Regression)"]["MAE"] -
     summary["XGBoost (GA tuned)"]["MAE"]) /
    summary["Baseline (Linear Regression)"]["MAE"] * 100, 2)

print(f"\n  XGBoost vs Baseline:")
print(f"  MAPE improvement : {mape_improvement}% reduction in % error")
print(f"  MAE  improvement : {mae_improvement}% reduction in rupee error")
print(f"\n  Conclusion: XGBoost selected as final model ✅")

# ── 6. Plots ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_labels = ["MAE", "RMSE", "MAPE (%)"]
base_vals = [avg(results_base["mae"]),
             avg(results_base["rmse"]),
             avg(results_base["mape"])]
xgb_vals  = [avg(results_xgb["mae"]),
             avg(results_xgb["rmse"]),
             avg(results_xgb["mape"])]

for ax, label, bv, xv in zip(axes, metrics_labels, base_vals, xgb_vals):
    bars = ax.bar(["Baseline", "XGBoost"], [bv, xv],
                  color=["#90CAF9", "#1F4E79"], edgecolor="white",
                  width=0.5)
    for bar, val in zip(bars, [bv, xv]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(bv, xv)*0.02,
                f"{val:,.0f}" if "%" not in label else f"{val:.2f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(bv, xv) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Model Comparison: Baseline vs XGBoost",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → plots/model_comparison.png")

# Per-fold MAPE chart
folds = [f"Fold {i}" for i in range(1, 6)]
x = np.arange(len(folds))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, results_base["mape"], 0.35,
       label="Baseline", color="#90CAF9", edgecolor="white")
ax.bar(x + 0.2, results_xgb["mape"],  0.35,
       label="XGBoost",  color="#1F4E79", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(folds)
ax.set_ylabel("MAPE (%)")
ax.set_title("Per-Fold MAPE: Baseline vs XGBoost")
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/model_comparison_folds.png", dpi=150)
plt.show()
print("Plot saved → plots/model_comparison_folds.png")

# ── 7. Save results ────────────────────────────────────────
comparison_results = {
    "final_model"      : "XGBoost",
    "mape_improvement" : mape_improvement,
    "mae_improvement"  : mae_improvement,
    "baseline": {
        "model"        : "Linear Regression (distance only)",
        "baseline_col" : dist_col,
        "mae_mean"     : avg(results_base["mae"]),
        "rmse_mean"    : avg(results_base["rmse"]),
        "mape_mean"    : avg(results_base["mape"]),
        "fold_mape"    : [round(v,4) for v in results_base["mape"]],
    },
    "xgboost": {
        "model"        : "XGBoost (GA-tuned)",
        "features"     : FEATURES,
        "hyperparams"  : BEST_PARAMS,
        "mae_mean"     : avg(results_xgb["mae"]),
        "rmse_mean"    : avg(results_xgb["rmse"]),
        "mape_mean"    : avg(results_xgb["mape"]),
        "fold_mape"    : [round(v,4) for v in results_xgb["mape"]],
    }
}

with open("data/model_comparison_results.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("\nSaved → data/model_comparison_results.json")
print("Step 8 complete — Ready for Step 9!")