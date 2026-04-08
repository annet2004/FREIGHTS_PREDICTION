# ============================================================
# STEP 10 — SAVE FINAL MODEL
# ============================================================
import pandas as pd
import numpy as np
import json, joblib, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from config import TARGET_COL, APP_CURRENCY

df = pd.read_csv("data/processed_data.csv", low_memory=False)
with open("data/best_hyperparameters.json") as f:
    BEST_PARAMS = json.load(f)
with open("data/best_features.json") as f:
    FEATURES = json.load(f)
with open("data/ablation_results.json") as f:
    ablation = json.load(f)

BEST_PARAMS['random_state'] = 42
X = df[FEATURES]
y = df[TARGET_COL]

# Final CV evaluation
print("Running final TimeSeriesSplit evaluation...")
tscv  = TimeSeriesSplit(n_splits=5)
mapes = []
for fold, (tr, te) in enumerate(tscv.split(X), 1):
    model = GradientBoostingRegressor(**BEST_PARAMS)
    model.fit(X.iloc[tr], y.iloc[tr])
    mape = mean_absolute_percentage_error(y.iloc[te],
           model.predict(X.iloc[te]))
    mapes.append(mape)
    print(f"  Fold {fold}: MAPE = {mape*100:.2f}%")

final_mape = np.mean(mapes) * 100
print(f"\n  Final Average MAPE: {final_mape:.2f}%")

# Train on full data
print("\nTraining final model on full dataset...")
final_model = GradientBoostingRegressor(**BEST_PARAMS)
final_model.fit(X, y)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/freight_model.pkl")

metadata = {
    "model_type"        : "GradientBoostingRegressor",
    "features"          : FEATURES,
    "target"            : TARGET_COL,
    "hyperparameters"   : BEST_PARAMS,
    "final_mape_pct"    : round(final_mape, 4),
    "baseline_mape_pct" : ablation["baseline_mape_mean"],
    "improvement_pct"   : ablation["improvement_pct"],
    "trained_on_rows"   : len(df),
    "cv_strategy"       : "TimeSeriesSplit (5 folds)",
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Verify
loaded = joblib.load("models/freight_model.pkl")
test_input = pd.DataFrame([{f: X[f].mean() for f in FEATURES}])
test_pred  = loaded.predict(test_input)[0]
print(f"\nTest prediction on mean values: {APP_CURRENCY}{test_pred:,.2f} ✅")

print(f"\n{'='*50}")
print("MODEL SAVED")
print(f"{'='*50}")
print(f"  File       : models/freight_model.pkl")
print(f"  Features   : {FEATURES}")
print(f"  Final MAPE : {final_mape:.2f}%")
print(f"  Improvement: {ablation['improvement_pct']}% over baseline")
print("\nStep 10 complete — Ready for Step 11!")