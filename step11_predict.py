# ============================================================
# STEP 11 — PREDICT WITH SHAP EXPLANATION
# ============================================================
# Edit the SHIPMENT dict below and run:
#   python step11_predict.py
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap, joblib, json, warnings
warnings.filterwarnings("ignore")
from config import APP_CURRENCY

# Load model and metadata
model = joblib.load("models/freight_model.pkl")
with open("models/model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
TARGET   = metadata["target"]
print(f"Model loaded  | Target: {TARGET} | MAPE: {metadata['final_mape_pct']:.2f}%")
print(f"Features: {FEATURES}")

# ============================================================
# ENTER YOUR SHIPMENT DETAILS HERE
# Only include features that are in the FEATURES list above.
# The script will auto-populate any missing features with
# the dataset average so you never get errors.
# ============================================================
SHIPMENT = {
    # Edit these values:
    "TOTAL_KM"                    : 1500,
    "TONS"                        : 20,
    "ENTRY_MONTH_MEAN_PRICE_PER_KM": 2.8,
    "HAZMAT"                      : 0,
    "QTY_LOADS"                   : 1,
    # Add more features here if your model uses them
}
# ============================================================

# Load dataset averages as fallback for missing features
df = pd.read_csv("data/processed_data.csv", low_memory=False)
col_means = df[FEATURES].mean().to_dict()

# Build input — use SHIPMENT values where provided, else dataset mean
input_dict = {}
for feat in FEATURES:
    if feat in SHIPMENT:
        input_dict[feat] = SHIPMENT[feat]
    else:
        input_dict[feat] = col_means[feat]
        print(f"  Note: '{feat}' not in SHIPMENT — using dataset mean {col_means[feat]:.2f}")

input_df   = pd.DataFrame([input_dict])
prediction = model.predict(input_df)[0]

# SHAP
explainer      = shap.TreeExplainer(model)
shap_values    = explainer.shap_values(input_df)
base_value_raw = explainer.expected_value
base_value     = float(base_value_raw[0]) if hasattr(base_value_raw, '__len__') else float(base_value_raw)
shap_row       = shap_values[0]

# Print prediction
print(f"\n{'='*55}")
print("FREIGHT PRICE PREDICTION")
print(f"{'='*55}")
for feat in FEATURES:
    print(f"  {feat:<38}: {input_dict[feat]:.2f}")
print(f"\n  Predicted {TARGET:<28}: {APP_CURRENCY}{prediction:,.2f}")
print(f"{'='*55}")

# Print SHAP explanation
print(f"\n{'='*55}")
print("WHY THIS PRICE? (SHAP Explanation)")
print(f"{'='*55}")
print(f"\n  Base average price : {APP_CURRENCY}{base_value:,.2f}")
print(f"\n  Feature contributions:")
for feat, sv in sorted(zip(FEATURES, shap_row),
                        key=lambda x: abs(x[1]), reverse=True):
    d = "↑ pushed UP  " if sv > 0 else "↓ pushed DOWN"
    print(f"  {feat:<38}: {d}  {APP_CURRENCY}{sv:+,.2f}")
print(f"\n  Final price        : {APP_CURRENCY}{prediction:,.2f}")
print(f"{'='*55}")

# SHAP waterfall
shap_obj = explainer(input_df)
shap.waterfall_plot(shap_obj[0], show=False)
plt.title(f"SHAP Explanation — {APP_CURRENCY}{prediction:,.2f} predicted")
plt.tight_layout()
plt.savefig("plots/prediction_shap_waterfall.png", dpi=150, bbox_inches='tight')
plt.show()

# SHAP bar
sorted_idx = np.argsort(np.abs(shap_row))[::-1]
feats  = [FEATURES[i] for i in sorted_idx]
values = [shap_row[i] for i in sorted_idx]
colors = ['#E53935' if v > 0 else '#1E88E5' for v in values]

plt.figure(figsize=(8, 5))
plt.barh(feats[::-1], values[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel(f"SHAP Value ({APP_CURRENCY} contribution)")
plt.title(f"Price Explanation — {APP_CURRENCY}{prediction:,.2f} predicted\n"
          f"Red = UP  |  Blue = DOWN")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/prediction_shap_bar.png", dpi=150)
plt.show()
print("Plots saved → plots/prediction_shap_waterfall.png + prediction_shap_bar.png")