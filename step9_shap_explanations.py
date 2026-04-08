# ============================================================
# STEP 9 — SHAP EXPLANATIONS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap, json, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
from config import TARGET_COL, APP_CURRENCY

df = pd.read_csv("data/processed_data.csv", low_memory=False)
with open("data/best_hyperparameters.json") as f:
    BEST_PARAMS = json.load(f)
with open("data/best_features.json") as f:
    FEATURES = json.load(f)

BEST_PARAMS['random_state'] = 42
X = df[FEATURES]
y = df[TARGET_COL]

print(f"Features : {FEATURES}")
print(f"Target   : {TARGET_COL}")

print("\nTraining model on full dataset...")
model = GradientBoostingRegressor(**BEST_PARAMS)
model.fit(X, y)
print("Done.")

# Built-in feature importance
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': FEATURES,
                             'Importance': importances}
                           ).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feat_imp_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(feat_imp_df['Feature'][::-1],
         feat_imp_df['Importance'][::-1], color='#1F4E79')
plt.xlabel("Importance Score")
plt.title("Built-in Feature Importance")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("plots/feature_importance_builtin.png", dpi=150)
plt.show()

# SHAP
print("\nCalculating SHAP values (500 row sample)...")
X_sample    = X.sample(500, random_state=42)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
base_value_raw = explainer.expected_value
base_value  = float(base_value_raw[0]) if hasattr(base_value_raw, '__len__') else float(base_value_raw)
print("Done.")

# SHAP Bar
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()

# SHAP Beeswarm
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Beeswarm")
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()

# Waterfall
shap_obj = explainer(X_sample)
shap.waterfall_plot(shap_obj[0], show=False)
plt.tight_layout()
plt.savefig("plots/shap_waterfall.png", dpi=150, bbox_inches='tight')
plt.show()

# Readable explanation
sample_row  = X_sample.iloc[0]
predicted   = model.predict(sample_row.values.reshape(1, -1))[0]
shap_row    = shap_values[0]

print(f"\n{'='*55}")
print("SHAP EXPLANATION — Single Shipment")
print(f"{'='*55}")
for feat, val in zip(FEATURES, sample_row.values):
    print(f"  {feat:<35}: {val:.2f}")
print(f"\nBase average price : {APP_CURRENCY}{base_value:,.2f}")
for feat, sv in sorted(zip(FEATURES, shap_row),
                        key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if sv > 0 else "↓"
    print(f"  {feat:<35}: {direction} {APP_CURRENCY}{sv:+,.2f}")
print(f"\nPredicted price    : {APP_CURRENCY}{predicted:,.2f}")

pd.DataFrame(shap_values, columns=FEATURES).to_csv(
    "data/shap_values.csv", index=False)
print("\nSaved → data/shap_values.csv")
print("Step 9 complete — Ready for Step 10!")