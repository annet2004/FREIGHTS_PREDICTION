# 🚚 Freight Price Prediction System

An end-to-end machine learning pipeline that predicts road freight shipment
prices and explains every prediction using SHAP values. The system uses
Genetic Algorithms for both feature selection and hyperparameter tuning,
with XGBoost as the final prediction model.

> **One-line summary:**
> *"We built an end-to-end ML pipeline using Genetic Algorithms for both
> feature selection and hyperparameter tuning, combined with XGBoost for
> accurate and interpretable freight price prediction."*

---

## 📌 Key Features

- **Dataset-agnostic** — switch datasets by editing only `config.py`
- **Genetic Algorithm** for optimal feature selection (Ridge scorer — fast and neutral)
- **Genetic Algorithm** for XGBoost hyperparameter tuning
- **Multicollinearity removal** — keeps informationally unique features only
- **Mean Target Encoding** — converts categorical columns to meaningful numerics
- **Model Comparison** — XGBoost vs distance-only baseline (MAE, RMSE, MAPE)
- **SHAP Explainability** — global and per-shipment price breakdown
- **Streamlit Web App** — 6-page interactive frontend

---

## 🗂️ Project Structure

```
freight_prediction/
│
├── config.py                        ← only file you edit per dataset
│
├── data/
│   ├── freight_price_dataset.csv    ← raw input (5,000 records, 54 columns)
│   ├── cleaned_data.csv             ← after Step 2
│   ├── processed_data.csv           ← after Step 4 (feature engineered)
│   ├── selected_features.csv        ← after Step 5 (multicollinearity filtered)
│   ├── selected_features_per_fold.json  ← after Step 6 (GA per fold)
│   ├── feature_selection_frequency.csv  ← after Step 6
│   ├── best_hyperparameters.json    ← after Step 7
│   ├── best_features.json           ← after Step 7
│   └── model_comparison_results.json   ← after Step 8
│
├── models/
│   ├── freight_model.pkl            ← saved XGBoost model
│   └── model_metadata.json         ← features, params, MAPE, row count
│
├── plots/                           ← all charts auto-saved here
│
├── step1_view_raw_data.py           ← inspect dataset structure
├── step2_clean_data.py              ← clean date columns
├── step3_eda.py                     ← missing values + target distribution
├── step4_feature_engineering.py     ← RELATION feature + mean target encoding
├── step5_correlation_analysis.py    ← multicollinearity removal (threshold 0.92)
├── step6_feature_selection_ga.py    ← GA feature selection (Ridge scorer)
├── step7_hyperparameter_tuning.py   ← GA XGBoost hyperparameter tuning
├── step8_model_comparison.py        ← baseline vs XGBoost (MAE, RMSE, MAPE)
├── step9_shap_explanations.py       ← SHAP global + local explanations
├── step10_save_model.py             ← save final XGBoost model
├── step11_predict.py                ← predict new shipment + SHAP output
├── app.py                           ← Streamlit web app (6 pages)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Ridge scorer, Linear Regression baseline, metrics, CV |
| XGBoost | Final prediction model |
| DEAP | Genetic Algorithm (feature selection + hyperparameter tuning) |
| SHAP | Model explainability |
| matplotlib / seaborn | Visualisation |
| joblib | Model serialisation |
| Streamlit | Web frontend |

---

## 📊 Dataset

- **File:** `freight_price_dataset.csv`
- **Records:** 5,000 shipments
- **Columns:** 54
- **Target:** `FREIGHT_PRICE_INR` (₹5,000 – ₹7,44,829)
- **No missing values**
- **Coverage:** Indian road freight — cities, states, regions, vehicle types,
  cargo categories, seasonal flags, carrier details

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd freight_prediction

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset in data/ folder

# 5. Edit config.py to match your column names (if using a different dataset)
```

---

## ▶️ Running the Pipeline

Run steps in order — each step saves output the next step reads.

```bash
python step1_view_raw_data.py          # inspect raw dataset
python step2_clean_data.py             # clean date columns
python step3_eda.py                    # missing values + target distribution
python step4_feature_engineering.py   # feature creation → processed_data.csv
python step5_correlation_analysis.py  # multicollinearity filter → selected_features.csv
python step6_feature_selection_ga.py  # GA feature selection (~15 mins)
python step7_hyperparameter_tuning.py # GA XGBoost tuning (~8 mins)
python step8_model_comparison.py      # compare baseline vs XGBoost
python step9_shap_explanations.py     # generate SHAP plots
python step10_save_model.py           # save final model
python step11_predict.py              # predict a new shipment
```

### Launch the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🔬 Pipeline — How It Works

### 1. Configuration Layer — `config.py`
Central configuration file defining target column, blacklisted features,
categorical columns, GA parameters, and XGBoost hyperparameter bounds.
Changing dataset requires editing only this file.

### 2. Data Understanding — `step1_view_raw_data.py`
Loads and inspects the raw CSV — shape, column names, data types, first
rows, and a line chart of the target variable. Ensures no surprises before
any processing begins.

### 3. Data Cleaning — `step2_clean_data.py`
Converts date columns stored as strings (`"2022-08-17 00:00"`) into proper
Python date objects using `pd.to_datetime()`. This dataset had time features
(YEAR, MONTH, WEEKDAY, IS_FESTIVAL_MONTH) pre-built, so no datetime
extraction was needed — only date string cleaning.

### 4. Exploratory Data Analysis — `step3_eda.py`
Analyses missing values across all columns and plots the distribution of
`FREIGHT_PRICE_INR` with mean, median, and 95th percentile marked. The
dataset showed no missing values and a right-skewed price distribution
(skew = 1.61) — most shipments are cheap, a small tail is expensive.

### 5. Feature Engineering — `step4_feature_engineering.py`
Three main transformations:

- **PRICE_PER_KM** — target normalised by distance, used as encoding target
- **RELATION feature** — combines ORIGIN_CITY + DEST_CITY + VEHICLE_TYPE
  into a single label (e.g. `Mumbai_Delhi_Truck 32T`) capturing route+vehicle
  pricing signal
- **Mean Target Encoding** — replaces each categorical column (origin city,
  state, region, vehicle type, service type, load type, carrier) with the
  average PRICE_PER_KM for that group. This gives the model a meaningful
  numeric signal instead of arbitrary label codes

Raw text and date columns are dropped after encoding. A safety imputation
pass fills any remaining nulls with column means.

### 6. Multicollinearity Removal — `step5_correlation_analysis.py`
Computes feature-feature correlation and identifies pairs with correlation
above **0.92** (near-duplicate information). From each such pair, the feature
with lower correlation to the target is dropped. This step does NOT filter
by correlation to target — that would remove features that look weak
individually but may be powerful in combination. The GA handles that.

**Result:** 43 features → 8 duplicates removed → **35 features to GA**

Key pairs removed:
- `NUM_PALLETS ↔ WEIGHT_KG` (corr 0.9996) — near-identical cargo size signals
- `STRAIGHT_DISTANCE_KM ↔ ROAD_DISTANCE_KM` (corr 0.99) — same distance, different measurement
- `FUEL_SURCHARGE_PCT ↔ DIESEL_PRICE_PER_LITRE` (corr 1.0) — perfect duplicate

### 7. Feature Selection — `step6_feature_selection_ga.py`
A **Genetic Algorithm** (DEAP library) finds the best combination of features
from the 35 filtered candidates.

**Why Ridge as the scorer (not XGBoost)?**
The GA trains a model hundreds of times internally. XGBoost would be
computationally expensive here. Ridge Regression is extremely fast,
unbiased toward any tree model, and reliably identifies which features
carry genuine signal vs noise. Features selected this way are useful for
any downstream model.

**How the GA works:**
- Each individual = binary list (1 = use feature, 0 = ignore)
- Fitness = MAPE scored by Ridge on inner TimeSeriesSplit folds
- Operators: tournament selection, uniform crossover (cxpb=0.7), bit-flip mutation (mutpb=0.2)
- Population: 20 | Generations: 10 | Outer folds: 5 | Inner folds: 3

Features selected in ALL 5 folds are the most reliable and go to Step 7.

### 8. Hyperparameter Tuning — `step7_hyperparameter_tuning.py`
A **second Genetic Algorithm** optimises XGBoost hyperparameters.

**Parameters tuned:**
| Parameter | Range | What it controls |
|---|---|---|
| n_estimators | 50–400 | Number of boosting rounds |
| learning_rate | 0.01–0.30 | Step size per tree |
| max_depth | 3–10 | Maximum tree depth |
| subsample | 0.5–1.0 | Fraction of rows per tree |
| colsample_bytree | 0.5–1.0 | Fraction of features per tree |
| min_child_weight | 1–10 | Minimum leaf instance weight |

Each individual is a set of numeric hyperparameter values (not binary).
Custom mutation nudges values by small random amounts within bounds.
Fitness = cross-validated MAPE using TimeSeriesSplit (3 folds).

### 9. Model Comparison — `step8_model_comparison.py`
Compares two models on the GA-selected features using 5-fold TimeSeriesSplit:

| Model | Description |
|---|---|
| Baseline | Linear Regression on distance only (`ROAD_DISTANCE_KM`) |
| XGBoost | GA-selected features + GA-tuned hyperparameters |

**Metrics reported:** MAE, RMSE, MAPE per fold and averaged.

XGBoost is selected as the final model based on metric comparison.

### 10. SHAP Explanations — `step9_shap_explanations.py`
Uses `shap.TreeExplainer` to explain the XGBoost model at two levels:

- **Global** — which features matter most across all shipments (bar chart + beeswarm)
- **Local** — why a specific shipment got its price (waterfall chart)

Every rupee in a prediction is attributed to a feature. Example:
```
Base average price : ₹1,19,276
ROAD_DISTANCE_KM   : ↑ +₹32,450  (long route)
WEIGHT_KG          : ↑ +₹18,200  (heavy cargo)
IS_FESTIVAL_MONTH  : ↑ +₹8,100   (peak season)
SAME_REGION        : ↓ -₹4,300   (same region = shorter haul discount)
Predicted price    : ₹1,73,726
```

### 11. Save Model — `step10_save_model.py`
Trains final XGBoost on the complete dataset and saves to
`models/freight_model.pkl` (joblib). Saves metadata including feature
list, hyperparameters, and final MAPE to `models/model_metadata.json`.

### 12. Predict — `step11_predict.py`
Loads the saved model, accepts shipment details, predicts price, and
generates SHAP explanation. Missing feature values auto-fill with
dataset averages.

---

## 🌐 Web App — 6 Pages

| Page | Content |
|---|---|
| 📊 EDA | Dataset metrics, price distribution, missing values, top feature correlations |
| 🔬 Feature Selection | Correlation bar chart, GA frequency chart, final selected features |
| 📊 Model Comparison | Metrics table (MAE, RMSE, MAPE), side-by-side bar chart, per-fold MAPE |
| 🤖 Model Results | Best hyperparameters table, MAPE improvement summary |
| 🔍 SHAP Analysis | Global importance, beeswarm plot, per-shipment waterfall explanation |
| 💡 Predict Price | Smart input form with Yes/No dropdowns, category dropdowns, SHAP output |

---

## 🔄 Switching to a New Dataset

1. Place CSV in `data/`
2. Edit `config.py`:
   - `DATASET_FILENAME`, `TARGET_COL`, `DISTANCE_COL`
   - `DATE_COLS`, `DATETIME_COL`
   - `CATEGORICAL_COLS`, `COLS_TO_DROP_ALWAYS`
   - `BLACKLIST_FEATURES` (target + any cost sub-components)
   - `APP_CURRENCY`
3. Run from `step1` onwards
4. Nothing else changes

---


## 🐛 Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Target column 'EUR' not found` | Old config still active | Edit config.py — update TARGET_COL |
| `KeyError: 'baseline_mape_mean'` | Old ablation_results.json loaded | Rerun step8_model_comparison.py |
| `No module named 'xgboost'` | Not installed | `pip install xgboost` |
| GA running over 1 hour | Population/generations too high | Reduce GA_POPULATION_SIZE=10, GA_NUM_GENERATIONS=5 |
| SHAP TypeError on base_value | SHAP version difference | Already handled in scripts |
| Streamlit shows stale data | Cached results | Clear cache: three-dot menu → Clear cache |

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
deap
shap
joblib
streamlit
openpyxl
```

Install: `pip install -r requirements.txt`

---

## 📚 Reference

Budzynski, A. et al. (2025).
*Enhancing Road Freight Price Forecasting Using Gradient Boosting Ensemble
Supervised Machine Learning Algorithm.*
Mathematics, 13(18), 2964.
https://doi.org/10.3390/math13182964