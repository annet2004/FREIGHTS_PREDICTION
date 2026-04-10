# ============================================================
# config.py — UNIVERSAL DATASET CONFIGURATION
# ============================================================
# This is the ONLY file you need to edit when switching datasets.
# All pipeline steps and app.py read their settings from here.
#
# CURRENT DATASET: Indian Freight Price Dataset
# Rows: 5,000  |  Columns: 54  |  Target: FREIGHT_PRICE_INR
# ============================================================

# ── Dataset file ───────────────────────────────────────────
DATASET_FILENAME = "freight_price_dataset.csv"

# ── Target variable ────────────────────────────────────────
TARGET_COL = "FREIGHT_PRICE_INR"

# ── Distance column (used as baseline model feature) ───────
DISTANCE_COL = "ROAD_DISTANCE_KM"

# ── Date columns ───────────────────────────────────────────
# These store date+time as strings ("2022-08-17 00:00")
# We strip the time and keep only the date part
DATE_COLS = [
    "START_LOAD_DATE",
    "END_LOAD_DATE",
    "START_DELIVERY_DATE",
    "END_DELIVERY_DATE",
]

# ── Datetime column ────────────────────────────────────────
# Set to None because this dataset has no TIME_OF_ENTRY column.
# Time features (YEAR, MONTH, WEEKDAY etc.) are already
# pre-built in the dataset — no extraction needed.
DATETIME_COL = None

# ── Categorical columns ────────────────────────────────────
# Text columns that will be mean target encoded.
# NOTE: ORIGIN_TIER and DEST_TIER are integers (1 or 2)
#       so they are NOT included here — they go in as numeric.
# NOTE: CUSTOMER_ID is dropped (it is an ID, not a feature).
CATEGORICAL_COLS = [
    "ORIGIN_CITY",
    "ORIGIN_STATE",
    "ORIGIN_REGION",
    "DEST_CITY",
    "DEST_STATE",
    "DEST_REGION",
    "VEHICLE_TYPE",
    "SERVICE_TYPE",
    "LOAD_TYPE",
    "CARRIER_ID",
]

# ── Columns to always drop ─────────────────────────────────
# IDs and identifiers that carry no predictive signal
COLS_TO_DROP_ALWAYS = [
    "ORDER_ID",      # unique shipment ID — not a feature
    "CUSTOMER_ID",   # customer ID — not a feature
]

# ── Correlation threshold ──────────────────────────────────
# Features below this absolute correlation with TARGET_COL
# are removed before the Genetic Algorithm.
CORRELATION_THRESHOLD = 0.05

# ── Blacklist features ─────────────────────────────────────
# These columns must NEVER be used as model features.
# FUEL_COST_INR, TOLL_COST_INR, DRIVER_ALLOWANCE_INR,
# INSURANCE_COST_INR, FUEL_SURCHARGE_PCT are direct cost
# sub-components that add up to FREIGHT_PRICE_INR.
# Using them would be DATA LEAKAGE — the model would
# essentially know the answer before predicting.
BLACKLIST_FEATURES = [
    "FREIGHT_PRICE_INR",      # target itself
    "PRICE_PER_KM",           # derived from target during feature engineering
    "FUEL_COST_INR",          # sub-component of target → leakage
    "TOLL_COST_INR",          # sub-component of target → leakage
    "DRIVER_ALLOWANCE_INR",   # sub-component of target → leakage
    "INSURANCE_COST_INR",     # sub-component of target → leakage
    "FUEL_SURCHARGE_PCT",     # directly determines fuel cost → leakage
]

# ── GA parameters ──────────────────────────────────────────
GA_POPULATION_SIZE = 20
GA_NUM_GENERATIONS = 10
GA_OUTER_FOLDS     = 5
GA_INNER_FOLDS     = 3

# ── Hyperparameter tuning ──────────────────────────────────
HP_POPULATION_SIZE = 15
HP_NUM_GENERATIONS = 8
HP_CV_FOLDS        = 3

# XGBoost hyperparameter bounds for GA tuning
HP_BOUNDS = {
    "n_estimators"  : (50,  400),
    "learning_rate" : (0.01, 0.30),
    "max_depth"     : (3,   10),
    "subsample"     : (0.5,  1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1,  10),
}

# Multicollinearity threshold — remove one from any pair above this
MULTICOLLINEARITY_THRESHOLD = 0.92

# ── App display settings ───────────────────────────────────
APP_TITLE       = "Freight Price Prediction System"
APP_CURRENCY    = "₹"
APP_PRICE_LABEL = "Freight Price (INR)"

# ============================================================
# SWITCHING BACK TO FICTIONAL DATASET:
# ============================================================
# DATASET_FILENAME    = "fictional_freight_dataset.csv"
# TARGET_COL          = "EUR"
# DISTANCE_COL        = "TOTAL_KM"
# DATETIME_COL        = "TIME_OF_ENTRY"
# DATE_COLS           = ["START_LOAD_DATE","END_LOAD_DATE",
#                         "START_DELIVERY_DATE","END_DELIVERY_DATE"]
# CATEGORICAL_COLS    = ["ORIGIN_COUNTRY","DESTINATION_COUNTRY",
#                         "VEHICLE_TYPE","CARRIER_ID","SHIPPER_ID"]
# COLS_TO_DROP_ALWAYS = ["EPALE"]
# BLACKLIST_FEATURES  = ["EUR","PRICE_PER_KM","TOTAL_PRICE"]
# APP_CURRENCY        = "€"
# APP_PRICE_LABEL     = "Freight Price (EUR)"
# ============================================================