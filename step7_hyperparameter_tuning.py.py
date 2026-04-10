# ============================================================
# STEP 7 — EVOLUTIONARY HYPERPARAMETER TUNING (XGBoost)
# ============================================================
# WHAT WE DO:
#   Use a Genetic Algorithm to find the best hyperparameters
#   for XGBoost using the GA-selected features from Step 6.
#
# HYPERPARAMETERS TUNED:
#   n_estimators     → number of boosting rounds (trees)
#   learning_rate    → step size shrinkage (eta)
#   max_depth        → maximum tree depth
#   subsample        → fraction of rows per tree
#   colsample_bytree → fraction of features per tree
#   min_child_weight → minimum sum of instance weight in a leaf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools, algorithms
import json, warnings
warnings.filterwarnings("ignore")
from config import (TARGET_COL, BLACKLIST_FEATURES,
                    HP_POPULATION_SIZE, HP_NUM_GENERATIONS,
                    HP_CV_FOLDS, HP_BOUNDS)

# ── 1. Load data and best features from Step 6 ────────────
df = pd.read_csv("data/processed_data.csv", low_memory=False)
freq_df   = pd.read_csv("data/feature_selection_frequency.csv")
top_folds = freq_df['Times_Selected'].max()
FEATURES  = freq_df[
    freq_df['Times_Selected'] == top_folds
]['Feature'].tolist()
FEATURES  = [f for f in FEATURES if f not in BLACKLIST_FEATURES]

print(f"Loaded: {df.shape[0]:,} rows")
print(f"Features selected in all {top_folds} folds: {FEATURES}")
print(f"Model: XGBoost")

X = df[FEATURES].values
y = df[TARGET_COL].values

# ── 2. DEAP setup ──────────────────────────────────────────
if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# XGBoost bounds from config:
# n_estimators, learning_rate, max_depth,
# subsample, colsample_bytree, min_child_weight
bounds_list = list(HP_BOUNDS.values())

def generate_individual():
    b = bounds_list
    return creator.Individual([
        np.random.randint(b[0][0], b[0][1]),    # n_estimators (int)
        np.random.uniform(b[1][0], b[1][1]),    # learning_rate (float)
        np.random.randint(b[2][0], b[2][1]),    # max_depth (int)
        np.random.uniform(b[3][0], b[3][1]),    # subsample (float)
        np.random.uniform(b[4][0], b[4][1]),    # colsample_bytree (float)
        np.random.randint(b[5][0], b[5][1]),    # min_child_weight (int)
    ])

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ── 3. Fitness evaluation ─────────────────────────────────
def eval_xgb(individual):
    b = bounds_list
    params = {
        "n_estimators"    : max(b[0][0], int(individual[0])),
        "learning_rate"   : float(np.clip(individual[1], b[1][0], b[1][1])),
        "max_depth"       : max(1, int(round(individual[2]))),
        "subsample"       : float(np.clip(individual[3], b[3][0], b[3][1])),
        "colsample_bytree": float(np.clip(individual[4], b[4][0], b[4][1])),
        "min_child_weight": max(1, int(round(individual[5]))),
        "random_state"    : 42,
        "verbosity"       : 0,
    }
    model = XGBRegressor(**params)
    tscv  = TimeSeriesSplit(n_splits=HP_CV_FOLDS)
    scores = []
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr],
                  eval_set=[(X[te], y[te])],
                  verbose=False)
        scores.append(mean_absolute_percentage_error(
            y[te], model.predict(X[te])))
    return (float(np.mean(scores)),)

# ── 4. Custom mutation ─────────────────────────────────────
def custom_mutation(individual, indpb):
    b = bounds_list
    int_indices = [0, 2, 5]   # n_estimators, max_depth, min_child_weight
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            lo, hi = b[i]
            if i in int_indices:
                individual[i] = int(np.clip(
                    individual[i] + np.random.randint(-30, 30), lo, hi))
            else:
                individual[i] = float(np.clip(
                    individual[i] + np.random.uniform(-0.05, 0.05), lo, hi))
    return individual,

toolbox.register("evaluate", eval_xgb)
toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("mate",     tools.cxUniform,     indpb=0.5)
toolbox.register("mutate",   custom_mutation,     indpb=0.3)

# ── 5. Run GA ──────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"HYPERPARAMETER TUNING — XGBoost")
print(f"  Population  : {HP_POPULATION_SIZE}")
print(f"  Generations : {HP_NUM_GENERATIONS}")
print(f"  CV folds    : {HP_CV_FOLDS}")
print(f"  Bounds      : {dict(zip(HP_BOUNDS.keys(), HP_BOUNDS.values()))}")
print(f"{'='*55}")

population   = toolbox.population(n=HP_POPULATION_SIZE)
hall_of_fame = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

population, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb=0.6, mutpb=0.3,
    ngen=HP_NUM_GENERATIONS,
    stats=stats, halloffame=hall_of_fame,
    verbose=True
)

# ── 6. Best hyperparameters ────────────────────────────────
best = hall_of_fame[0]
b    = bounds_list
best_params = {
    "n_estimators"    : int(best[0]),
    "learning_rate"   : round(float(best[1]), 6),
    "max_depth"       : int(round(best[2])),
    "subsample"       : round(float(best[3]), 6),
    "colsample_bytree": round(float(best[4]), 6),
    "min_child_weight": int(round(best[5])),
}

print(f"\n{'='*55}")
print("BEST XGBoost HYPERPARAMETERS:")
for k, v in best_params.items():
    print(f"  {k:<22}: {v}")
print(f"\n  Best MAPE: {best.fitness.values[0]*100:.2f}%")

# ── 7. Plot evolution ──────────────────────────────────────
gen_nums = [e['gen'] for e in logbook]
plt.figure(figsize=(10, 5))
plt.plot(gen_nums, [e['avg'] for e in logbook],
         label='Average MAPE', marker='o')
plt.plot(gen_nums, [e['min'] for e in logbook],
         label='Best MAPE', marker='s', linestyle='--')
plt.xlabel('Generation')
plt.ylabel('MAPE')
plt.title('XGBoost Hyperparameter Tuning — MAPE Evolution')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/hyperparameter_ga_evolution.png", dpi=150)
plt.show()
print("Plot saved → plots/hyperparameter_ga_evolution.png")

# ── 8. Save ───────────────────────────────────────────────
with open("data/best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

with open("data/best_features.json", "w") as f:
    json.dump(FEATURES, f, indent=2)

pd.DataFrame(logbook).to_csv("data/hyperparameter_logbook.csv", index=False)

print("\nSaved → data/best_hyperparameters.json")
print("Saved → data/best_features.json")
print("Step 7 complete — Ready for Step 8!")