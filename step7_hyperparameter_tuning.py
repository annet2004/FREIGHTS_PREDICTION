# ============================================================
# STEP 7 — EVOLUTIONARY HYPERPARAMETER TUNING
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools, algorithms
import json, warnings
warnings.filterwarnings("ignore")
from config import (TARGET_COL, BLACKLIST_FEATURES,
                    HP_POPULATION_SIZE, HP_NUM_GENERATIONS,
                    HP_CV_FOLDS, HP_BOUNDS)

df = pd.read_csv("data/processed_data.csv", low_memory=False)

# Load best features from Step 6
freq_df  = pd.read_csv("data/feature_selection_frequency.csv")
top_folds = freq_df['Times_Selected'].max()
FEATURES  = freq_df[freq_df['Times_Selected'] == top_folds]['Feature'].tolist()
FEATURES  = [f for f in FEATURES if f not in BLACKLIST_FEATURES]

print(f"Loaded: {df.shape[0]:,} rows")
print(f"Features selected in all {top_folds} folds: {FEATURES}")

X = df[FEATURES].values
y = df[TARGET_COL].values

if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def generate_individual():
    bounds = list(HP_BOUNDS.values())
    return creator.Individual([
        np.random.randint(*bounds[0]),
        np.random.uniform(*bounds[1]),
        np.random.randint(*bounds[2]),
        np.random.uniform(*bounds[3]),
        np.random.randint(*bounds[4]),
    ])

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_gb(individual):
    bounds = list(HP_BOUNDS.values())
    params = {
        'n_estimators'      : max(bounds[0][0], int(individual[0])),
        'learning_rate'     : float(np.clip(individual[1], *bounds[1])),
        'max_depth'         : max(1, int(round(individual[2]))),
        'subsample'         : float(np.clip(individual[3], *bounds[3])),
        'min_samples_split' : max(2, int(round(individual[4]))),
        'random_state'      : 42
    }
    model = GradientBoostingRegressor(**params)
    tscv  = TimeSeriesSplit(n_splits=HP_CV_FOLDS)
    scores = []
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        scores.append(mean_absolute_percentage_error(y[te], model.predict(X[te])))
    return (float(np.mean(scores)),)

def custom_mutation(individual, indpb):
    bounds = list(HP_BOUNDS.values())
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            lo, hi = bounds[i]
            if isinstance(lo, int):
                individual[i] = int(np.clip(individual[i] +
                                    np.random.randint(-50, 50), lo, hi))
            else:
                individual[i] = float(np.clip(individual[i] +
                                      np.random.uniform(-0.05, 0.05), lo, hi))
    return individual,

toolbox.register("evaluate", eval_gb)
toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("mate",     tools.cxUniform,     indpb=0.5)
toolbox.register("mutate",   custom_mutation, indpb=0.3)

print(f"\n{'='*50}")
print(f"HYPERPARAMETER TUNING GA")
print(f"  Population  : {HP_POPULATION_SIZE}")
print(f"  Generations : {HP_NUM_GENERATIONS}")
print(f"  CV folds    : {HP_CV_FOLDS}")
print(f"{'='*50}")

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

best = hall_of_fame[0]
bounds = list(HP_BOUNDS.values())
best_params = {
    'n_estimators'      : int(best[0]),
    'learning_rate'     : round(float(best[1]), 6),
    'max_depth'         : int(round(best[2])),
    'subsample'         : round(float(best[3]), 6),
    'min_samples_split' : int(round(best[4])),
}

print(f"\n{'='*50}")
print("BEST HYPERPARAMETERS:")
for k, v in best_params.items():
    print(f"  {k:<22}: {v}")
print(f"  Best MAPE: {best.fitness.values[0]*100:.2f}%")

# Plot evolution
gen_nums = [e['gen'] for e in logbook]
plt.figure(figsize=(10, 5))
plt.plot(gen_nums, [e['avg'] for e in logbook], label='Average MAPE', marker='o')
plt.plot(gen_nums, [e['min'] for e in logbook], label='Best MAPE',
         marker='s', linestyle='--')
plt.xlabel('Generation')
plt.ylabel('MAPE')
plt.title('MAPE Evolution — Hyperparameter Tuning')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/hyperparameter_ga_evolution.png", dpi=150)
plt.show()

# Save
with open("data/best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)
pd.DataFrame(logbook).to_csv("data/hyperparameter_logbook.csv", index=False)
# Save best features used
with open("data/best_features.json", "w") as f:
    json.dump(FEATURES, f, indent=2)

print("\nSaved → data/best_hyperparameters.json")
print("Saved → data/best_features.json")
print("Step 7 complete — Ready for Step 8!")