# ============================================================
# STEP 6 — EVOLUTIONARY FEATURE SELECTION (GA)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools, algorithms
from collections import Counter
import random, json, warnings
warnings.filterwarnings("ignore")
from config import (TARGET_COL, BLACKLIST_FEATURES,
                    GA_POPULATION_SIZE, GA_NUM_GENERATIONS,
                    GA_OUTER_FOLDS, GA_INNER_FOLDS)

df = pd.read_csv("data/processed_data.csv", low_memory=False)
selected = pd.read_csv("data/selected_features.csv")
corr_features = [f for f in selected['feature'].tolist()
                 if f not in BLACKLIST_FEATURES]

print(f"Loaded: {df.shape[0]:,} rows")
print(f"\n{len(corr_features)} features from Step 5 going into GA:")
for f in corr_features: print(f"  → {f}")

X = df[corr_features]
y = df[TARGET_COL]

outer_cv     = TimeSeriesSplit(n_splits=GA_OUTER_FOLDS)
num_features = X.shape[1]

if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool",  random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",       tools.cxUniform,     indpb=0.5)
toolbox.register("mutate",     tools.mutFlipBit,    indpb=0.1)
toolbox.register("select",     tools.selTournament, tournsize=3)

def evalFitness(individual, X_train, y_train, X_valid, y_valid):
    idx = [i for i, b in enumerate(individual) if b == 1]
    if not idx: return 1e6,
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train.iloc[:, idx], y_train)
    return mean_absolute_percentage_error(y_valid,
           model.predict(X_valid.iloc[:, idx])),

mape_scores, selected_all = [], []

print(f"\n{'='*50}")
print(f"STARTING GA  |  {num_features} features  |  "
      f"pop={GA_POPULATION_SIZE}  gen={GA_NUM_GENERATIONS}")
print(f"{'='*50}")

for fold, (tr, te) in enumerate(outer_cv.split(X), 1):
    print(f"\nProcessing Fold {fold}...")
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    inner_cv = TimeSeriesSplit(n_splits=GA_INNER_FOLDS)

    def fw(ind):
        scores = []
        for i, v in inner_cv.split(Xtr):
            scores.append(evalFitness(ind, Xtr.iloc[i], ytr.iloc[i],
                                      Xtr.iloc[v], ytr.iloc[v])[0])
        return np.mean(scores),

    toolbox.register("evaluate", fw)
    pop = toolbox.population(n=GA_POPULATION_SIZE)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                        ngen=GA_NUM_GENERATIONS, verbose=False)

    best = tools.selBest(pop, 1)[0]
    idx  = [i for i, b in enumerate(best) if b == 1]
    names = X.columns[idx].tolist()
    selected_all.append(names)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr.iloc[:, idx], ytr)
    mape = mean_absolute_percentage_error(yte, model.predict(Xte.iloc[:, idx]))
    print(f"  Features: {names}")
    print(f"  MAPE = {mape:.4f} ({mape*100:.2f}%)")
    mape_scores.append(mape)

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
for i, s in enumerate(mape_scores, 1):
    print(f"  Fold {i}: MAPE = {s*100:.2f}%")
print(f"  Average MAPE: {np.mean(mape_scores)*100:.2f}%")

counts = Counter([f for fold in selected_all for f in fold])
freq_df = pd.DataFrame(counts.items(),
                       columns=['Feature','Times_Selected']
                       ).sort_values('Times_Selected', ascending=False)
print("\nFeature frequency:")
for _, row in freq_df.iterrows():
    bar = "█" * int(row['Times_Selected'] * 2)
    print(f"  {row['Feature']:<45} {row['Times_Selected']}/{GA_OUTER_FOLDS}  {bar}")

with open("data/selected_features_per_fold.json", "w") as f:
    json.dump(selected_all, f, indent=2)
freq_df.to_csv("data/feature_selection_frequency.csv", index=False)
print("\nSaved → data/selected_features_per_fold.json")
print("Step 6 complete — Ready for Step 7!")