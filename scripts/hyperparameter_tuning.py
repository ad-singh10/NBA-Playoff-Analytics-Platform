import pandas as pd
import joblib

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/processed/final_dataset_dynamic.csv"
)

# ==========================================================
# FEATURES
# ==========================================================

features = [

    "NET_RATING_DIFF",
    "OFF_RATING_DIFF",
    "DEF_RATING_DIFF",
    "PACE_DIFF",
    "TS_PCT_DIFF",
    "EFG_PCT_DIFF",
    "PIE_DIFF",

    "HOME_LAST10_WIN_PCT",
    "AWAY_LAST10_WIN_PCT",
    "LAST10_DIFF",

    "HOME_WIN_STREAK",
    "AWAY_WIN_STREAK",
    "STREAK_DIFF",

    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "REST_DAY_DIFF",

    "HOME_ELO",
    "AWAY_ELO",
    "ELO_DIFF"

]

X = df[features]

y = df["HOME_WIN"]

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf_pipeline = Pipeline([

    ("model",
     RandomForestClassifier(
         random_state=42
     ))

])

rf_params = {

    "model__n_estimators":[200,300,500],

    "model__max_depth":[5,10,15],

    "model__min_samples_split":[2,5],

    "model__min_samples_leaf":[1,2]

}

rf_grid = GridSearchCV(

    estimator=rf_pipeline,

    param_grid=rf_params,

    cv=cv,

    scoring="roc_auc",

    n_jobs=-1,

    verbose=2

)

rf_grid.fit(X,y)

print("\n")
print("="*60)
print("BEST RANDOM FOREST")
print("="*60)

print("Best ROC AUC")
print(rf_grid.best_score_)

print("\nBest Parameters")
print(rf_grid.best_params_)

joblib.dump(

    rf_grid.best_estimator_,

    "models/random_forest_tuned.pkl"

)

print("\n✓ Tuned Random Forest Saved")

# ==========================================================
# XGBOOST
# ==========================================================

xgb_pipeline = Pipeline([

    ("model",

     XGBClassifier(

        random_state=42,

        eval_metric="logloss"

     ))

])

xgb_params = {

    "model__n_estimators":[200,300,500],

    "model__learning_rate":[0.01,0.05,0.1],

    "model__max_depth":[3,4,5],

    "model__subsample":[0.8,1.0],

    "model__colsample_bytree":[0.8,1.0]

}

xgb_grid = GridSearchCV(

    estimator=xgb_pipeline,

    param_grid=xgb_params,

    cv=cv,

    scoring="roc_auc",

    n_jobs=-1,

    verbose=2

)

xgb_grid.fit(X,y)

print("\n")
print("="*60)
print("BEST XGBOOST")
print("="*60)

print("Best ROC AUC")
print(xgb_grid.best_score_)

print("\nBest Parameters")
print(xgb_grid.best_params_)

joblib.dump(

    xgb_grid.best_estimator_,

    "models/xgboost_tuned.pkl"

)

print("\n Tuned XGBoost Saved")