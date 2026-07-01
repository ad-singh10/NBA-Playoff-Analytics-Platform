import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression 
import joblib

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

    # Static Features
    "NET_RATING_DIFF",
    "OFF_RATING_DIFF",
    "DEF_RATING_DIFF",
    "PACE_DIFF",
    "TS_PCT_DIFF",
    "EFG_PCT_DIFF",
    "PIE_DIFF",

    # Last 10
    "HOME_LAST10_WIN_PCT",
    "AWAY_LAST10_WIN_PCT",
    "LAST10_DIFF",

    # Win Streak
    "HOME_WIN_STREAK",
    "AWAY_WIN_STREAK",
    "STREAK_DIFF",

    # Rest Days
    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "REST_DAY_DIFF",

    # Elo
    "HOME_ELO",
    "AWAY_ELO",
    "ELO_DIFF"

]

X = df[features]

y = df["HOME_WIN"]

# Train-test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#Info Print

print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print()

print("Training Samples :", X_train.shape)

print("Testing Samples  :", X_test.shape)

print()

print("Features Used")

for feature in features:

    print("-", feature)


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

logistic_model = LogisticRegression(

    random_state=42,

    max_iter=1000
)

logistic_model.fit(

    X_train_scaled,

    y_train

)

print("\n Logistic Regression Trained Successfully")


# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = logistic_model.predict(

    X_test_scaled

)

y_prob = logistic_model.predict_proba(

    X_test_scaled

)[:,1]


joblib.dump(

    logistic_model,

    "models/logistic_regression.pkl"

)

joblib.dump(

    scaler,

    "models/scaler.pkl"

)

print("Model Saved")


from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# RANDOM FOREST
# ==========================================================

random_forest_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42

)

random_forest_model.fit(

    X_train,

    y_train

)

print("\n✓ Random Forest Trained Successfully")

joblib.dump(

    random_forest_model,

    "models/random_forest.pkl"

)

print("✓ Random Forest Saved")


# ==========================================================
# XGBOOST
# ==========================================================

xgboost_model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=4,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"

)

xgboost_model.fit(

    X_train,

    y_train

)

print("\n✓ XGBoost Trained Successfully")

joblib.dump(

    xgboost_model,

    "models/xgboost.pkl"

)

print("✓ XGBoost Saved")

